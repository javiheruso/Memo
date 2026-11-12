"""End-to-end personal-lookup boundary and narrative continuity.

Narrative mentions, self-disclosures and general chat must never be promoted
into personal retrieval, entity resolution or identity knowledge gaps. Explicit
personal requests, natural indirect phrasings, STT-style lowercase transcripts,
verified carry-over, memory recall, ambiguity selection and the sensitive-topic
policy must keep working.

These tests assert observable behavior: intent, response text, persisted
sources, knowledge gaps and claim status. They do not stop at intent labels.
"""

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import ClaimStatus, Intent
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.deterministic import DeterministicResponder
from app.models import Claim, Fact, KnowledgeGap, Memory
from app.repositories.profile import ConversationProfileRepository
from app.schemas.interpreter import InterpreterSuggestion
from app.schemas.profile import ConversationProfileCreate, SensitiveTopicCreate
from app.services.conversation import ConversationService
from tests.fixtures.universe import SeedUniverse


def _service(
    session: AsyncSession,
    *,
    interpreter=None,
) -> ConversationService:
    return ConversationService(
        session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
        interpreter=interpreter,
    )


def _source_keys(result) -> set[tuple[str, str]]:
    return {(source.entity_type, str(source.entity_id)) for source in result.sources}


async def _gap_count(session: AsyncSession) -> int:
    return (
        await session.execute(select(func.count()).select_from(KnowledgeGap))
    ).scalar_one()


async def _claims(session: AsyncSession, conversation_id: uuid.UUID) -> list[Claim]:
    return list(
        (
            await session.execute(
                select(Claim).where(Claim.conversation_id == conversation_id)
            )
        ).scalars()
    )


class FixedInterpreter:
    """Interpreter stub that always proposes a personal person query."""

    max_input_chars = 400
    history_turns = 4
    max_history_chars = 1200

    def __init__(self, term: str = "Carmen") -> None:
        self._term = term
        self.calls = 0

    async def suggest(self, text: str, *, history=()):
        self.calls += 1
        return InterpreterSuggestion(intent="person_query", term=self._term)


async def _activate_sensitive_profile(
    session: AsyncSession, person_id: uuid.UUID
) -> None:
    await ConversationProfileRepository(session).create(
        ConversationProfileCreate(
            person_id=person_id,
            preferred_name="Carmen",
            positive_topics=["costura"],
            sensitive_topics=[
                SensitiveTopicCreate(topic="fallecimiento", triggers=["murió"])
            ],
            is_active=True,
        )
    )


# --------------------------------------------------------------------------
# Narrative mentions stay conversational
# --------------------------------------------------------------------------


async def test_narrative_mentions_never_resolve_people_or_open_identity_gaps(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    conversation_id = None
    results = []

    for text in (
        "Ayer vi a María en el mercado.",
        "Eustaquio viene mañana a visitarme.",
        "Mi marido se llamaba Miguel.",
        "El domingo viene mi hija.",
    ):
        result = await service.handle_user_message(
            text, conversation_id=conversation_id
        )
        conversation_id = result.conversation_id
        results.append(result)

    for result in results:
        assert result.intent == Intent.general_chat
        assert result.sources == []
        assert result.gap is None
        assert result.gap_created is False

    assert await _gap_count(db_session) == 0

    # The same unknown person is only a lookup when explicitly asked about.
    explicit = await service.handle_user_message(
        "¿Quién era Eustaquio?", conversation_id=conversation_id
    )
    assert explicit.intent == Intent.person_query
    assert explicit.sources == []
    assert explicit.gap is not None
    assert explicit.gap_created is True
    assert await _gap_count(db_session) == 1


async def test_narrative_claims_stay_pending_candidates_without_promotion(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    facts_before = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()
    memories_before = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()
    service = _service(db_session)

    result = await service.handle_user_message(
        "Ayer estuve con María en el mercado."
    )

    assert result.intent == Intent.general_chat
    assert result.sources == []
    assert result.gap is None

    claims = await _claims(db_session, result.conversation_id)
    assert len(claims) == 1
    assert claims[0].status is ClaimStatus.candidate
    assert claims[0].original_text.startswith("Ayer estuve con María")
    assert claims[0].canonical_subject is None
    assert claims[0].canonical_predicate is None
    assert claims[0].canonical_object is None

    facts_after = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()
    memories_after = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()
    assert facts_after == facts_before
    assert memories_after == memories_before
    assert await _gap_count(db_session) == 0


async def test_narrative_is_not_promoted_even_when_interpreter_proposes_it(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    interpreter = FixedInterpreter("Carmen")
    service = _service(db_session, interpreter=interpreter)

    result = await service.handle_user_message("Ayer estuve pensando en Carmen")

    assert result.intent == Intent.general_chat
    assert result.sources == []
    assert result.gap is None
    assert interpreter.calls == 0
    assert await _gap_count(db_session) == 0


# --------------------------------------------------------------------------
# Explicit, indirect and STT requests still retrieve
# --------------------------------------------------------------------------


async def test_direct_and_indirect_personal_requests_keep_working(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)

    direct = await service.handle_user_message("Háblame de Carmen")
    assert direct.intent == Intent.person_query
    assert ("person", str(seed.person_ids["carmen"])) in _source_keys(direct)
    assert direct.gap is None

    indirect = await service.handle_user_message(
        "Me gustaría saber algo sobre Antonio"
    )
    assert indirect.intent == Intent.person_query
    assert ("person", str(seed.person_ids["antonio"])) in _source_keys(indirect)
    assert indirect.gap is None

    curious = await service.handle_user_message("Tengo curiosidad por María")
    assert curious.intent == Intent.person_query
    assert ("person", str(seed.person_ids["maria"])) in _source_keys(curious)
    assert curious.gap is None


async def test_lowercase_stt_transcripts_without_punctuation_keep_working(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    conversation_id = None

    person = await service.handle_user_message("hablame de carmen")
    conversation_id = person.conversation_id
    assert person.intent == Intent.person_query
    assert ("person", str(seed.person_ids["carmen"])) in _source_keys(person)

    fact = await service.handle_user_message(
        "que le gusta a antonio", conversation_id=conversation_id
    )
    assert fact.intent == Intent.fact_query
    assert "Fishing" in fact.answer
    assert "fact" in {source.entity_type for source in fact.sources}

    event = await service.handle_user_message(
        "cuando viene maria", conversation_id=conversation_id
    )
    assert event.intent == Intent.event_query
    assert ("event", str(seed.event_ids["maria_sunday_visit"])) in _source_keys(
        event
    )

    # A lowercase term that matches no stored person is not a reliable identity
    # reference: the turn stays conversational and opens no gap.
    unknown = await service.handle_user_message(
        "cuando viene eustaquio", conversation_id=conversation_id
    )
    assert unknown.intent == Intent.general_chat
    assert unknown.sources == []
    assert unknown.gap is None
    assert await _gap_count(db_session) == 0


# --------------------------------------------------------------------------
# Ambiguity and pending candidate selection
# --------------------------------------------------------------------------


async def test_ambiguous_name_and_candidate_selection_survive_narrative_noise(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)

    ambiguous = await service.handle_user_message("¿Quién era Paco?")
    assert ambiguous.intent == Intent.person_query
    assert ambiguous.ambiguity is not None
    assert ambiguous.gap is None
    assert {source.usage_type for source in ambiguous.sources} == {"candidate"}

    selection = await service.handle_user_message(
        "el panadero", conversation_id=ambiguous.conversation_id
    )
    assert selection.intent == Intent.person_query
    assert selection.ambiguity is None
    assert ("person", str(seed.person_ids["paco_garcia"])) in _source_keys(
        selection
    )
    assert selection.gap is None

    fresh = await service.handle_user_message("¿Quién era Paco?")
    assert fresh.ambiguity is not None

    narrative = await service.handle_user_message(
        "Ayer vi a Paco García en el mercado.",
        conversation_id=fresh.conversation_id,
    )
    assert narrative.intent == Intent.general_chat
    assert narrative.ambiguity is None
    assert narrative.sources == []
    assert narrative.gap is None
    assert ("person", str(seed.person_ids["paco_garcia"])) not in _source_keys(
        narrative
    )


# --------------------------------------------------------------------------
# Continuity: verified carry, acceptance, topic change and memory recall
# --------------------------------------------------------------------------


async def test_verified_pronoun_carry_memory_recall_and_acceptance(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)

    first = await service.handle_user_message("Háblame de Carmen")
    assert first.intent == Intent.person_query

    pronoun = await service.handle_user_message(
        "¿Y ella?", conversation_id=first.conversation_id
    )
    assert pronoun.intent == Intent.person_query
    assert ("person", str(seed.person_ids["carmen"])) in _source_keys(pronoun)

    memory = await service.handle_user_message(
        "Do you remember when we went dancing?",
        conversation_id=first.conversation_id,
    )
    assert memory.intent == Intent.memory_query
    assert memory.sources
    assert all(source.entity_type == "memory" for source in memory.sources)

    accepted = await service.handle_user_message(
        "Sí.", conversation_id=first.conversation_id
    )
    assert accepted.intent == Intent.memory_query
    assert accepted.sources
    assert all(source.entity_type == "memory" for source in accepted.sources)


async def test_narrative_does_not_establish_pronoun_carry_or_topic_context(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)

    first = await service.handle_user_message("Háblame de Carmen")
    assert first.sources

    narrative = await service.handle_user_message(
        "Ayer vi a Carmen en el mercado.",
        conversation_id=first.conversation_id,
    )
    assert narrative.intent == Intent.general_chat
    assert narrative.sources == []

    pronoun = await service.handle_user_message(
        "¿Y ella?", conversation_id=narrative.conversation_id
    )
    assert pronoun.intent == Intent.general_chat
    assert pronoun.sources == []
    assert ("person", str(seed.person_ids["carmen"])) not in _source_keys(pronoun)

    changed = await service.handle_user_message(
        "Hoy hace buen día para pasear.",
        conversation_id=narrative.conversation_id,
    )
    assert changed.intent == Intent.general_chat
    assert changed.sources == []


# --------------------------------------------------------------------------
# Sensitive boundary remains intact
# --------------------------------------------------------------------------


async def test_sensitive_boundary_and_narrative_suppression_stay_intact(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    await _activate_sensitive_profile(db_session, seed.person_ids["carmen"])
    service = _service(db_session)

    sensitive = await service.handle_user_message("Antonio murió ayer")
    assert "No quiero inventar detalles" in sensitive.answer
    assert sensitive.sources == []
    assert sensitive.gap is None

    narrative = await service.handle_user_message(
        "Ayer estuve pensando en Antonio",
        conversation_id=sensitive.conversation_id,
    )
    assert narrative.intent == Intent.general_chat
    assert narrative.sources == []
    assert "murió" not in narrative.answer.lower()
    assert "fallecimiento" not in narrative.answer.lower()

    safe = await service.handle_user_message(
        "Háblame de Carmen", conversation_id=narrative.conversation_id
    )
    assert safe.intent == Intent.person_query
    assert ("person", str(seed.person_ids["carmen"])) in _source_keys(safe)
