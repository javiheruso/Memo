"""Provisional conversational referent continuity.

The referent is a pure, per-turn projection over user-authored conversation
turns. These tests prove that it survives multiple turns, that corrections win,
that it never becomes stored data or a source, that an explicit lookup does not
silently select a different registered person, and that an unresolved,
sensitive, ambiguous, expired or explicitly changed context drops it.

They also prove the model prompt receives the referent only in a separate
unverified conversation block with no entity ids or database details.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.conversation_window import ConversationWindow, TurnSnapshot
from app.domain.enums import ClaimStatus, Intent
from app.domain.provisional_referent import (
    ProvisionalReferent,
    project_provisional_referent,
    surfaces_match,
)
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.base import Responder, Response
from app.llm.deterministic import DeterministicResponder
from app.llm.openai_compatible import OpenAICompatibleResponder
from app.llm.prompt_builder import allowed_sources, build_prompt
from app.models import (
    Claim,
    Fact,
    KnowledgeGap,
    Memory,
    Relationship,
    ResponseSource,
)
from app.repositories.profile import ConversationProfileRepository
from app.retrieval.context_builder import AnswerContext
from app.schemas.profile import ConversationProfileCreate, SensitiveTopicCreate
from app.services.conversation import ConversationService
from tests.fixtures.universe import SeedUniverse


def _turn(text: str, speaker: str = "user") -> TurnSnapshot:
    return TurnSnapshot(
        message_id=uuid.uuid4(),
        speaker=speaker,
        text=text,
        intent="general_chat",
        created_at=datetime.now(UTC),
    )


def _window(*turns: TurnSnapshot) -> ConversationWindow:
    return ConversationWindow(conversation_id=uuid.uuid4(), turns=tuple(turns))


class CapturingResponder(Responder):
    """Capture the exact context given to a responder, then answer safely."""

    def __init__(self) -> None:
        self._inner = DeterministicResponder()
        self.calls: list[dict] = []

    async def respond(
        self,
        context: AnswerContext,
        *,
        message: str | None = None,
        history=(),
        style=None,
    ) -> Response:
        self.calls.append(
            {
                "context": context,
                "message": message,
                "history": tuple(history),
                "style": style,
            }
        )
        return await self._inner.respond(
            context, message=message, history=history, style=style
        )


def _service(
    session: AsyncSession, responder: Responder | None = None
) -> ConversationService:
    return ConversationService(
        session,
        responder or DeterministicResponder(),
        embedder=DeterministicEmbedder(),
    )


async def _claims(
    session: AsyncSession, conversation_id: uuid.UUID
) -> list[Claim]:
    return list(
        (
            await session.execute(
                select(Claim).where(Claim.conversation_id == conversation_id)
            )
        ).scalars()
    )


async def _count(session: AsyncSession, model) -> int:
    return (
        await session.execute(select(func.count()).select_from(model))
    ).scalar_one()


# --------------------------------------------------------------------------
# Pure projection
# --------------------------------------------------------------------------


def test_projection_uses_only_user_turns_and_survives_multiple_turns() -> None:
    window = _window(
        _turn("María, mi vecina, vino a casa."),
        # Assistant text must never establish or update a referent.
        _turn("Antonio, mi vecino, es panadero.", speaker="assistant"),
        _turn("Es muy simpática."),
        _turn("Te escucho.", speaker="assistant"),
        _turn("Ella siempre me cuenta historias."),
    )

    referent = project_provisional_referent(
        window, current_message="Mi vecina me ayuda con la compra."
    )

    assert referent is not None
    assert referent.surface == "María"
    assert referent.description == "vecina"
    assert referent.verified_person_id is None
    assert referent.verified_display_name is None


def test_assistant_text_alone_never_establishes_a_referent() -> None:
    window = _window(
        _turn("María es tu vecina.", speaker="assistant"),
        _turn("Te escucho.", speaker="assistant"),
    )

    assert project_provisional_referent(window) is None


def test_correction_supersedes_the_earlier_interpretation() -> None:
    window = _window(
        _turn("María es mi hija."),
        _turn("No, perdón, María no es mi hija, es mi vecina."),
        _turn("Ella vive al lado."),
    )

    referent = project_provisional_referent(window)

    assert referent is not None
    assert referent.surface == "María"
    assert referent.description == "vecina"
    assert referent.origin_text.startswith("No, perdón, María")


def test_clear_topic_shift_drops_the_referent() -> None:
    window = _window(
        _turn("María, mi vecina, vino a casa."),
        _turn("Cambiando de tema, ¿me recomiendas una película?"),
    )

    assert project_provisional_referent(window) is None


def test_explicit_lookup_of_a_different_person_drops_the_referent() -> None:
    def lookup_term_for(text: str) -> str | None:
        return "Antonio" if "Antonio" in text else None

    window = _window(
        _turn("María, mi vecina, vino a casa."),
        _turn("¿Quién es Antonio?"),
    )

    assert (
        project_provisional_referent(
            window, lookup_term_for=lookup_term_for
        )
        is None
    )


def test_sensitive_turn_ambiguity_and_window_expiry_block_carry() -> None:
    intro = _turn("María, mi vecina, vino a casa.")
    sensitive = _turn("Mi vecina murió el año pasado.")

    assert (
        project_provisional_referent(
            _window(intro, sensitive),
            is_sensitive=lambda text: "murió" in text,
        )
        is None
    )
    assert (
        project_provisional_referent(_window(intro), ambiguity_pending=True)
        is None
    )
    # The introducing turn left the bounded window: the projection cannot see
    # it, so nothing is carried.
    assert project_provisional_referent(_window(_turn("Ella es simpática."))) is None


def test_surfaces_match_whole_tokens_only() -> None:
    assert surfaces_match("María", "María Gómez")
    assert surfaces_match("maria", "María")
    assert not surfaces_match("María", "Mariana")


def test_verified_identity_is_a_separate_optional_attachment() -> None:
    referent = ProvisionalReferent(
        surface="María", description="vecina", origin_text="María, mi vecina."
    )
    person_id = uuid.uuid4()
    verified = referent.with_verified_identity(
        person_id=person_id, display_name="María Gómez"
    )

    assert referent.verified_person_id is None
    assert verified.verified_person_id == person_id
    assert verified.surface == referent.surface


# --------------------------------------------------------------------------
# Prompt separation
# --------------------------------------------------------------------------


def test_prompt_renders_only_the_unverified_block_without_ids() -> None:
    referent = ProvisionalReferent(
        surface="María",
        description="vecina",
        origin_text="María, mi vecina, vino a casa.",
    )
    context = AnswerContext(
        intent=Intent.general_chat, provisional_referent=referent
    )

    prompt = build_prompt(context, message="Ella me ayuda mucho.")

    assert "PROVISIONAL CONVERSATION REFERENT" in prompt.user
    assert "unverified, conversation-only; NOT stored data" in prompt.user
    assert "Surface name or expression: María" in prompt.user
    assert "Conversational description or relationship: vecina" in prompt.user
    assert "Never cite it as a source" in prompt.user
    assert allowed_sources(context) == {}
    # The provisional block is not an entity and never carries ids.
    assert str(uuid.uuid4()) not in prompt.user
    assert "Allowed source ids you may reference:\n\nUNTRUSTED" not in prompt.user
    assert "Allowed source ids you may reference:" in prompt.user
    allowed_section = prompt.user.split("Allowed source ids you may reference:", 1)[1]
    assert allowed_section.strip() == ""


async def test_model_request_prompt_has_no_source_ids_or_database_details(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "Gracias por contármelo.",
                                    "sources": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        responder = OpenAICompatibleResponder(
            base_url="http://llm.test",
            model="test-model",
            client=client,
        )
        service = _service(db_session, responder)
        result = await service.handle_user_message(
            "María, mi vecina, vino a casa y estuvimos charlando."
        )

    assert result.sources == []
    assert captured, "the real responder must have built one prompt"
    prompt = captured[0]["messages"][1]["content"]
    assert "PROVISIONAL CONVERSATION REFERENT" in prompt
    assert "María" in prompt
    assert "vecina" in prompt
    # The registered daughter María exists in storage, but its id and any
    # database detail must not leak into the provisional block.
    assert str(seed.person_ids["maria"]) not in prompt
    assert "María Gómez" not in prompt
    assert "Daughter of Carmen" not in prompt
    allowed_section = prompt.split("Allowed source ids you may reference:", 1)[1]
    assert allowed_section.strip() == ""


# --------------------------------------------------------------------------
# Multi-turn integration
# --------------------------------------------------------------------------


async def test_neighbor_maria_conflict_asks_identity_without_lookup_or_gap(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture = CapturingResponder()
    service = _service(db_session, capture)
    facts_before = await _count(db_session, Fact)
    memories_before = await _count(db_session, Memory)
    relationships_before = await _count(db_session, Relationship)

    first = await service.handle_user_message(
        "María, mi vecina, vino a casa y estuvimos charlando."
    )
    assert first.intent == Intent.general_chat
    assert first.sources == []
    assert first.gap is None
    assert first.gap_created is False
    context = capture.calls[-1]["context"]
    assert context.provisional_referent is not None
    assert context.provisional_referent.surface == "María"
    assert context.provisional_referent.description == "vecina"
    # The provisional referent is not an entity item.
    assert context.person is None
    assert context.candidates == []
    assert allowed_sources(context) == {}

    second = await service.handle_user_message(
        "Es muy simpática y siempre me ayuda.",
        conversation_id=first.conversation_id,
    )
    assert second.sources == []
    assert (
        capture.calls[-1]["context"].provisional_referent.description == "vecina"
    )

    calls_before_conflict = len(capture.calls)
    conflict = await service.handle_user_message(
        "¿Qué sabes de María?", conversation_id=first.conversation_id
    )
    # "¿Qué sabes de X?" is an explicit personal lookup; the router may label
    # it a person or fact query, and both must not silently select the
    # registered person.
    assert conflict.intent in (Intent.person_query, Intent.fact_query)
    assert conflict.sources == []
    assert conflict.gap is None
    assert conflict.gap_created is False
    assert "Para no confundirme" in conflict.answer
    assert "María" in conflict.answer
    assert "vecina" in conflict.answer
    # The pipeline asked a clarification instead of running retrieval or a
    # responder call.
    assert len(capture.calls) == calls_before_conflict
    assert str(seed.person_ids["maria"]) not in conflict.answer

    # No source row was invented for any narrative or clarification turn.
    assert await _count(db_session, ResponseSource) == 0
    assert await _count(db_session, KnowledgeGap) == 0
    assert await _count(db_session, Fact) == facts_before
    assert await _count(db_session, Memory) == memories_before
    assert await _count(db_session, Relationship) == relationships_before

    claims = await _claims(db_session, first.conversation_id)
    assert claims, "the narrative turn should produce candidate claims"
    assert all(claim.status is ClaimStatus.candidate for claim in claims)
    assert all(claim.canonical_subject is None for claim in claims)


async def test_correction_updates_referent_and_pronoun_keeps_latest(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture = CapturingResponder()
    service = _service(db_session, capture)

    first = await service.handle_user_message("María es mi hija.")
    assert (
        capture.calls[-1]["context"].provisional_referent.description == "hija"
    )

    corrected = await service.handle_user_message(
        "No, perdón, María no es mi hija, es mi vecina.",
        conversation_id=first.conversation_id,
    )
    assert corrected.sources == []
    assert (
        capture.calls[-1]["context"].provisional_referent.description == "vecina"
    )

    follow_up = await service.handle_user_message(
        "Ella vive al lado y me ayuda.",
        conversation_id=first.conversation_id,
    )
    assert follow_up.sources == []
    assert (
        capture.calls[-1]["context"].provisional_referent.description == "vecina"
    )


async def test_pronouns_and_indirect_references_survive_beyond_immediate_turn(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture = CapturingResponder()
    service = _service(db_session, capture)

    first = await service.handle_user_message(
        "Tengo una vecina que se llama María."
    )
    assert capture.calls[-1]["context"].provisional_referent.surface == "María"

    for text in (
        "Ella es muy amable.",
        "Ayer tomamos café juntas.",
        "Mi vecina me ayuda con la compra.",
    ):
        result = await service.handle_user_message(
            text, conversation_id=first.conversation_id
        )
        assert result.sources == []
        referent = capture.calls[-1]["context"].provisional_referent
        assert referent is not None
        assert referent.surface == "María"
        assert referent.description == "vecina"


async def test_clear_topic_shift_and_explicit_new_subject_drop_the_referent(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture = CapturingResponder()
    service = _service(db_session, capture)

    first = await service.handle_user_message(
        "María, mi vecina, vino a casa."
    )
    assert capture.calls[-1]["context"].provisional_referent is not None

    shifted = await service.handle_user_message(
        "Cambiando de tema, ¿me recomiendas una película?",
        conversation_id=first.conversation_id,
    )
    assert shifted.sources == []
    assert capture.calls[-1]["context"].provisional_referent is None

    # A later turn does not revive the dropped referent.
    await service.handle_user_message(
        "Ella me ayuda mucho.", conversation_id=first.conversation_id
    )
    assert capture.calls[-1]["context"].provisional_referent is None

    # An explicit lookup of a different registered person is a normal lookup,
    # not a clarification, and clears the provisional carry.
    other = await service.handle_user_message(
        "¿Quién es Antonio?", conversation_id=first.conversation_id
    )
    assert other.intent == Intent.person_query
    assert "Para no confundirme" not in other.answer
    assert any(
        source.entity_type == "person"
        and source.entity_id == seed.person_ids["antonio"]
        for source in other.sources
    )
    await service.handle_user_message(
        "Él era muy simpático.", conversation_id=first.conversation_id
    )
    assert capture.calls[-1]["context"].provisional_referent is None


async def test_normal_explicit_lookup_and_existing_ambiguity_still_work(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)

    lookup = await service.handle_user_message("¿Quién es María?")
    assert lookup.intent == Intent.person_query
    assert "Para no confundirme" not in lookup.answer
    assert any(
        source.entity_type == "person"
        and source.entity_id == seed.person_ids["maria"]
        for source in lookup.sources
    )

    ambiguous = await service.handle_user_message(
        "¿Quién era Gómez?", conversation_id=lookup.conversation_id
    )
    assert ambiguous.intent == Intent.person_query
    assert ambiguous.ambiguity is not None
    assert len(ambiguous.ambiguity.candidates) > 1
    assert ambiguous.sources
    assert all(source.usage_type == "candidate" for source in ambiguous.sources)


async def test_unknown_conversational_person_stays_unstored_and_lookup_is_unchanged(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture = CapturingResponder()
    service = _service(db_session, capture)

    first = await service.handle_user_message(
        "Eustaquio, mi vecino, es muy amable."
    )
    assert first.intent == Intent.general_chat
    assert first.sources == []
    assert first.gap is None
    assert capture.calls[-1]["context"].provisional_referent.surface == "Eustaquio"

    follow_up = await service.handle_user_message(
        "Él siempre me saluda.", conversation_id=first.conversation_id
    )
    assert follow_up.sources == []
    assert follow_up.gap is None
    assert (
        capture.calls[-1]["context"].provisional_referent.surface == "Eustaquio"
    )

    # Existing unknown-person behavior is unchanged: an explicit proper-name
    # lookup still opens one knowledge gap and cites no source.
    explicit = await service.handle_user_message(
        "¿Quién es Eustaquio?", conversation_id=first.conversation_id
    )
    assert explicit.intent == Intent.person_query
    assert explicit.sources == []
    assert explicit.gap is not None
    assert explicit.gap_created is True
    assert await _count(db_session, KnowledgeGap) == 1


async def test_sensitive_context_clears_the_referent(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    await ConversationProfileRepository(db_session).create(
        ConversationProfileCreate(
            person_id=seed.person_ids["carmen"],
            preferred_name="Carmen",
            sensitive_topics=[
                SensitiveTopicCreate(topic="fallecimiento", triggers=["murió"])
            ],
            is_active=True,
        )
    )
    capture = CapturingResponder()
    service = _service(db_session, capture)

    first = await service.handle_user_message(
        "María, mi vecina, vino a casa."
    )
    assert capture.calls[-1]["context"].provisional_referent is not None

    sensitive = await service.handle_user_message(
        "Mi vecina murió el año pasado.",
        conversation_id=first.conversation_id,
    )
    assert sensitive.sources == []
    assert sensitive.gap is None
    assert capture.calls[-1]["context"].provisional_referent is None

    after = await service.handle_user_message(
        "Ella era muy buena.", conversation_id=first.conversation_id
    )
    assert after.sources == []
    assert capture.calls[-1]["context"].provisional_referent is None


async def test_ambiguity_blocks_provisional_carry(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture = CapturingResponder()
    service = _service(db_session, capture)

    first = await service.handle_user_message(
        "María, mi vecina, vino a casa."
    )
    assert capture.calls[-1]["context"].provisional_referent is not None

    ambiguous = await service.handle_user_message(
        "¿Quién era Gómez?", conversation_id=first.conversation_id
    )
    assert ambiguous.ambiguity is not None
    assert ambiguous.sources
    assert all(source.usage_type == "candidate" for source in ambiguous.sources)

    after = await service.handle_user_message(
        "Ella es simpática.", conversation_id=first.conversation_id
    )
    assert after.sources == []
    assert capture.calls[-1]["context"].provisional_referent is None
