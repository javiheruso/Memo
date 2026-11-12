"""Ported V2 conversation-continuity policy and its direct contract tests.

These tests exercise the production ``ConversationService`` on in-memory SQLite
only. They cover the audit families that motivated the port: explicit pronoun
questions, elliptical fact continuations after a resolved person answer,
interpreter term grounding, topic change/return, family roles, ambiguity,
sensitive turns, and conversation isolation.

Continuity is always derived from stored response-source metadata. Transcript
text is never treated as a source, a fact, or authority for a person pick.
"""

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import Intent
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.deterministic import DeterministicResponder
from app.repositories.profile import ConversationProfileRepository
from app.retrieval.entity_resolver import Resolution
from app.retrieval.interpreter_verifier import (
    InterpreterVerifier,
    fold,
    term_is_grounded,
)
from app.schemas.interpreter import InterpreterSuggestion
from app.schemas.profile import ConversationProfileCreate, SensitiveTopicCreate
from app.services.conversation import ConversationService
from tests.fixtures.universe import SeedUniverse

_GENERAL_REPLY = "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."
_CONTINUE_REPLY = "Te escucho."


class ScriptedInterpreter:
    """Deterministic, in-process interpreter double; no network or history use."""

    def __init__(self, suggestions=None) -> None:
        self._suggestions = dict(suggestions or {})
        self.calls = 0
        self.max_input_chars = 500
        self.history_turns = 4
        self.max_history_chars = 1200

    async def suggest(self, text, *, history=()):
        self.calls += 1
        return self._suggestions.get(text)


@dataclass
class _Person:
    id: uuid.UUID
    display_name: str


class _RecordingResolver:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def resolve(self, term: str) -> Resolution:
        self.calls.append(term)
        return Resolution.from_people(term, [_Person(uuid.uuid4(), "Carmen")])


def _service(
    session: AsyncSession,
    interpreter=None,
) -> ConversationService:
    return ConversationService(
        session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
        interpreter=interpreter,
    )


async def _activate_sensitive_profile(
    session: AsyncSession,
    person_id: uuid.UUID,
) -> None:
    await ConversationProfileRepository(session).create(
        ConversationProfileCreate(
            person_id=person_id,
            preferred_name="Carmen",
            conversational_style="Cercano, pausado y con frases cortas.",
            positive_topics=["costura"],
            sensitive_topics=[
                SensitiveTopicCreate(topic="fallecimiento", triggers=["murió"])
            ],
            is_active=True,
        )
    )


def _subject_ids(result) -> set[uuid.UUID]:
    return {
        source.entity_id
        for source in result.sources
        if source.entity_type == "person" and source.usage_type == "subject"
    }


# --------------------------------------------------------------------------
# Explicit pronoun questions
# --------------------------------------------------------------------------


async def test_explicit_pronoun_question_reuses_person_query_subject(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    first = await service.handle_user_message("Háblame de Antonio")
    follow = await service.handle_user_message(
        "¿Quién es él?", conversation_id=first.conversation_id
    )

    assert first.intent == Intent.person_query
    assert follow.intent == Intent.person_query
    assert follow.answer == first.answer
    assert follow.gap is None
    assert "No tengo guardado quién era él" not in follow.answer
    assert _subject_ids(follow) == {seed.person_ids["antonio"]}


async def test_explicit_pronoun_question_reuses_fact_query_subject(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    first = await service.handle_user_message("¿Qué aficiones tiene Antonio?")
    follow = await service.handle_user_message(
        "¿Quién era él?", conversation_id=first.conversation_id
    )

    assert first.intent == Intent.fact_query
    assert follow.intent == Intent.person_query
    assert "pareja de Carmen" in follow.answer
    assert follow.gap is None
    assert _subject_ids(follow) == {seed.person_ids["antonio"]}


async def test_explicit_pronoun_question_without_active_person_stays_general(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    result = await service.handle_user_message("¿Quién es él?")

    assert result.intent == Intent.general_chat
    assert result.sources == []
    assert result.ambiguity is None
    assert result.gap is None
    assert result.answer == _GENERAL_REPLY


async def test_explicit_pronoun_question_after_ambiguity_never_picks(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    ambiguous = await service.handle_user_message("¿Quién era Paco?")
    assert ambiguous.ambiguity is not None

    follow = await service.handle_user_message(
        "¿Quién es él?", conversation_id=ambiguous.conversation_id
    )

    assert follow.intent == Intent.general_chat
    assert follow.ambiguity is None
    assert follow.sources == []
    assert follow.gap is None
    assert "Paco García" not in follow.answer
    assert "Francisco Navarro" not in follow.answer


async def test_explicit_pronoun_question_never_calls_the_interpreter(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    interpreter = ScriptedInterpreter(
        {
            "¿Quién es él?": InterpreterSuggestion(
                intent="person_query", term="Antonio"
            )
        }
    )
    service = _service(db_session, interpreter=interpreter)
    first = await service.handle_user_message("Háblame de Antonio")
    follow = await service.handle_user_message(
        "¿Quién es él?", conversation_id=first.conversation_id
    )

    assert interpreter.calls == 0
    assert follow.intent == Intent.person_query
    assert "pareja de Carmen" in follow.answer


# --------------------------------------------------------------------------
# Elliptical fact continuations
# --------------------------------------------------------------------------


async def test_fact_continuation_after_person_query_carries_subject(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    first = await service.handle_user_message("Háblame de Antonio")
    follow = await service.handle_user_message(
        "¿Y de comer?", conversation_id=first.conversation_id
    )

    assert first.intent == Intent.person_query
    assert follow.intent == Intent.fact_query
    assert follow.answer == "No tengo guardado ningún dato sobre eso de Antonio."
    assert follow.gap is None
    assert _subject_ids(follow) == {seed.person_ids["antonio"]}


async def test_fact_continuation_after_fact_query_still_carries_subject(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    first = await service.handle_user_message("¿Qué aficiones tiene Antonio?")
    follow = await service.handle_user_message(
        "¿Y de comer?", conversation_id=first.conversation_id
    )

    assert "Fishing" in first.answer
    assert follow.intent == Intent.fact_query
    assert "Antonio" in follow.answer
    assert "Fishing" not in follow.answer
    assert _subject_ids(follow) == {seed.person_ids["antonio"]}


async def test_unrelated_turn_breaks_elliptical_fact_continuation(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    first = await service.handle_user_message("Háblame de Antonio")
    unrelated = await service.handle_user_message(
        "Hoy hace buen día para pasear", conversation_id=first.conversation_id
    )
    follow = await service.handle_user_message(
        "¿Y de comer?", conversation_id=first.conversation_id
    )

    assert unrelated.intent == Intent.general_chat
    assert follow.intent == Intent.general_chat
    assert follow.sources == []
    assert follow.answer == _CONTINUE_REPLY
    assert "Antonio" not in follow.answer


async def test_topic_change_and_explicit_return_reestablish_continuity(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    first = await service.handle_user_message("¿Qué aficiones tiene Antonio?")
    await service.handle_user_message(
        "Hoy hace buen día para pasear", conversation_id=first.conversation_id
    )
    returned = await service.handle_user_message(
        "¿Qué aficiones tiene Antonio?", conversation_id=first.conversation_id
    )
    follow = await service.handle_user_message(
        "¿Y de comer?", conversation_id=first.conversation_id
    )

    assert "Fishing" in returned.answer
    assert follow.intent == Intent.fact_query
    assert follow.answer == "No tengo guardado ningún dato sobre eso de Antonio."
    assert _subject_ids(follow) == {seed.person_ids["antonio"]}


async def test_family_role_then_pronoun_keeps_the_active_subject(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    first = await service.handle_user_message("Háblame de Carmen")
    role = await service.handle_user_message(
        "¿Quién es mi marido?", conversation_id=first.conversation_id
    )
    follow = await service.handle_user_message(
        "¿Y ella?", conversation_id=first.conversation_id
    )

    assert role.intent == Intent.person_query
    assert "Antonio" in role.answer
    assert "marido" in role.answer
    assert follow.intent == Intent.person_query
    assert follow.answer == first.answer
    assert _subject_ids(follow) == {seed.person_ids["carmen"]}


async def test_family_role_without_active_person_never_guesses(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    first = await service.handle_user_message("¿Quién es mi marido?")
    assert first.intent == Intent.person_query
    assert "Antes de responder" in first.answer
    assert first.sources == []
    assert first.gap is None

    established = await service.handle_user_message(
        "Háblame de Carmen", conversation_id=first.conversation_id
    )
    assert "pareja de Antonio" in established.answer


# --------------------------------------------------------------------------
# Interpreter grounding
# --------------------------------------------------------------------------


async def test_interpreter_term_absent_from_message_is_rejected(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    interpreter = ScriptedInterpreter(
        {
            "dime qué datos tienes de una persona": InterpreterSuggestion(
                intent="person_query", term="Carmen"
            )
        }
    )
    service = _service(db_session, interpreter=interpreter)
    result = await service.handle_user_message("dime qué datos tienes de una persona")

    assert interpreter.calls == 1
    assert result.intent == Intent.general_chat
    assert result.sources == []
    assert result.gap is None
    assert result.gap_created is False
    assert result.answer == _GENERAL_REPLY


async def test_interpreter_history_only_term_is_rejected(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    interpreter = ScriptedInterpreter(
        {
            "dime qué datos tienes de una persona": InterpreterSuggestion(
                intent="person_query", term="Carmen"
            )
        }
    )
    service = _service(db_session, interpreter=interpreter)
    first = await service.handle_user_message("Háblame de Carmen")
    result = await service.handle_user_message(
        "dime qué datos tienes de una persona",
        conversation_id=first.conversation_id,
    )

    assert interpreter.calls == 1
    assert result.intent == Intent.general_chat
    assert result.sources == []
    assert result.gap is None
    assert result.answer == _CONTINUE_REPLY


async def test_interpreter_substring_term_is_rejected(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    interpreter = ScriptedInterpreter(
        {
            "dime qué datos tienes de Antoniola": InterpreterSuggestion(
                intent="person_query", term="Antonio"
            )
        }
    )
    service = _service(db_session, interpreter=interpreter)
    result = await service.handle_user_message("dime qué datos tienes de Antoniola")

    assert result.intent == Intent.general_chat
    assert result.sources == []
    assert result.gap is None


async def test_interpreter_term_grounding_ignores_case_and_accents(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    interpreter = ScriptedInterpreter(
        {
            "dime qué datos tienes de maria": InterpreterSuggestion(
                intent="person_query", term="María"
            )
        }
    )
    service = _service(db_session, interpreter=interpreter)
    result = await service.handle_user_message("dime qué datos tienes de maria")

    assert result.intent == Intent.person_query
    assert "María" in result.answer
    assert _subject_ids(result) == {seed.person_ids["maria"]}


async def test_interpreter_grounded_name_is_still_verified_and_used(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    interpreter = ScriptedInterpreter(
        {
            "dime qué datos tienes de Antonio": InterpreterSuggestion(
                intent="person_query", term="Antonio"
            )
        }
    )
    service = _service(db_session, interpreter=interpreter)
    result = await service.handle_user_message("dime qué datos tienes de Antonio")

    assert result.intent == Intent.person_query
    assert "pareja de Carmen" in result.answer
    assert _subject_ids(result) == {seed.person_ids["antonio"]}


# --------------------------------------------------------------------------
# Isolation and sensitive turns
# --------------------------------------------------------------------------


async def test_continuity_never_leaks_across_conversations(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    first = await service.handle_user_message("Háblame de Antonio")
    assert _subject_ids(first) == {seed.person_ids["antonio"]}

    other = await service.handle_user_message("¿Quién es él?")
    assert other.conversation_id != first.conversation_id
    assert other.intent == Intent.general_chat
    assert other.sources == []
    assert other.answer == _GENERAL_REPLY

    follow = await service.handle_user_message(
        "¿Y de comer?", conversation_id=other.conversation_id
    )
    assert follow.intent == Intent.general_chat
    assert follow.sources == []
    assert "Antonio" not in follow.answer


async def test_sensitive_blocked_turn_breaks_pronoun_carry(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    await _activate_sensitive_profile(db_session, seed.person_ids["carmen"])
    service = _service(db_session)
    first = await service.handle_user_message("Háblame de Antonio")
    sensitive = await service.handle_user_message(
        "¿Cómo murió Antonio?", conversation_id=first.conversation_id
    )
    follow = await service.handle_user_message(
        "¿Quién es él?", conversation_id=first.conversation_id
    )

    assert sensitive.intent == Intent.general_chat
    assert "No quiero inventar detalles" in sensitive.answer
    assert follow.intent == Intent.general_chat
    assert follow.sources == []
    assert "Antonio" not in follow.answer


# --------------------------------------------------------------------------
# Direct contract unit tests
# --------------------------------------------------------------------------


def test_term_is_grounded_requires_a_whole_token_run() -> None:
    assert term_is_grounded("dime qué datos tienes de Antonio", "Antonio")
    assert term_is_grounded("antonio", "Antonio")
    assert term_is_grounded("Háblame de José", "Jose")
    assert term_is_grounded("HÁBLAME DE JOSÉ", "josé")
    assert term_is_grounded("¿Qué sabes de Paco García?", "paco garcia")
    assert term_is_grounded("Antonio's friend", "Antonio")

    assert not term_is_grounded("dime qué datos tienes de una persona", "Carmen")
    assert not term_is_grounded("Antoniola", "Antonio")
    assert not term_is_grounded("", "Carmen")
    assert not term_is_grounded("Carmen", None)


def test_fold_removes_case_and_accents_only() -> None:
    assert fold("José") == "jose"
    assert fold("MARÍA") == "maria"
    assert fold("Paco García") == "paco garcia"
    assert fold("¿Quién?") == "¿quien?"


async def test_verifier_rejects_ungrounded_before_any_resolution() -> None:
    resolver = _RecordingResolver()
    verifier = InterpreterVerifier(resolver)
    suggestion = InterpreterSuggestion(intent="person_query", term="Carmen")

    assert (
        await verifier.verify(
            suggestion, message="dime qué datos tienes de una persona"
        )
        is None
    )
    assert resolver.calls == []

    verified = await verifier.verify(
        suggestion, message="Ayer estuve pensando en Carmen"
    )
    assert verified is not None
    assert resolver.calls == ["Carmen"]


async def test_ungrounded_rejection_logs_no_text_or_ids(caplog) -> None:
    resolver = _RecordingResolver()
    verifier = InterpreterVerifier(resolver)
    suggestion = InterpreterSuggestion(intent="person_query", term="Carmen")
    message = "Ayer estuve pensando en una persona muy concreta"
    caplog.set_level(logging.WARNING)

    assert await verifier.verify(suggestion, message=message) is None
    assert "Carmen" not in caplog.text
    assert message not in caplog.text
    assert "una persona" not in caplog.text
