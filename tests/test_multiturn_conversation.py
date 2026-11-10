"""Multi-turn conversational continuity, policy and isolation.

These tests simulate long safe conversations with the deterministic responder
and mocked providers only. They prove that verified person/topic context carries
across adjacent turns, that a pronoun or omitted subject never guesses after an
ambiguous or unrelated turn, that history is bounded and never becomes a source,
that sensitive topics stay suppressed, and that voice turns share the exact same
``ConversationService`` behavior as text.
"""

import json
import uuid

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.conversation_context import (
    ConversationTurn,
    bound_conversation_history,
)
from app.domain.enums import Intent
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.base import Responder, Response
from app.llm.deterministic import DeterministicResponder
from app.llm.interpreter import OpenAICompatibleInterpreter
from app.models import Message, ResponseSource
from app.repositories.profile import ConversationProfileRepository
from app.retrieval.context_builder import AnswerContext
from app.schemas.profile import ConversationProfileCreate, SensitiveTopicCreate
from app.services.conversation import ConversationService
from app.voice.service import VoiceService
from tests.fixtures.universe import SeedUniverse

_GENERAL_REPLY = "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."


def _service(
    session: AsyncSession,
    responder: Responder | None = None,
    interpreter=None,
) -> ConversationService:
    return ConversationService(
        session,
        responder or DeterministicResponder(),
        embedder=DeterministicEmbedder(),
        interpreter=interpreter,
    )


def _connection_sources(result) -> set[tuple[str, str]]:
    return {(source.entity_type, str(source.entity_id)) for source in result.sources}


async def _activate_sensitive_profile(
    session: AsyncSession,
    person_id: uuid.UUID,
    *,
    triggers: list[str] | None = None,
) -> None:
    await ConversationProfileRepository(session).create(
        ConversationProfileCreate(
            person_id=person_id,
            preferred_name="Carmen",
            conversational_style="Cercano, pausado y con frases cortas.",
            positive_topics=["costura"],
            sensitive_topics=[
                SensitiveTopicCreate(
                    topic="fallecimiento",
                    triggers=triggers or ["murió"],
                )
            ],
            is_active=True,
        )
    )


class CapturingResponder(Responder):
    """Capture responder inputs while delegating to the deterministic copy."""

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


class ScriptedSTT:
    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)
        self.calls = 0

    async def transcribe(self, audio: bytes, *, content_type=None) -> str:
        transcript = self._transcripts[self.calls]
        self.calls += 1
        return transcript


def _interpreter_client(
    content: str = "{}",
    *,
    error: Exception | None = None,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if error is not None:
            raise error
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------
# Long safe conversation
# --------------------------------------------------------------------------


async def test_fifteen_turn_conversation_keeps_verified_context_and_sources(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    results = []
    conversation_id = None

    turns = [
        ("Hola", Intent.general_chat),
        ("Háblame de Carmen", Intent.person_query),
        ("¿Y ella?", Intent.person_query),
        ("Háblame de Antonio", Intent.person_query),
        ("¿Y él?", Intent.person_query),
        ("¿Qué aficiones tiene Antonio?", Intent.fact_query),
        ("Háblame sobre su infancia", Intent.fact_query),
        ("¿Y dónde vivió?", Intent.fact_query),
        ("Me encanta la tortilla de patatas", Intent.general_chat),
        ("Háblame sobre su infancia", Intent.general_chat),
        ("¿Quién era Paco?", Intent.person_query),
        ("Francisco", Intent.person_query),
        ("Háblame sobre su familia", Intent.fact_query),
        ("¿Quién era Eustaquio?", Intent.person_query),
        ("¿Y ella?", Intent.general_chat),
    ]
    for text, expected_intent in turns:
        result = await service.handle_user_message(
            text, conversation_id=conversation_id
        )
        if conversation_id is None:
            conversation_id = result.conversation_id
        else:
            assert result.conversation_id == conversation_id
        assert result.intent == expected_intent, text
        results.append(result)

    # The first greeting is not repeated every turn.
    assert results[0].answer == _GENERAL_REPLY
    assert results[8].answer == "Gracias por contármelo."
    assert results[9].answer == "Te escucho."
    assert results[14].answer == "Te escucho."

    # Verified person and topic continuity across safe adjacent turns.
    assert "pareja de Antonio" in results[1].answer
    assert results[2].answer == results[1].answer
    assert "pareja de Carmen" in results[3].answer
    assert results[4].answer == results[3].answer
    assert "Fishing" in results[5].answer
    assert results[6].answer == "No sé nada de la infancia de Antonio todavía."
    assert "Guardamar de la Safor" in results[7].answer
    # An unrelated conversational turn resets the carried subject.
    assert "Antonio" not in results[9].answer
    assert "infancia" not in results[9].answer

    # Ambiguity is surfaced and only the explicit disambiguation selects one.
    assert results[10].ambiguity is not None
    assert len(results[10].ambiguity.candidates) == 2
    assert results[11].ambiguity is None
    assert "amigo de Antonio" in results[11].answer
    assert results[12].answer == (
        "No sé nada de la familia de Francisco Navarro todavía."
    )

    # An unknown person creates a gap and does not leak into the next pronoun.
    assert results[13].gap is not None
    assert results[14].gap is None
    assert results[14].sources == []

    # Every general-chat turn returns no sources; conversation history never
    # becomes a fact or a response source.
    for index in (0, 8, 9, 14):
        assert results[index].sources == []
    allowed_types = {
        "person",
        "relationship",
        "fact",
        "source",
        "memory",
        "event",
        "media",
    }
    for result in results:
        assert all(source.entity_type in allowed_types for source in result.sources)

    # One atomic user/assistant pair per turn, with stored response sources.
    conversation_uuid = conversation_id
    messages = (
        (
            await db_session.execute(
                select(Message).where(Message.conversation_id == conversation_uuid)
            )
        )
        .scalars()
        .all()
    )
    assert len(messages) == 2 * len(turns)
    assistants = [message for message in messages if message.speaker == "assistant"]
    assert len(assistants) == len(turns)
    assert all(message.audio_path is None for message in messages)
    stored_sources = (
        await db_session.execute(
            select(func.count())
            .select_from(ResponseSource)
            .join(Message, Message.id == ResponseSource.message_id)
            .where(Message.conversation_id == conversation_uuid)
        )
    ).scalar_one()
    assert stored_sources > 0


async def test_reformulation_pronoun_assertion_and_topic_return(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)
    conversation_id = None
    results = []

    for text in (
        "Háblame de Carmen",
        "¿Quién era Carmen?",
        "Háblame de ese",
        "Vivíamos en Madrid en 1972.",
        "¿Qué aficiones tiene Antonio?",
        "Hoy hace buen día para pasear",
        "¿Qué aficiones tiene Antonio?",
        "¿Y de comer?",
    ):
        result = await service.handle_user_message(
            text, conversation_id=conversation_id
        )
        conversation_id = result.conversation_id
        results.append(result)

    # A reformulation and a pronoun resolve the same verified person.
    assert results[0].intent == Intent.person_query
    assert results[1].intent == Intent.person_query
    assert results[1].answer == results[0].answer
    assert results[2].intent == Intent.person_query
    assert results[2].answer == results[0].answer

    # A first-person assertion is acknowledged conversationally, with no fact
    # promotion and no source.
    assert results[3].intent == Intent.general_chat
    assert results[3].answer == "Te escucho."
    assert results[3].sources == []

    # Changing topic and returning to a fact works; an elliptical follow-up
    # carries the returned subject.
    assert "Fishing" in results[4].answer
    assert results[5].intent == Intent.general_chat
    assert results[6].intent == Intent.fact_query
    assert "Fishing" in results[6].answer
    assert results[7].intent == Intent.fact_query
    assert "Fishing" not in results[7].answer
    assert "guardado" in results[7].answer


async def test_history_never_leaks_across_independent_conversations(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)

    first = await service.handle_user_message("Háblame de Carmen")
    first_follow = await service.handle_user_message(
        "¿Y ella?", conversation_id=first.conversation_id
    )
    assert first_follow.intent == Intent.person_query
    assert "Carmen" in first_follow.answer

    other = await service.handle_user_message("¿Y ella?")
    assert other.conversation_id != first.conversation_id
    assert other.intent == Intent.general_chat
    assert "Carmen" not in other.answer
    assert other.sources == []

    unrelated = await service.handle_user_message("Háblame sobre su infancia")
    assert unrelated.intent == Intent.general_chat
    assert "Carmen" not in unrelated.answer
    assert unrelated.sources == []


async def test_pronoun_after_ambiguity_never_picks_a_candidate(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session)

    ambiguous = await service.handle_user_message("¿Quién era Paco?")
    assert ambiguous.ambiguity is not None

    follow = await service.handle_user_message(
        "¿Y él?", conversation_id=ambiguous.conversation_id
    )
    assert follow.intent == Intent.general_chat
    assert follow.ambiguity is None
    assert follow.sources == []
    assert "Francisco" not in follow.answer
    assert "Paco García" not in follow.answer


# --------------------------------------------------------------------------
# Responder history and style
# --------------------------------------------------------------------------


async def test_responder_receives_current_message_bounded_filtered_history_and_style(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    await _activate_sensitive_profile(db_session, seed.person_ids["carmen"])
    responder = CapturingResponder()
    service = _service(db_session, responder)

    first = await service.handle_user_message("Háblame de Carmen")
    sensitive = await service.handle_user_message(
        "Carmen murió ayer", conversation_id=first.conversation_id
    )
    assert "No quiero inventar detalles" in sensitive.answer
    third = await service.handle_user_message(
        "Hola", conversation_id=first.conversation_id
    )
    assert third.intent == Intent.general_chat

    call = responder.calls[-1]
    assert call["message"] == "Hola"
    assert call["history"]
    assert len(call["history"]) <= 8
    assert sum(len(turn.text) for turn in call["history"]) <= 2000
    assert all(turn.role in ("user", "assistant") for turn in call["history"])
    assert all("murió" not in turn.text.lower() for turn in call["history"])
    assert all(turn.text != "Hola" for turn in call["history"])
    assert call["style"].preferred_name == "Carmen"
    assert call["style"].conversational_style.startswith("Cercano")
    assert call["style"].positive_topics == ("costura",)


def test_history_bound_keeps_the_newest_turns_and_characters() -> None:
    history = [
        ConversationTurn(role="user", text=f"turno-{index}-" + "x" * 30)
        for index in range(20)
    ]

    bounded = bound_conversation_history(history, max_turns=3, max_chars=90)

    assert len(bounded) == 3
    assert [turn.text.split("-")[1] for turn in bounded] == ["17", "18", "19"]
    assert sum(len(turn.text) for turn in bounded) <= 90
    assert bound_conversation_history(history, max_turns=0) == ()
    assert bound_conversation_history(history, max_chars=0) == ()
    assert (
        bound_conversation_history(
            [ConversationTurn(role="system", text="ignored")], max_turns=2
        )
        == ()
    )


# --------------------------------------------------------------------------
# Sensitive policy across turns
# --------------------------------------------------------------------------


async def test_sensitive_topic_stays_controlled_across_turns(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    await _activate_sensitive_profile(db_session, seed.person_ids["carmen"])
    service = _service(db_session)

    first = await service.handle_user_message("Háblame de Antonio")
    assert first.intent == Intent.person_query
    assert "pareja de Carmen" in first.answer

    data = await service.handle_user_message(
        "¿Qué datos tienes de Antonio?", conversation_id=first.conversation_id
    )
    assert data.intent == Intent.fact_query
    assert "Guardamar de la Safor" in data.answer
    assert "Fishing" in data.answer
    assert "fallecimiento" not in data.answer.lower()
    assert "paso" not in data.answer.lower()
    assert all(source.entity_type != "memory" for source in data.sources)

    explicit = await service.handle_user_message(
        "Antonio murió, ¿verdad?", conversation_id=first.conversation_id
    )
    assert "No quiero inventar detalles" in explicit.answer
    assert "costura" in explicit.answer
    assert explicit.sources == []

    after = await service.handle_user_message(
        "Háblame sobre su familia", conversation_id=first.conversation_id
    )
    assert after.intent == Intent.general_chat
    assert after.answer == "Te escucho."
    assert "Antonio" not in after.answer
    assert "fallecimiento" not in after.answer.lower()


# --------------------------------------------------------------------------
# Interpreter failure keeps deterministic continuity
# --------------------------------------------------------------------------


async def test_interpreter_timeout_falls_back_and_later_turns_still_continue(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    async with _interpreter_client(error=httpx.TimeoutException("timeout")) as client:
        interpreter = OpenAICompatibleInterpreter(
            base_url="http://llm.test",
            model="test-model",
            client=client,
        )
        service = _service(db_session, interpreter=interpreter)

        first = await service.handle_user_message("Háblame de Carmen")
        assert first.intent == Intent.person_query

        fallback = await service.handle_user_message(
            "Ayer estuve pensando en Carmen", conversation_id=first.conversation_id
        )
        assert fallback.intent == Intent.general_chat
        assert fallback.answer == "Te escucho."

        later = await service.handle_user_message(
            "Háblame de Antonio", conversation_id=first.conversation_id
        )
        assert later.intent == Intent.person_query
        follow = await service.handle_user_message(
            "¿Y él?", conversation_id=first.conversation_id
        )
        assert follow.intent == Intent.person_query
        assert "pareja de Carmen" in follow.answer


async def test_interpreter_invalid_response_falls_back_without_inventing(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    payload = json.dumps(
        {
            "intent": "person_query",
            "term": "Carmen",
            "answer": "Carmen nació en 1900.",
        }
    )
    async with _interpreter_client(payload) as client:
        interpreter = OpenAICompatibleInterpreter(
            base_url="http://llm.test",
            model="test-model",
            client=client,
        )
        service = _service(db_session, interpreter=interpreter)

        result = await service.handle_user_message("Ayer estuve pensando en Carmen")

    assert result.intent == Intent.general_chat
    assert result.answer == _GENERAL_REPLY
    assert result.sources == []


# --------------------------------------------------------------------------
# Voice shares the conversation service
# --------------------------------------------------------------------------


async def test_voice_and_text_share_continuity_and_natural_replies(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    transcripts = [
        "Háblame de Carmen",
        "¿Y ella?",
        "Me encanta la tortilla de patatas",
        "Háblame sobre su infancia",
    ]

    text_service = _service(db_session)
    text_conversation = None
    text_answers = []
    for text in transcripts:
        result = await text_service.handle_user_message(
            text, conversation_id=text_conversation
        )
        text_conversation = result.conversation_id
        text_answers.append((result.intent, result.answer, result.sources))

    voice_service = VoiceService(
        db_session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
        stt=ScriptedSTT(transcripts),
    )
    voice_conversation = None
    for index, transcript in enumerate(transcripts):
        result = await voice_service.process(
            audio=b"\x00wav",
            audio_content_type="audio/wav",
            conversation_id=voice_conversation,
        )
        voice_conversation = result.conversation_id
        expected_intent, expected_answer, expected_sources = text_answers[index]
        assert result.transcript == transcript
        assert result.intent == expected_intent
        assert result.answer == expected_answer
        assert _connection_sources(result) == {
            (source.entity_type, str(source.entity_id)) for source in expected_sources
        }
