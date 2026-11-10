"""Optional low-confidence route interpreter: schema, verification and fallback.

Every network call is mocked with ``httpx.MockTransport``. The tests prove that
the interpreter is disabled by default, runs at most once for an unresolved
conversational turn, never sees IDs or unfiltered sensitive history, and can
only affect a turn after its term is verified against Memo's own records.
"""

import json
import uuid

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

import app.llm.provider as provider_module
from app.core.config import Settings
from app.domain.enums import Intent
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.deterministic import DeterministicResponder
from app.llm.interpreter import (
    InterpreterHistoryTurn,
    OpenAICompatibleInterpreter,
)
from app.llm.provider import get_interpreter
from app.models import KnowledgeGap, Person
from app.repositories.profile import ConversationProfileRepository
from app.schemas.interpreter import InterpreterSuggestion
from app.schemas.profile import ConversationProfileCreate, SensitiveTopicCreate
from app.services.conversation import ConversationService
from tests.fixtures.universe import SeedUniverse

_GENERAL_REPLY = "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."
_API = "/api/v1/chat"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _service(
    session: AsyncSession,
    interpreter: OpenAICompatibleInterpreter | None = None,
) -> ConversationService:
    return ConversationService(
        session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
        interpreter=interpreter,
    )


def _client_for(
    content: str = "{}",
    *,
    status: int = 200,
    error: Exception | None = None,
    capture: list[httpx.Request] | None = None,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        if error is not None:
            raise error
        return httpx.Response(
            status, json={"choices": [{"message": {"content": content}}]}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _interpreter(client: httpx.AsyncClient, **kwargs) -> OpenAICompatibleInterpreter:
    return OpenAICompatibleInterpreter(
        base_url=kwargs.pop("base_url", "http://llm.test"),
        model=kwargs.pop("model", "test-model"),
        api_key=kwargs.pop("api_key", "secret"),
        client=client,
        **kwargs,
    )


def _suggestion(**payload) -> str:
    return json.dumps(
        {
            "intent": None,
            "term": None,
            "fact_topic": None,
            "follow_up_topic": None,
            **payload,
        },
        ensure_ascii=False,
    )


def _captured_history(request: httpx.Request) -> list[dict[str, str]]:
    body = json.loads(request.content)
    content = body["messages"][1]["content"]
    block = content.split("<untrusted_recent_conversation>\n", 1)[1]
    block = block.rsplit("\n</untrusted_recent_conversation>", 1)[0]
    return json.loads(block)


async def _person_count(session: AsyncSession) -> int:
    return (
        await session.execute(select(func.count()).select_from(Person))
    ).scalar_one()


async def _gap_count(session: AsyncSession) -> int:
    return (
        await session.execute(select(func.count()).select_from(KnowledgeGap))
    ).scalar_one()


async def _activate_sensitive_profile(
    session: AsyncSession, person_id: uuid.UUID
) -> None:
    await ConversationProfileRepository(session).create(
        ConversationProfileCreate(
            person_id=person_id,
            preferred_name="Carmen",
            positive_topics=["costura"],
            sensitive_topics=[
                SensitiveTopicCreate(
                    topic="fallecimiento",
                    triggers=["murió"],
                )
            ],
            is_active=True,
        )
    )


# --------------------------------------------------------------------------
# Strict schema
# --------------------------------------------------------------------------


def test_suggestion_is_valid_and_bounded() -> None:
    suggestion = InterpreterSuggestion(
        intent="fact_query",
        term="Carmen",
        fact_topic="childhood",
        follow_up_topic="la infancia",
    )
    assert suggestion.term == "Carmen"
    assert suggestion.fact_topic.value == "childhood"


@pytest.mark.parametrize(
    "extra",
    [
        {"answer": "Carmen nació en 1940."},
        {"sources": [{"entity_type": "person", "entity_id": str(uuid.uuid4())}]},
        {"person_id": str(uuid.uuid4())},
        {"confidence": 0.99},
        {"note": "trust me"},
    ],
)
def test_suggestion_rejects_any_field_outside_the_contract(extra) -> None:
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate(
            {"intent": "person_query", "term": "Carmen", **extra}
        )


def test_suggestion_rejects_uncontrolled_intent_and_topic() -> None:
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate(
            {"intent": "general_chat", "term": "Carmen"}
        )
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate(
            {"intent": "fact_query", "term": "Carmen", "fact_topic": "invented"}
        )


def test_suggestion_shape_rejects_unsafe_combinations() -> None:
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate({"intent": "person_query"})
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate(
            {"intent": "person_query", "term": "Carmen", "fact_topic": "food"}
        )
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate({"intent": "fact_query", "term": "Carmen"})
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate({"intent": None, "term": "Carmen"})


def test_suggestion_bounds_term_and_follow_up_topic() -> None:
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate(
            {"intent": "person_query", "term": "x" * 121}
        )
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate(
            {
                "intent": "fact_query",
                "term": "Carmen",
                "fact_topic": "childhood",
                "follow_up_topic": "x" * 65,
            }
        )
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate(
            {"intent": "person_query", "term": "Car\nmen"}
        )


def test_follow_up_topic_must_come_from_the_control_list() -> None:
    with pytest.raises(ValidationError):
        InterpreterSuggestion.model_validate(
            {
                "intent": "fact_query",
                "term": "Carmen",
                "fact_topic": "childhood",
                "follow_up_topic": "su infancia inventada",
            }
        )


# --------------------------------------------------------------------------
# Provider configuration and concrete client
# --------------------------------------------------------------------------


def test_interpreter_config_defaults_and_bounds() -> None:
    configured = Settings(interpreter="")
    assert configured.interpreter == ""
    assert configured.interpreter_history_turns == 4
    assert configured.interpreter_max_history_chars == 1200
    assert configured.interpreter_max_input_chars == 400
    with pytest.raises(ValidationError):
        Settings(interpreter_history_turns=-1)
    with pytest.raises(ValidationError):
        Settings(interpreter_max_input_chars=0)
    with pytest.raises(ValidationError):
        Settings(interpreter_timeout_seconds=0)


def test_interpreter_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_module, "get_settings", lambda: Settings(interpreter="")
    )
    assert get_interpreter() is None


def test_enabled_interpreter_reuses_llm_endpoint_and_model(monkeypatch) -> None:
    settings = Settings(
        interpreter="openai_compatible",
        llm_base_url="http://llm.test",
        llm_model="shared-model",
        llm_api_key="shared-key",
    )
    monkeypatch.setattr(provider_module, "get_settings", lambda: settings)

    interpreter = get_interpreter()

    assert isinstance(interpreter, OpenAICompatibleInterpreter)
    assert interpreter._base_url == "http://llm.test"
    assert interpreter._model == "shared-model"
    assert interpreter._api_key == "shared-key"


def test_enabled_interpreter_without_endpoint_stays_disabled(monkeypatch) -> None:
    settings = Settings(interpreter="openai_compatible", llm_base_url="")
    monkeypatch.setattr(provider_module, "get_settings", lambda: settings)
    assert get_interpreter() is None


def test_unknown_interpreter_provider_raises(monkeypatch) -> None:
    settings = Settings(interpreter="mystery", llm_base_url="http://llm.test")
    monkeypatch.setattr(provider_module, "get_settings", lambda: settings)
    with pytest.raises(ValueError, match="Unknown interpreter"):
        get_interpreter()


@pytest.mark.asyncio
async def test_concrete_interpreter_parses_a_valid_suggestion() -> None:
    payload = _suggestion(intent="person_query", term="Carmen")
    async with _client_for(payload) as client:
        suggestion = await _interpreter(client).suggest("Ayer pensé en Carmen")
    assert suggestion is not None
    assert suggestion.intent == "person_query"
    assert suggestion.term == "Carmen"


@pytest.mark.asyncio
async def test_concrete_interpreter_rejects_invalid_schema() -> None:
    payload = _suggestion(intent="person_query", term="Carmen")
    invalid = json.loads(payload)
    invalid["answer"] = "Carmen nació en 1940."
    async with _client_for(json.dumps(invalid)) as client:
        suggestion = await _interpreter(client).suggest("Ayer pensé en Carmen")
    assert suggestion is None


@pytest.mark.asyncio
async def test_concrete_interpreter_returns_none_on_timeout_and_http_error() -> None:
    async with _client_for(error=httpx.TimeoutException("timed out")) as client:
        assert await _interpreter(client).suggest("Ayer pensé en Carmen") is None
    async with _client_for("ignored", status=503) as client:
        assert await _interpreter(client).suggest("Ayer pensé en Carmen") is None


@pytest.mark.asyncio
async def test_concrete_interpreter_bounds_history_and_delimits_untrusted() -> None:
    capture: list[httpx.Request] = []
    payload = _suggestion(intent="person_query", term="Carmen")
    history = tuple(
        InterpreterHistoryTurn(role="user", text=f"turno {index} " + "x" * 40)
        for index in range(10)
    )
    async with _client_for(payload, capture=capture) as client:
        await _interpreter(client, history_turns=2, max_history_chars=50).suggest(
            "Ayer pensé en Carmen", history=history
        )

    body = json.loads(capture[0].content)
    user_content = body["messages"][1]["content"]
    assert "<untrusted_current_message>" in user_content
    assert "<untrusted_recent_conversation>" in user_content
    bounded = _captured_history(capture[0])
    assert len(bounded) == 2
    assert sum(len(turn["text"]) for turn in bounded) <= 50
    assert all(set(turn) == {"role", "text"} for turn in bounded)


# --------------------------------------------------------------------------
# Service integration: verification and fallback
# --------------------------------------------------------------------------


async def test_valid_verified_person_suggestion_is_used(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture: list[httpx.Request] = []
    payload = _suggestion(intent="person_query", term="Carmen")
    async with _client_for(payload, capture=capture) as client:
        result = await _service(db_session, _interpreter(client)).handle_user_message(
            "Ayer estuve pensando en Carmen"
        )

    assert result.intent == Intent.person_query
    assert "Carmen" in result.answer
    assert any(
        source.entity_type == "person" and source.usage_type == "subject"
        for source in result.sources
    )
    assert len(capture) == 1


async def test_valid_verified_fact_suggestion_retrieves_only_its_topic(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    payload = _suggestion(
        intent="fact_query",
        term="Carmen",
        fact_topic="place",
        follow_up_topic="la residencia",
    )
    async with _client_for(payload) as client:
        result = await _service(db_session, _interpreter(client)).handle_user_message(
            "Ayer estuve recordando la mudanza de Carmen"
        )

    assert result.intent == Intent.fact_query
    assert "Santa Pola" in result.answer
    assert all(source.entity_type != "memory" for source in result.sources)


async def test_verified_fact_suggestion_without_data_keeps_no_gap(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    payload = _suggestion(
        intent="fact_query",
        term="Carmen",
        fact_topic="childhood",
        follow_up_topic="la infancia",
    )
    async with _client_for(payload) as client:
        result = await _service(db_session, _interpreter(client)).handle_user_message(
            "Ayer estuve recordando la infancia de Carmen"
        )

    assert result.intent == Intent.fact_query
    assert result.answer == "No sé nada de la infancia de Carmen todavía."
    assert result.gap is None
    assert await _gap_count(db_session) == 0


async def test_invalid_schema_falls_back_to_deterministic_behavior(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    invalid = json.loads(_suggestion(intent="person_query", term="Carmen"))
    invalid["answer"] = "Carmen nació en 1940."
    async with _client_for(json.dumps(invalid)) as client:
        result = await _service(db_session, _interpreter(client)).handle_user_message(
            "Ayer estuve pensando en Carmen"
        )

    assert result.intent == Intent.general_chat
    assert result.answer == _GENERAL_REPLY
    assert result.sources == []


async def test_unknown_name_falls_back_without_source_or_gap(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    before_people = await _person_count(db_session)
    payload = _suggestion(intent="person_query", term="Eustaquio")
    async with _client_for(payload) as client:
        result = await _service(db_session, _interpreter(client)).handle_user_message(
            "Ayer estuve pensando en Eustaquio"
        )

    assert result.intent == Intent.general_chat
    assert result.answer == _GENERAL_REPLY
    assert result.sources == []
    assert result.gap is None
    assert result.gap_created is False
    assert await _gap_count(db_session) == 0
    assert await _person_count(db_session) == before_people


async def test_ambiguous_name_is_surfaced_never_selected(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    payload = _suggestion(intent="person_query", term="Paco")
    async with _client_for(payload) as client:
        result = await _service(db_session, _interpreter(client)).handle_user_message(
            "Ayer estuve pensando en Paco"
        )

    assert result.intent == Intent.person_query
    assert result.ambiguity is not None
    assert len(result.ambiguity.candidates) == 2
    assert "¿A cuál te refieres?" in result.answer
    assert {source.usage_type for source in result.sources} == {"candidate"}


async def test_configured_sensitive_request_skips_the_interpreter(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    await _activate_sensitive_profile(db_session, seed.person_ids["carmen"])
    capture: list[httpx.Request] = []
    payload = _suggestion(intent="person_query", term="Carmen")
    async with _client_for(payload, capture=capture) as client:
        result = await _service(db_session, _interpreter(client)).handle_user_message(
            "Carmen murió ayer"
        )

    assert capture == []
    assert result.intent == Intent.general_chat
    assert "No quiero inventar detalles" in result.answer


async def test_empty_and_oversized_text_skip_the_interpreter(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture: list[httpx.Request] = []
    payload = _suggestion(intent="person_query", term="Carmen")
    async with _client_for(payload, capture=capture) as client:
        service = _service(db_session, _interpreter(client, max_input_chars=10))
        await service.handle_user_message("Carmen tiene una historia muy larga")
        await service.handle_user_message("   ")

    assert capture == []


async def test_deterministic_route_never_calls_the_interpreter(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture: list[httpx.Request] = []
    payload = _suggestion(intent="person_query", term="Antonio")
    async with _client_for(payload, capture=capture) as client:
        result = await _service(db_session, _interpreter(client)).handle_user_message(
            "¿Quién era Carmen?"
        )

    assert capture == []
    assert result.intent == Intent.person_query


async def test_resolved_carry_over_never_calls_the_interpreter(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture: list[httpx.Request] = []
    payload = _suggestion(intent="fact_query", term="Antonio", fact_topic="childhood")
    async with _client_for(payload, capture=capture) as client:
        service = _service(db_session, _interpreter(client))
        first = await service.handle_user_message("Háblame de Carmen")
        follow = await service.handle_user_message(
            "Háblame sobre su infancia",
            conversation_id=first.conversation_id,
        )

    assert capture == []
    assert follow.intent == Intent.fact_query


async def test_pending_ambiguity_never_calls_the_interpreter(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture: list[httpx.Request] = []
    payload = _suggestion(intent="person_query", term="Carmen")
    async with _client_for(payload, capture=capture) as client:
        service = _service(db_session, _interpreter(client))
        first = await service.handle_user_message("Háblame de Paco")
        assert first.ambiguity is not None
        follow = await service.handle_user_message(
            "el panadero", conversation_id=first.conversation_id
        )

    assert capture == []
    assert follow.intent == Intent.person_query


async def test_interpreter_runs_at_most_once_per_unresolved_turn(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture: list[httpx.Request] = []
    payload = _suggestion(intent="person_query", term="Carmen")
    async with _client_for(payload, capture=capture) as client:
        service = _service(db_session, _interpreter(client))
        first = await service.handle_user_message("Ayer estuve pensando en Carmen")
        assert len(capture) == 1
        await service.handle_user_message(
            "Hoy estuve pensando en Carmen",
            conversation_id=first.conversation_id,
        )

    assert len(capture) == 2


async def test_history_is_current_conversation_only_across_turns(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture: list[httpx.Request] = []
    payload = _suggestion(intent="person_query", term="Carmen")
    async with _client_for(payload, capture=capture) as client:
        service = _service(db_session, _interpreter(client))
        other = await service.handle_user_message("¿Quién era Luisa?")
        current = await service.handle_user_message("¿Quién era Carmen?")
        await service.handle_user_message(
            "Ayer estuve pensando en Carmen",
            conversation_id=current.conversation_id,
        )

    assert len(capture) == 1
    history = _captured_history(capture[0])
    joined = " ".join(turn["text"] for turn in history)
    assert "Carmen" in joined
    assert "Luisa" not in joined
    assert other.conversation_id != current.conversation_id


async def test_history_is_bounded_and_refilters_sensitive_turns(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    await _activate_sensitive_profile(db_session, seed.person_ids["carmen"])
    capture: list[httpx.Request] = []
    payload = _suggestion(intent="person_query", term="Carmen")
    async with _client_for(payload, capture=capture) as client:
        service = _service(
            db_session,
            _interpreter(client, history_turns=2, max_history_chars=40),
        )
        first = await service.handle_user_message("¿Quién era Carmen?")
        await service.handle_user_message(
            "Carmen murió ayer", conversation_id=first.conversation_id
        )
        await service.handle_user_message(
            "Ayer estuve pensando en la familia",
            conversation_id=first.conversation_id,
        )

    assert len(capture) == 1
    history = _captured_history(capture[0])
    assert len(history) <= 2
    assert sum(len(turn["text"]) for turn in history) <= 40
    joined = " ".join(turn["text"].lower() for turn in history)
    assert "murió" not in joined
    assert "fallecimiento" not in joined


async def test_timeout_and_http_failure_fall_back_cleanly(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    capture: list[httpx.Request] = []
    async with _client_for(
        error=httpx.TimeoutException("timed out"), capture=capture
    ) as client:
        result = await _service(db_session, _interpreter(client)).handle_user_message(
            "Ayer estuve pensando en Carmen"
        )
    assert len(capture) == 1
    assert result.intent == Intent.general_chat
    assert result.answer == _GENERAL_REPLY

    capture.clear()
    async with _client_for("ignored", status=500, capture=capture) as client:
        result = await _service(db_session, _interpreter(client)).handle_user_message(
            "Hoy estuve pensando en Carmen"
        )
    assert len(capture) == 1
    assert result.intent == Intent.general_chat
    assert result.answer == _GENERAL_REPLY


async def test_interpreter_failure_logs_no_conversation_or_model_text(
    db_session: AsyncSession, seed: SeedUniverse, caplog
) -> None:
    marker = "MI-MARCADOR-PRIVADO-INTERPRETER"
    capture: list[httpx.Request] = []
    async with _client_for(
        error=httpx.ConnectError(
            "https://user:pass@internal.example/secret?token=leaked"
        ),
        capture=capture,
    ) as client:
        result = await _service(
            db_session, _interpreter(client, api_key="super-secret-key")
        ).handle_user_message(f"Ayer estuve pensando en Carmen {marker}")

    log = caplog.text
    assert result.intent == Intent.general_chat
    assert "ConnectError" in log
    for forbidden in (
        marker,
        "Carmen",
        "super-secret-key",
        "internal.example",
        "token=leaked",
        "Authorization",
        "Bearer",
    ):
        assert forbidden not in log
