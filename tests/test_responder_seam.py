import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import Intent
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.base import Responder, Response
from app.llm.deterministic import DeterministicResponder
from app.llm.provider import RESPONDERS, get_responder
from app.retrieval.context_builder import AnswerContext
from app.services.conversation import ConversationService
from tests.fixtures.universe import SeedUniverse


class StubResponder(Responder):
    """Fake responder used to prove the service accepts an injected one."""

    def __init__(self, text: str = "stub answer") -> None:
        self._text = text

    async def respond(self, context: AnswerContext) -> Response:
        return Response(text=self._text)


def test_provider_defaults_to_deterministic() -> None:
    assert "deterministic" in RESPONDERS
    assert isinstance(get_responder(), DeterministicResponder)


def test_provider_registers_openai_compatible() -> None:
    from app.llm.openai_compatible import OpenAICompatibleResponder

    assert "openai_compatible" in RESPONDERS
    assert RESPONDERS["openai_compatible"] is OpenAICompatibleResponder


def test_provider_selects_openai_compatible_from_settings(monkeypatch) -> None:
    from app.llm.openai_compatible import OpenAICompatibleResponder

    class _FakeSettings:
        responder = "openai_compatible"
        llm_base_url = "http://local"
        llm_model = "local-model"
        llm_api_key = "k"
        llm_timeout_seconds = 5.0

    monkeypatch.setattr("app.llm.provider.get_settings", lambda: _FakeSettings())
    responder = get_responder()
    assert isinstance(responder, OpenAICompatibleResponder)
    assert responder._base_url == "http://local"
    assert responder._model == "local-model"
    assert responder._api_key == "k"
    assert responder._timeout == 5.0


def test_provider_rejects_unknown_responder(monkeypatch) -> None:
    class _FakeSettings:
        responder = "nonexistent"

    monkeypatch.setattr(
        "app.llm.provider.get_settings", lambda: _FakeSettings()
    )
    with pytest.raises(ValueError, match="Unknown responder 'nonexistent'"):
        get_responder()


async def test_service_uses_injected_responder(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = ConversationService(
        db_session,
        StubResponder("respuesta de prueba"),
        embedder=DeterministicEmbedder(),
    )
    result = await service.handle_user_message("¿Quién era Francisco?")
    assert result.answer == "respuesta de prueba"
    assert result.intent == Intent.person_query


async def test_service_hardcoded_import_removed() -> None:
    import inspect

    import app.services.conversation as conversation_module

    source = inspect.getsource(conversation_module)
    assert "DeterministicResponder" not in source
    assert "DeterministicEmbedder" not in source


async def test_service_routes_orientation_through_code_clock(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = ConversationService(
        db_session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
    )
    result = await service.handle_user_message("¿Qué hora es?")
    assert result.intent == Intent.orientation_query
    assert result.answer.startswith("Hoy es ")
    assert result.sources == []
