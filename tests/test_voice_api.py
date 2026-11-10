import base64
import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.main import app
from app.models import Message
from app.voice.provider import get_speech_to_text, get_text_to_speech
from tests.fixtures.universe import SeedUniverse

API = "/api/v1/chat/voice"


@pytest.fixture(autouse=True)
def _never_use_real_voice_providers():
    """Keep every test in this module offline, even if the shell configures TTS."""
    app.dependency_overrides[get_speech_to_text] = lambda: None
    app.dependency_overrides[get_text_to_speech] = lambda: None
    yield
    app.dependency_overrides.pop(get_speech_to_text, None)
    app.dependency_overrides.pop(get_text_to_speech, None)


class FakeSTT:
    def __init__(self, transcript: str) -> None:
        self._transcript = transcript

    async def transcribe(self, audio: bytes, *, content_type=None) -> str:
        return self._transcript


class FakeTTS:
    def __init__(self, audio: bytes) -> None:
        self._audio = audio

    async def synthesize(self, text: str) -> bytes:
        return self._audio


class ExplodingTTS:
    """Provider that fails unexpectedly after the turn was persisted."""

    async def synthesize(self, text: str) -> bytes:
        raise RuntimeError("unexpected TTS failure")


async def test_text_only_voice_call_degrades_to_chat(
    app_client: AsyncClient,
) -> None:
    response = await app_client.post(API, data={"text": "¿Quién era Eustaquio?"})
    assert response.status_code == 200
    body = response.json()

    assert body["transcript"] is None
    assert body["text"] == "¿Quién era Eustaquio?"
    assert body["answer"] == "No tengo guardado quién era Eustaquio."
    assert body["audio"] is None
    assert body["audio_content_type"] is None
    assert body["sources"] == []
    assert body["gap"] is not None


async def test_audio_without_stt_degrades_to_text_when_text_is_present(
    app_client: AsyncClient,
) -> None:
    response = await app_client.post(
        API,
        data={"text": "¿Quién era Eustaquio?"},
        files={"audio": ("sample.wav", b"\x00wav", "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] is None
    assert body["text"] == "¿Quién era Eustaquio?"
    assert body["audio"] is None


async def test_audio_only_without_stt_is_rejected(
    app_client: AsyncClient,
) -> None:
    response = await app_client.post(
        API,
        files={"audio": ("sample.wav", b"\x00wav", "audio/wav")},
    )
    assert response.status_code == 400
    assert "no usable text" in response.json()["detail"].lower()


async def test_voice_round_trip_with_injected_providers(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    app.dependency_overrides[get_speech_to_text] = lambda: FakeSTT(
        "¿Quién era Francisco?"
    )
    app.dependency_overrides[get_text_to_speech] = lambda: FakeTTS(b"\x00MP3voice")
    try:
        response = await app_client.post(
            API,
            files={"audio": ("sample.wav", b"\x00wav", "audio/wav")},
        )
    finally:
        app.dependency_overrides.pop(get_speech_to_text, None)
        app.dependency_overrides.pop(get_text_to_speech, None)

    assert response.status_code == 200
    body = response.json()

    assert body["transcript"] == "¿Quién era Francisco?"
    assert body["text"] is None
    assert "amigo de Antonio" in body["answer"]
    assert base64.b64decode(body["audio"]) == b"\x00MP3voice"
    assert body["audio_content_type"] == "audio/mpeg"
    assert body["sources"]


async def test_voice_round_trip_surfaces_gap_and_ambiguity_still(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    app.dependency_overrides[get_speech_to_text] = lambda: FakeSTT("¿Quién era Paco?")
    app.dependency_overrides[get_text_to_speech] = lambda: FakeTTS(b"\x00MP3voice")
    try:
        response = await app_client.post(
            API,
            files={"audio": ("sample.wav", b"\x00wav", "audio/wav")},
        )
    finally:
        app.dependency_overrides.pop(get_speech_to_text, None)
        app.dependency_overrides.pop(get_text_to_speech, None)

    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "¿Quién era Paco?"
    assert body["ambiguity"] is not None


async def test_voice_accepts_browser_media_recorder_mime_parameters(
    app_client: AsyncClient,
) -> None:
    app.dependency_overrides[get_speech_to_text] = lambda: FakeSTT("Hola")
    try:
        response = await app_client.post(
            API,
            files={
                "audio": (
                    "voice.webm",
                    b"\x00webm",
                    "audio/webm;codecs=opus",
                )
            },
        )
    finally:
        app.dependency_overrides.pop(get_speech_to_text, None)

    assert response.status_code == 200
    assert response.json()["transcript"] == "Hola"


async def test_voice_rejects_unsupported_audio_mime(app_client: AsyncClient) -> None:
    response = await app_client.post(
        API,
        files={"audio": ("sample.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported audio format"


async def test_voice_rejects_oversized_audio(
    app_client: AsyncClient, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "voice_max_audio_bytes", 8)
    response = await app_client.post(
        API,
        files={"audio": ("sample.wav", b"123456789", "audio/wav")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Audio upload too large"


async def test_voice_rejects_empty_audio(app_client: AsyncClient) -> None:
    response = await app_client.post(
        API,
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Empty audio upload"


async def test_voice_turn_preserves_ownership_and_conversation_continuity(
    app_client: AsyncClient,
) -> None:
    app.dependency_overrides[get_speech_to_text] = lambda: FakeSTT("Hola")
    try:
        first = await app_client.post(
            API,
            files={"audio": ("voice.wav", b"\x00wav", "audio/wav")},
        )
        assert first.status_code == 200
        conversation_id = first.json()["conversation_id"]

        assert app_client.cookies.get("memo_chat_session") is not None
        assert app_client.cookies.get("memo_chat_conversation") == conversation_id

        second = await app_client.post(
            API,
            data={"conversation_id": conversation_id},
            files={"audio": ("voice.wav", b"\x00wav", "audio/wav")},
        )
    finally:
        app.dependency_overrides.pop(get_speech_to_text, None)

    assert second.status_code == 200
    assert second.json()["conversation_id"] == conversation_id


async def test_voice_turn_does_not_persist_raw_audio(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    app.dependency_overrides[get_speech_to_text] = lambda: FakeSTT("Hola")
    try:
        response = await app_client.post(
            API,
            files={"audio": ("voice.wav", b"\x00wav", "audio/wav")},
        )
    finally:
        app.dependency_overrides.pop(get_speech_to_text, None)

    assert response.status_code == 200
    conversation_id = uuid.UUID(response.json()["conversation_id"])
    messages = (
        (
            await db_session.execute(
                select(Message).where(Message.conversation_id == conversation_id)
            )
        )
        .scalars()
        .all()
    )
    assert messages
    assert all(message.audio_path is None for message in messages)


async def test_voice_turn_then_chat_page_renders_new_messages(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    """A committed voice turn is visible on the server-rendered chat page."""
    app.dependency_overrides[get_speech_to_text] = lambda: FakeSTT(
        "¿Quién era Francisco?"
    )
    try:
        voice = await app_client.post(
            API,
            files={"audio": ("voice.wav", b"\x00wav", "audio/wav")},
        )
    finally:
        app.dependency_overrides.pop(get_speech_to_text, None)

    assert voice.status_code == 200
    conversation_id = voice.json()["conversation_id"]
    assert app_client.cookies.get("memo_chat_session") is not None
    assert app_client.cookies.get("memo_chat_conversation") == conversation_id

    page = await app_client.get("/chat")
    assert page.status_code == 200
    assert "¿Quién era Francisco?" in page.text
    assert "amigo de Antonio" in page.text


async def test_tts_failure_keeps_cookies_and_turn_visible(
    app_client: AsyncClient,
) -> None:
    """An unexpected TTS failure still returns the stored text turn."""
    app.dependency_overrides[get_speech_to_text] = lambda: FakeSTT("Hola")
    app.dependency_overrides[get_text_to_speech] = lambda: ExplodingTTS()
    try:
        voice = await app_client.post(
            API,
            files={"audio": ("voice.wav", b"\x00wav", "audio/wav")},
        )
    finally:
        app.dependency_overrides.pop(get_speech_to_text, None)
        app.dependency_overrides.pop(get_text_to_speech, None)

    assert voice.status_code == 200
    body = voice.json()
    assert body["answer"]
    assert body["audio"] is None
    assert body["audio_content_type"] is None
    conversation_id = body["conversation_id"]
    assert app_client.cookies.get("memo_chat_session") is not None
    assert app_client.cookies.get("memo_chat_conversation") == conversation_id

    page = await app_client.get("/chat")
    assert page.status_code == 200
    assert "Hola" in page.text
    assert body["answer"] in page.text


async def test_voice_relationship_role_uses_active_person(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    """A voice role question resolves through the same conversation pipeline."""
    app.dependency_overrides[get_speech_to_text] = lambda: FakeSTT("¿Quién era Carmen?")
    try:
        first = await app_client.post(
            API,
            files={"audio": ("voice.wav", b"\x00wav", "audio/wav")},
        )
    finally:
        app.dependency_overrides.pop(get_speech_to_text, None)

    assert first.status_code == 200
    conversation_id = first.json()["conversation_id"]

    second = await app_client.post(
        API,
        data={"text": "¿Quién es mi marido?", "conversation_id": conversation_id},
    )
    assert second.status_code == 200
    body = second.json()
    assert "Antonio" in body["answer"]
    assert "marido" in body["answer"]


async def test_unexpected_post_persistence_error_keeps_turn_visible(
    app_client: AsyncClient, monkeypatch
) -> None:
    """A response-assembly bug after commit must not hide the stored turn."""
    from app.api.v1.endpoints import voice as voice_endpoint

    def _broken_response(*_args, **_kwargs):
        raise RuntimeError("unexpected response assembly failure")

    monkeypatch.setattr(voice_endpoint, "_voice_response", _broken_response)
    app.dependency_overrides[get_speech_to_text] = lambda: FakeSTT("Hola")
    try:
        voice = await app_client.post(
            API,
            files={"audio": ("voice.wav", b"\x00wav", "audio/wav")},
        )
    finally:
        app.dependency_overrides.pop(get_speech_to_text, None)

    assert voice.status_code == 200
    body = voice.json()
    assert body["answer"]
    conversation_id = body["conversation_id"]
    assert app_client.cookies.get("memo_chat_session") is not None
    assert app_client.cookies.get("memo_chat_conversation") == conversation_id

    page = await app_client.get("/chat")
    assert page.status_code == 200
    assert "Hola" in page.text
    assert body["answer"] in page.text
