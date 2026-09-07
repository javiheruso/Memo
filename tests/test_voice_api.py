import base64

from httpx import AsyncClient

from app.main import app
from app.voice.provider import get_speech_to_text, get_text_to_speech
from tests.fixtures.universe import SeedUniverse

API = "/api/v1/chat/voice"


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


async def test_text_only_voice_call_degrades_to_chat(
    app_client: AsyncClient,
) -> None:
    response = await app_client.post(
        API, data={"text": "¿Quién era Eustaquio?"}
    )
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
    app.dependency_overrides[get_speech_to_text] = lambda: FakeSTT(
        "¿Quién era Paco?"
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
    assert body["transcript"] == "¿Quién era Paco?"
    assert body["ambiguity"] is not None
