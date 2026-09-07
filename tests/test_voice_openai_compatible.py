import httpx
import pytest

from app.voice.openai_compatible import (
    InvalidSpeech,
    InvalidTranscription,
    OpenAICompatibleSTT,
    OpenAICompatibleTTS,
)


def _stt_client(
    *,
    payload: dict | None = None,
    text: str | None = None,
    status: int = 200,
    error: Exception | None = None,
    capture: list | None = None,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        if error is not None:
            raise error
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=payload or {"text": "hola"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _tts_client(
    *,
    content: bytes = b"\x00audio",
    status: int = 200,
    error: Exception | None = None,
    capture: list | None = None,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        if error is not None:
            raise error
        return httpx.Response(status, content=content)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _stt(client: httpx.AsyncClient, **kwargs) -> OpenAICompatibleSTT:
    return OpenAICompatibleSTT(
        base_url=kwargs.pop("base_url", "http://voice.test"),
        model=kwargs.pop("model", "whisper-1"),
        api_key=kwargs.pop("api_key", "secret"),
        client=client,
        **kwargs,
    )


def _tts(client: httpx.AsyncClient, **kwargs) -> OpenAICompatibleTTS:
    return OpenAICompatibleTTS(
        base_url=kwargs.pop("base_url", "http://voice.test"),
        model=kwargs.pop("model", "tts-1"),
        api_key=kwargs.pop("api_key", "secret"),
        client=client,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_stt_returns_transcribed_text() -> None:
    async with _stt_client(payload={"text": "¿Quién era Francisco?"}) as client:
        result = await _stt(client).transcribe(b"\x00wav", content_type="audio/wav")
    assert result == "¿Quién era Francisco?"


@pytest.mark.asyncio
async def test_stt_request_targets_transcriptions_with_model_and_auth() -> None:
    capture: list[httpx.Request] = []
    async with _stt_client(payload={"text": "hola"}, capture=capture) as client:
        await _stt(client).transcribe(b"\x00wav", content_type="audio/wav")
    request = capture[0]
    assert request.url == "http://voice.test/audio/transcriptions"
    assert request.headers["Authorization"] == "Bearer secret"
    body = request.content
    assert b'name="model"' in body
    assert b"whisper-1" in body
    assert b'name="file"' in body
    assert b"\x00wav" in body


@pytest.mark.asyncio
async def test_stt_strips_surrounding_whitespace() -> None:
    async with _stt_client(payload={"text": "  hola  "}) as client:
        result = await _stt(client).transcribe(b"\x00")
    assert result == "hola"


@pytest.mark.asyncio
async def test_stt_empty_text_is_invalid() -> None:
    async with _stt_client(payload={"text": ""}) as client:
        with pytest.raises(InvalidTranscription):
            await _stt(client).transcribe(b"\x00")


@pytest.mark.asyncio
async def test_stt_non_json_body_is_invalid() -> None:
    async with _stt_client(text="oops") as client:
        with pytest.raises(InvalidTranscription):
            await _stt(client).transcribe(b"\x00")


@pytest.mark.asyncio
async def test_stt_response_without_text_key_is_invalid() -> None:
    async with _stt_client(payload={"foo": "bar"}) as client:
        with pytest.raises(InvalidTranscription):
            await _stt(client).transcribe(b"\x00")


@pytest.mark.asyncio
async def test_stt_http_error_propagates() -> None:
    async with _stt_client(status=500) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await _stt(client).transcribe(b"\x00")


@pytest.mark.asyncio
async def test_stt_network_error_propagates() -> None:
    async with _stt_client(error=httpx.ConnectError("connection refused")) as client:
        with pytest.raises(httpx.HTTPError):
            await _stt(client).transcribe(b"\x00")


@pytest.mark.asyncio
async def test_tts_returns_audio_bytes() -> None:
    async with _tts_client(content=b"\x00MP3") as client:
        result = await _tts(client).synthesize("hola")
    assert result == b"\x00MP3"


@pytest.mark.asyncio
async def test_tts_request_targets_speech_with_model_voice_and_input() -> None:
    import json

    capture: list[httpx.Request] = []
    async with _tts_client(content=b"\x00MP3", capture=capture) as client:
        await _tts(client, voice="nova").synthesize("¿Quién era Francisco?")
    request = capture[0]
    assert request.url == "http://voice.test/audio/speech"
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["model"] == "tts-1"
    assert body["voice"] == "nova"
    assert body["input"] == "¿Quién era Francisco?"


@pytest.mark.asyncio
async def test_tts_empty_audio_body_is_invalid() -> None:
    async with _tts_client(content=b"") as client:
        with pytest.raises(InvalidSpeech):
            await _tts(client).synthesize("hola")


@pytest.mark.asyncio
async def test_tts_http_error_propagates() -> None:
    async with _tts_client(status=500) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await _tts(client).synthesize("hola")


@pytest.mark.asyncio
async def test_tts_network_error_propagates() -> None:
    async with _tts_client(error=httpx.ConnectError("connection refused")) as client:
        with pytest.raises(httpx.HTTPError):
            await _tts(client).synthesize("hola")


def test_stt_base_url_and_model_are_required() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleSTT(base_url="", model="m")
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleSTT(base_url="http://voice.test", model="")


def test_tts_base_url_model_and_voice_are_required() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleTTS(base_url="", model="m")
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleTTS(base_url="http://voice.test", model="")
    with pytest.raises(ValueError, match="voice"):
        OpenAICompatibleTTS(base_url="http://voice.test", model="m", voice="")


def test_stt_from_settings_maps_config() -> None:
    from app.core.config import Settings

    stt = OpenAICompatibleSTT.from_settings(
        Settings(
            stt="openai_compatible",
            stt_base_url="http://local",
            stt_model="whisper-1",
            stt_api_key=None,
            stt_timeout_seconds=5,
        )
    )
    assert stt._base_url == "http://local"
    assert stt._model == "whisper-1"
    assert stt._api_key is None
    assert stt._timeout == 5


def test_tts_from_settings_maps_config() -> None:
    from app.core.config import Settings

    tts = OpenAICompatibleTTS.from_settings(
        Settings(
            tts="openai_compatible",
            tts_base_url="http://local",
            tts_model="tts-1",
            tts_api_key="k",
            tts_voice="nova",
            tts_timeout_seconds=5,
        )
    )
    assert tts._base_url == "http://local"
    assert tts._model == "tts-1"
    assert tts._voice == "nova"
    assert tts._api_key == "k"
    assert tts._timeout == 5
