"""Real OpenAI-compatible STT/TTS providers behind the voice registry.

Mirror of ``app/llm/openai_compatible.py`` and
``app/embeddings/openai_compatible.py`` for the optional voice layer: speech
input hits ``/audio/transcriptions`` and speech output hits ``/audio/speech``.
There is no deterministic fallback for speech, so "fail closed" means a
provider/network problem or a response that cannot be trusted (empty or
malformed transcription, empty audio body) raises a typed error instead of
silently returning garbage. ``VoiceService`` catches those errors and degrades
the turn to text-only (no transcript, no audio), so a voice endpoint mistake
can never crash the companion or inject invented words or audio into a reply.
"""

from __future__ import annotations

import httpx

from app.core.config import Settings
from app.voice.audio import (
    DEFAULT_AUDIO_CONTENT_TYPE,
    audio_file_extension,
    normalize_audio_content_type,
)
from app.voice.base import SpeechToText, TextToSpeech

_MULTILINGUAL_LANGUAGES_MODEL = "gpt-transcribe"


def _uses_multilingual_languages(model: str) -> bool:
    """Return whether ``model`` expects the ``languages[]`` field.

    ``gpt-transcribe`` (and its dated snapshots) replaced the legacy singular
    ``language`` hint with a multilingual ``languages[]`` array. Every other
    OpenAI-compatible model keeps sending the singular field.
    """
    return model == _MULTILINGUAL_LANGUAGES_MODEL or model.startswith(
        f"{_MULTILINGUAL_LANGUAGES_MODEL}-"
    )


class InvalidTranscription(ValueError):
    """Raised when a transcription response cannot be trusted as recognized text."""


class InvalidSpeech(ValueError):
    """Raised when a speech response carries no usable audio."""


class OpenAICompatibleSTT(SpeechToText):
    """Speech-to-text backed by any OpenAI-compatible transcriptions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        language: str | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError(
                "base_url is required for the OpenAI-compatible STT provider"
            )
        if not model:
            raise ValueError("model is required for the OpenAI-compatible STT provider")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._language = (language or "").strip()
        self._timeout = timeout_seconds
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleSTT:
        return cls(
            base_url=settings.stt_base_url,
            model=settings.stt_model,
            api_key=settings.stt_api_key,
            language=settings.stt_language,
            timeout_seconds=settings.stt_timeout_seconds,
        )

    async def transcribe(self, audio: bytes, *, content_type: str | None = None) -> str:
        url = f"{self._base_url}/audio/transcriptions"
        normalized_content_type = normalize_audio_content_type(content_type)
        filename = _upload_filename(normalized_content_type)
        files = {
            "file": (
                filename,
                audio,
                normalized_content_type or DEFAULT_AUDIO_CONTENT_TYPE,
            )
        }
        data = {"model": self._model}
        if self._language:
            if _uses_multilingual_languages(self._model):
                data["languages[]"] = self._language
            else:
                data["language"] = self._language
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        if self._client is not None:
            response = await self._client.post(
                url, headers=headers, data=data, files=files
            )
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    url, headers=headers, data=data, files=files
                )
        response.raise_for_status()

        try:
            text = response.json().get("text")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise InvalidTranscription(
                "transcription response did not match the expected shape"
            ) from exc
        if not isinstance(text, str) or not text.strip():
            raise InvalidTranscription("transcription response is empty")
        return text.strip()


class OpenAICompatibleTTS(TextToSpeech):
    """Text-to-speech backed by any OpenAI-compatible speech endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        voice: str = "alloy",
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError(
                "base_url is required for the OpenAI-compatible TTS provider"
            )
        if not model:
            raise ValueError("model is required for the OpenAI-compatible TTS provider")
        if not voice:
            raise ValueError("voice is required for the OpenAI-compatible TTS provider")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._voice = voice
        self._timeout = timeout_seconds
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleTTS:
        return cls(
            base_url=settings.tts_base_url,
            model=settings.tts_model,
            api_key=settings.tts_api_key,
            voice=settings.tts_voice,
            timeout_seconds=settings.tts_timeout_seconds,
        )

    async def synthesize(self, text: str) -> bytes:
        url = f"{self._base_url}/audio/speech"
        payload = {"model": self._model, "input": text, "voice": self._voice}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if self._client is not None:
            response = await self._client.post(url, headers=headers, json=payload)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()

        content = response.content
        if not content:
            raise InvalidSpeech("speech response is empty")
        return content


def _upload_filename(content_type: str | None) -> str:
    extension = audio_file_extension(content_type)
    return f"audio.{extension}" if extension else "audio"
