"""Voice provider seams.

The voice endpoint and ``VoiceService`` receive their speech providers from
``app/voice/provider`` (mirroring ``app/llm/provider`` and
``app/embeddings/provider``) so neither hardcodes a concrete implementation.
Unlike the responder/embedder seams there is no deterministic speech
implementation: a voice feature is either backed by a configured real provider
or absent. Providers therefore resolve to ``None`` when unconfigured and text
mode stays the zero-config default.
"""

from typing import Protocol


class SpeechToText(Protocol):
    """Replaceable audio-to-text recognizer.

    Implementations receive raw audio bytes and the declared media type (when
    the client sent one) and return the recognized utterance. They must not
    invent text: a provider or validation failure surfaces as an exception that
    ``VoiceService`` turns into a text-only degradation.
    """

    async def transcribe(
        self, audio: bytes, *, content_type: str | None = None
    ) -> str: ...


class TextToSpeech(Protocol):
    """Replaceable text-to-speech synthesizer.

    Implementations receive the reply text and return the synthesized audio
    bytes (MP3 by default for OpenAI-compatible endpoints). They must not
    fabricate audio: a provider or validation failure surfaces as an exception
    that ``VoiceService`` turns into a text-only reply with no audio.
    """

    async def synthesize(self, text: str) -> bytes: ...
