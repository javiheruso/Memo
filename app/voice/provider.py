"""Speech-to-text / text-to-speech wiring through settings.

Mirror of ``app/llm/provider.py`` and ``app/embeddings/provider.py`` for the
optional voice layer. The voice endpoint receives its providers from here so
neither the endpoint nor ``VoiceService`` hardcodes a concrete implementation.
Unlike the responder/embedder seams there is no deterministic speech backend:
a voice feature is only resolved when it is explicitly configured, so
``get_speech_to_text`` / ``get_text_to_speech`` return ``None`` in the default
configuration and the chat surface stays text-first with zero setup.
"""

from app.core.config import get_settings
from app.voice.base import SpeechToText, TextToSpeech
from app.voice.openai_compatible import OpenAICompatibleSTT, OpenAICompatibleTTS

STT_PROVIDERS: dict[str, type[SpeechToText]] = {
    "openai_compatible": OpenAICompatibleSTT,
}

TTS_PROVIDERS: dict[str, type[TextToSpeech]] = {
    "openai_compatible": OpenAICompatibleTTS,
}

_DISABLED = {"", "none", "disabled"}


def get_speech_to_text() -> SpeechToText | None:
    """Resolve the configured STT provider, or ``None`` when not configured."""
    settings = get_settings()
    provider = (settings.stt or "").strip().lower()
    if provider in _DISABLED:
        return None
    try:
        provider_cls = STT_PROVIDERS[provider]
    except KeyError as exc:
        raise ValueError(
            f"Unknown STT provider '{provider}'. "
            f"Known: {', '.join(sorted(STT_PROVIDERS))}."
        ) from exc
    factory = getattr(provider_cls, "from_settings", None)
    if factory is not None:
        return factory(settings)
    return provider_cls()


def get_text_to_speech() -> TextToSpeech | None:
    """Resolve the configured TTS provider, or ``None`` when not configured."""
    settings = get_settings()
    provider = (settings.tts or "").strip().lower()
    if provider in _DISABLED:
        return None
    try:
        provider_cls = TTS_PROVIDERS[provider]
    except KeyError as exc:
        raise ValueError(
            f"Unknown TTS provider '{provider}'. "
            f"Known: {', '.join(sorted(TTS_PROVIDERS))}."
        ) from exc
    factory = getattr(provider_cls, "from_settings", None)
    if factory is not None:
        return factory(settings)
    return provider_cls()
