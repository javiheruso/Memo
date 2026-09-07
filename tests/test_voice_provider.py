import pytest

from app.voice.provider import (
    STT_PROVIDERS,
    TTS_PROVIDERS,
    get_speech_to_text,
    get_text_to_speech,
)


def test_provider_defaults_to_unconfigured() -> None:
    assert get_speech_to_text() is None
    assert get_text_to_speech() is None


def test_provider_returns_none_for_explicit_disabled() -> None:
    from app.voice.provider import _DISABLED

    assert "none" in _DISABLED
    assert "disabled" in _DISABLED


def test_provider_registers_openai_compatible() -> None:
    from app.voice.openai_compatible import OpenAICompatibleSTT, OpenAICompatibleTTS

    assert STT_PROVIDERS["openai_compatible"] is OpenAICompatibleSTT
    assert TTS_PROVIDERS["openai_compatible"] is OpenAICompatibleTTS


def test_provider_resolves_stt_from_settings(monkeypatch) -> None:
    from app.voice.openai_compatible import OpenAICompatibleSTT

    class _FakeSettings:
        stt = "openai_compatible"
        stt_base_url = "http://local"
        stt_model = "whisper-1"
        stt_api_key = "k"
        stt_timeout_seconds = 5.0

    monkeypatch.setattr("app.voice.provider.get_settings", lambda: _FakeSettings())
    stt = get_speech_to_text()
    assert isinstance(stt, OpenAICompatibleSTT)
    assert stt._base_url == "http://local"
    assert stt._model == "whisper-1"
    assert stt._api_key == "k"
    assert stt._timeout == 5.0


def test_provider_resolves_tts_from_settings(monkeypatch) -> None:
    from app.voice.openai_compatible import OpenAICompatibleTTS

    class _FakeSettings:
        tts = "openai_compatible"
        tts_base_url = "http://local"
        tts_model = "tts-1"
        tts_api_key = None
        tts_voice = "nova"
        tts_timeout_seconds = 5.0

    monkeypatch.setattr("app.voice.provider.get_settings", lambda: _FakeSettings())
    tts = get_text_to_speech()
    assert isinstance(tts, OpenAICompatibleTTS)
    assert tts._base_url == "http://local"
    assert tts._model == "tts-1"
    assert tts._voice == "nova"
    assert tts._timeout == 5.0


def test_provider_ignores_case_and_whitespace(monkeypatch) -> None:
    class _FakeSettings:
        stt = "  OPENAI_COMPATIBLE "
        stt_base_url = "http://local"
        stt_model = "whisper-1"
        stt_api_key = None
        stt_timeout_seconds = 30.0
        tts = ""
        tts_base_url = ""
        tts_model = "tts-1"
        tts_api_key = None
        tts_voice = "alloy"
        tts_timeout_seconds = 30.0

    monkeypatch.setattr("app.voice.provider.get_settings", lambda: _FakeSettings())
    assert get_speech_to_text() is not None
    assert get_text_to_speech() is None


@pytest.mark.parametrize(
    "field,resolver",
    [
        ("stt", get_speech_to_text),
        ("tts", get_text_to_speech),
    ],
)
def test_provider_rejects_unknown_provider(monkeypatch, field, resolver) -> None:
    class _FakeSettings:
        stt = "nonexistent"
        tts = "nonexistent"

    monkeypatch.setattr("app.voice.provider.get_settings", lambda: _FakeSettings())
    with pytest.raises(ValueError, match=f"Unknown {field.upper()} provider"):
        resolver()
