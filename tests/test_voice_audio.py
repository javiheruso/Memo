from app.voice.audio import (
    audio_file_extension,
    is_allowed_audio_content_type,
    normalize_audio_content_type,
)


def test_normalize_strips_media_recorder_parameters_and_case() -> None:
    assert normalize_audio_content_type("Audio/WebM; codecs=opus") == "audio/webm"
    assert normalize_audio_content_type(" audio/ogg ; codecs=opus ") == "audio/ogg"


def test_normalize_handles_missing_and_empty_values() -> None:
    assert normalize_audio_content_type(None) is None
    assert normalize_audio_content_type("") is None
    assert normalize_audio_content_type("   ") is None


def test_allow_list_uses_the_normalized_media_type() -> None:
    assert is_allowed_audio_content_type("audio/webm;codecs=opus")
    assert is_allowed_audio_content_type("audio/wav")
    assert not is_allowed_audio_content_type("text/plain")
    assert not is_allowed_audio_content_type("application/octet-stream")
    assert not is_allowed_audio_content_type(None)


def test_extension_mapping_uses_the_normalized_media_type() -> None:
    assert audio_file_extension("audio/webm;codecs=opus") == "webm"
    assert audio_file_extension("audio/ogg;codecs=opus") == "ogg"
    assert audio_file_extension("audio/mp4") == "m4a"
    assert audio_file_extension("text/plain") is None
