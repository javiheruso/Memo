"""Audio upload constraints and content-type normalization.

Browsers report the output of ``MediaRecorder`` with parameters, for example
``audio/webm;codecs=opus``. The parameters are transport hints, not part of the
media type. The public boundary and the provider mapping therefore normalize
the value first: an allow-list check must not be bypassable with a parameter
suffix, and the provider must receive a deterministic filename/content type.
"""

from __future__ import annotations

DEFAULT_AUDIO_CONTENT_TYPE = "application/octet-stream"

AUDIO_EXTENSIONS: dict[str, str] = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mpga": "mpga",
}

ALLOWED_AUDIO_CONTENT_TYPES = frozenset(AUDIO_EXTENSIONS)


def normalize_audio_content_type(value: str | None) -> str | None:
    """Return the bare media type, with parameters removed and case folded.

    ``"Audio/WebM; codecs=opus"`` -> ``"audio/webm"``.
    """
    if value is None:
        return None
    media_type = value.split(";", 1)[0].strip().lower()
    return media_type or None


def is_allowed_audio_content_type(value: str | None) -> bool:
    return normalize_audio_content_type(value) in ALLOWED_AUDIO_CONTENT_TYPES


def audio_file_extension(value: str | None) -> str | None:
    return AUDIO_EXTENSIONS.get(normalize_audio_content_type(value) or "")
