"""Public chat assets must be cache-busted by content, not by app version.

A deployment can change ``voice.js``, ``voice-silence.js`` or ``voice.css``
while ``app_version`` stays the same. The rendered page therefore appends a
short digest of the file bytes to each static URL, so an already-open browser
requests the new asset instead of reusing a stale cached response.
"""

import hashlib
from pathlib import Path

from httpx import AsyncClient

from app.web.assets import STATIC_DIR, asset_url

VOICE_ASSETS = ("voice-silence.js", "voice.js", "voice.css")


def _expected_token(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


async def test_chat_page_fingerprints_each_voice_asset(
    app_client: AsyncClient,
) -> None:
    first = await app_client.get("/chat")
    assert first.status_code == 200

    for name in VOICE_ASSETS:
        expected = f"/static/{name}?v={_expected_token(STATIC_DIR / name)}"
        assert expected in first.text

    # Rendering is deterministic: unchanged files keep the same URL.
    second = await app_client.get("/chat")
    assert second.status_code == 200
    for name in VOICE_ASSETS:
        expected = f"/static/{name}?v={_expected_token(STATIC_DIR / name)}"
        assert expected in second.text


def test_asset_url_changes_only_when_content_changes(tmp_path: Path) -> None:
    asset = tmp_path / "voice.js"
    asset.write_text("first-version", encoding="utf-8")

    first = asset_url("voice.js", static_dir=tmp_path)
    assert first == (
        f"/static/voice.js?v={hashlib.sha256(b'first-version').hexdigest()[:12]}"
    )
    assert asset_url("voice.js", static_dir=tmp_path) == first

    # A different length guarantees a distinct (mtime, size) cache key even on
    # filesystems with coarse timestamp resolution.
    asset.write_text("second-version-with-more-bytes", encoding="utf-8")
    second = asset_url("voice.js", static_dir=tmp_path)

    assert second != first
    assert second.endswith(
        hashlib.sha256(b"second-version-with-more-bytes").hexdigest()[:12]
    )
