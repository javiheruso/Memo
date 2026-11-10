"""Deterministic cache-busting for same-origin static assets.

A deployment can replace ``voice.js``, ``voice-silence.js`` or ``voice.css``
without bumping ``app_version``, so the public chat template must not rely on a
hand-maintained version string. Each asset URL carries a short digest of the
file bytes instead: changed content gets a new URL and unchanged content stays
cacheable across requests and deployments.

The served path is intentionally unchanged (``/static/<name>``); the ``?v=``
token is enough for browsers to store the old and new responses separately.
"""

import hashlib
from functools import cache
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
_DIGEST_LENGTH = 12
_CHUNK_SIZE = 64 * 1024


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()[:_DIGEST_LENGTH]


@cache
def _cached_digest(path: str, _mtime_ns: int, _size: int) -> str:
    """Hash once per (path, mtime, size) tuple.

    The mtime and size are cache keys only, so editing an asset during local
    development invalidates the entry without hashing on every page render.
    """
    return _file_digest(Path(path))


def asset_url(filename: str, *, static_dir: Path | None = None) -> str:
    """Return the cache-busted URL for a static asset.

    ``static_dir`` exists for tests; callers use the module's ``STATIC_DIR``.
    """
    base = STATIC_DIR if static_dir is None else static_dir
    path = base / filename
    stat = path.stat()
    digest = _cached_digest(str(path), stat.st_mtime_ns, stat.st_size)
    return f"/static/{filename}?v={digest}"
