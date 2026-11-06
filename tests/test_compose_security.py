"""Compose must fail closed instead of shipping unsafe defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "compose.yaml"
CADDYFILE = ROOT / "docker" / "caddy" / "Caddyfile"


def _compose_text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def _caddyfile_text() -> str:
    return CADDYFILE.read_text(encoding="utf-8")


def test_compose_requires_database_password() -> None:
    text = _compose_text()
    assert "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}" in text
    assert "${POSTGRES_PASSWORD:-memo}" not in text


def test_compose_publishes_postgres_on_loopback_only() -> None:
    text = _compose_text()
    assert '"127.0.0.1:${POSTGRES_PORT:-5432}:5432"' in text
    assert '"${POSTGRES_PORT:-5432}:5432"' not in text


def test_compose_publishes_app_on_loopback_only() -> None:
    text = _compose_text()
    assert '"127.0.0.1:${APP_PORT:-8000}:8000"' in text
    assert '"${APP_PORT:-8000}:8000"' not in text
    assert "${APP_PORT:-8000}" in text


def test_compose_app_uses_the_required_database_password() -> None:
    text = _compose_text()
    database_url_line = next(
        line for line in text.splitlines() if "MEMO_DATABASE_URL:" in line
    )
    assert "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}" in database_url_line
    assert "${POSTGRES_PASSWORD:-memo}" not in database_url_line


def test_compose_requires_chat_session_secret() -> None:
    text = _compose_text()
    assert "${MEMO_CHAT_SESSION_SECRET:?MEMO_CHAT_SESSION_SECRET must be set}" in text


def test_compose_publishes_caddy_https_on_loopback_only() -> None:
    text = _compose_text()
    assert "image: caddy:2-alpine" in text
    assert '"127.0.0.1:${CADDY_HTTPS_PORT:-8443}:8443"' in text
    assert '"${CADDY_HTTPS_PORT:-8443}:8443"' not in text
    assert '"8443:8443"' not in text


def test_compose_mounts_caddyfile_and_persists_caddy_state() -> None:
    text = _compose_text()
    assert "./docker/caddy/Caddyfile:/etc/caddy/Caddyfile:ro" in text
    assert "caddy_data:/data" in text
    assert "caddy_config:/config" in text
    assert "  caddy_data:" in text
    assert "  caddy_config:" in text


def test_caddyfile_uses_local_tls_and_proxies_to_the_app() -> None:
    text = _caddyfile_text()
    assert "memo.localhost:8443 {" in text
    assert "tls internal" in text
    assert "reverse_proxy app:8000" in text
