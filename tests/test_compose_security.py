"""Compose must fail closed instead of shipping unsafe defaults."""

from pathlib import Path

COMPOSE = Path(__file__).resolve().parents[1] / "compose.yaml"


def _compose_text() -> str:
    return COMPOSE.read_text(encoding="utf-8")


def test_compose_requires_database_password() -> None:
    text = _compose_text()
    assert "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}" in text
    assert "${POSTGRES_PASSWORD:-memo}" not in text


def test_compose_publishes_postgres_on_loopback_only() -> None:
    text = _compose_text()
    assert '"127.0.0.1:${POSTGRES_PORT:-5432}:5432"' in text
    assert '"${POSTGRES_PORT:-5432}:5432"' not in text


def test_compose_app_uses_the_required_database_password() -> None:
    text = _compose_text()
    database_url_line = next(
        line for line in text.splitlines() if "MEMO_DATABASE_URL:" in line
    )
    assert "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}" in database_url_line
    assert "${POSTGRES_PASSWORD:-memo}" not in database_url_line
