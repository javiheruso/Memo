"""Configuration normalization and production startup guards."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.main import app


def test_bare_postgresql_url_gains_psycopg_driver() -> None:
    settings = Settings(database_url="postgresql://user:pw@host:5432/db")
    assert settings.database_url == "postgresql+psycopg://user:pw@host:5432/db"


def test_postgres_url_gains_psycopg_driver() -> None:
    settings = Settings(database_url="postgres://user:pw@host:5432/db")
    assert settings.database_url == "postgresql+psycopg://user:pw@host:5432/db"


def test_explicit_psycopg_url_is_unchanged() -> None:
    url = "postgresql+psycopg://user:pw@host:5432/db"
    assert Settings(database_url=url).database_url == url


def test_other_dialects_are_unchanged() -> None:
    url = "sqlite+aiosqlite:///:memory:"
    assert Settings(database_url=url).database_url == url


def test_test_database_url_is_normalized_too() -> None:
    settings = Settings(test_database_url="postgresql://user:pw@host:5432/db")
    assert settings.test_database_url == "postgresql+psycopg://user:pw@host:5432/db"


def test_production_requires_admin_password() -> None:
    with pytest.raises(ValidationError, match="MEMO_ADMIN_PASSWORD"):
        Settings(environment="production", admin_password="")


def test_non_development_environment_requires_admin_password() -> None:
    with pytest.raises(ValidationError, match="MEMO_ADMIN_PASSWORD"):
        Settings(environment="staging", admin_password="   ")


def test_production_requires_chat_session_secret() -> None:
    with pytest.raises(ValidationError, match="MEMO_CHAT_SESSION_SECRET"):
        Settings(
            environment="production",
            admin_password="a-strong-password",
            chat_session_secret="",
        )


def test_non_development_environment_requires_chat_session_secret() -> None:
    with pytest.raises(ValidationError, match="MEMO_CHAT_SESSION_SECRET"):
        Settings(
            environment="staging",
            admin_password="a-strong-password",
            chat_session_secret="   ",
        )


def test_chat_session_secret_must_not_reuse_the_admin_password() -> None:
    with pytest.raises(ValidationError, match="MEMO_CHAT_SESSION_SECRET"):
        Settings(
            environment="production",
            admin_password="same-secret",
            chat_session_secret="same-secret",
        )


def test_previous_chat_session_keys_must_not_include_the_active_key_id() -> None:
    with pytest.raises(ValidationError, match="MEMO_CHAT_SESSION_PREVIOUS_KEYS"):
        Settings(
            environment="production",
            admin_password="a-strong-password",
            chat_session_secret="a-strong-chat-session-secret",
            chat_session_key_id="v2",
            chat_session_previous_keys={"v2": "retired-secret"},
        )


def test_production_accepts_security_secrets() -> None:
    configured = Settings(
        environment="production",
        admin_password="a-strong-password",
        chat_session_secret="a-strong-chat-session-secret",
    )
    assert configured.admin_password == "a-strong-password"
    assert configured.chat_session_secret == "a-strong-chat-session-secret"


def test_development_keeps_explicit_fail_open_mode() -> None:
    configured = Settings(environment="development", admin_password="")
    assert configured.admin_password == ""
    assert configured.chat_session_secret == ""


async def test_lifespan_refuses_to_start_without_admin_password_outside_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "admin_password", "")
    with pytest.raises(RuntimeError, match="MEMO_ADMIN_PASSWORD"):
        async with app.router.lifespan_context(app):
            pass


async def test_lifespan_refuses_to_start_without_chat_session_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "admin_password", "a-strong-password")
    monkeypatch.setattr(settings, "chat_session_secret", "")
    with pytest.raises(RuntimeError, match="MEMO_CHAT_SESSION_SECRET"):
        async with app.router.lifespan_context(app):
            pass
