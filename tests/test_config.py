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


def test_development_keeps_explicit_fail_open_mode() -> None:
    configured = Settings(environment="development", admin_password="")
    assert configured.admin_password == ""


async def test_lifespan_refuses_to_start_without_admin_password_outside_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "admin_password", "")
    with pytest.raises(RuntimeError, match="MEMO_ADMIN_PASSWORD"):
        async with app.router.lifespan_context(app):
            pass
