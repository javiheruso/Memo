"""Configuration URL normalization for managed Postgres providers."""

from app.core.config import Settings


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
