from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEMO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "memo"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    admin_password: str = ""
    auto_migrate: bool = False
    database_url: str = "postgresql+psycopg://memo:memo@localhost:5432/memo"
    test_database_url: str | None = None
    responder: str = "deterministic"
    embedder: str = "deterministic"
    claim_extractor: str = "deterministic"
    claim_grouper: str = "deterministic"
    claim_contradiction_detector: str = "deterministic"
    llm_base_url: str = ""
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0
    embedder_base_url: str = ""
    embedder_api_key: str | None = None
    embedder_model: str = "text-embedding-3-small"
    embedder_timeout_seconds: float = 30.0
    stt: str = ""
    stt_base_url: str = ""
    stt_api_key: str | None = None
    stt_model: str = "whisper-1"
    stt_timeout_seconds: float = 30.0
    tts: str = ""
    tts_base_url: str = ""
    tts_api_key: str | None = None
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    tts_timeout_seconds: float = 30.0
    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("database_url", "test_database_url")
    @classmethod
    def _ensure_psycopg_driver(cls, value: str | None) -> str | None:
        """Rewrite a bare ``postgres(ql)://`` URL to use the psycopg3 driver.

        Managed Postgres providers (Railway, Supabase, Neon, Render) hand out
        ``postgresql://`` URLs that SQLAlchemy maps to psycopg2 by default; this
        project only installs psycopg3, so the URL must carry the explicit
        ``+psycopg`` driver suffix.
        """
        if value and value.startswith("postgresql://"):
            return "postgresql+psycopg://" + value[len("postgresql://") :]
        if value and value.startswith("postgres://"):
            return "postgresql+psycopg://" + value[len("postgres://") :]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
