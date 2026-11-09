from functools import lru_cache

from pydantic import Field, field_validator, model_validator
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
    # Dedicated signing key for public chat-session cookies. It must never
    # reuse ``admin_password``. Outside development it is required; local
    # development falls back to a fixed non-production key.
    chat_session_secret: str = ""
    # Identifies the active signing key. ``chat_session_previous_keys`` maps
    # retired key ids to their secrets so existing chat sessions survive a
    # rotation until they naturally expire or are dropped by the client.
    chat_session_key_id: str = "v1"
    chat_session_previous_keys: dict[str, str] = Field(default_factory=dict)
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
    # Language hint sent to speech-to-text. Spanish is the product default so a
    # spoken pilot turn does not force the provider to auto-detect the language.
    stt_language: str = "es"
    stt_timeout_seconds: float = 30.0
    # Bound the audio accepted at the application boundary. The browser UI also
    # stops recording after one minute, but a client can always send more bytes.
    voice_max_audio_bytes: int = Field(default=8 * 1024 * 1024, gt=0)
    tts: str = ""
    tts_base_url: str = ""
    tts_api_key: str | None = None
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    tts_timeout_seconds: float = 30.0
    cors_origins: list[str] = ["http://localhost:3000"]

    @model_validator(mode="after")
    def _require_admin_password_outside_development(self) -> "Settings":
        """Fail closed when a non-development process has no admin password.

        Local development keeps the explicit fail-open behaviour documented in
        the README and the default test suite. Any other environment is treated
        as deployable, so the admin UI/API cannot silently start unauthenticated.
        """
        if (
            self.environment.strip().lower() != "development"
            and not self.admin_password.strip()
        ):
            raise ValueError(
                "MEMO_ADMIN_PASSWORD must be set when MEMO_ENVIRONMENT is not "
                "'development'"
            )
        return self

    @model_validator(mode="after")
    def _validate_chat_session_keys(self) -> "Settings":
        """Keep public chat-session signing independent from the admin secret.

        A non-development process must configure a dedicated secret. The
        active key id must be usable in the cookie envelope, and retired key
        ids must be distinct from the active one so a rotation is explicit.
        """
        active_secret = self.chat_session_secret.strip()
        active_key_id = self.chat_session_key_id.strip()
        previous_keys = {
            key_id.strip(): secret.strip()
            for key_id, secret in self.chat_session_previous_keys.items()
        }

        if self.environment.strip().lower() != "development" and not active_secret:
            raise ValueError(
                "MEMO_CHAT_SESSION_SECRET must be set when MEMO_ENVIRONMENT is "
                "not 'development'"
            )

        admin_password = self.admin_password.strip()
        if active_secret and admin_password and active_secret == admin_password:
            raise ValueError(
                "MEMO_CHAT_SESSION_SECRET must differ from MEMO_ADMIN_PASSWORD"
            )
        if any(
            secret and admin_password and secret == admin_password
            for secret in previous_keys.values()
        ):
            raise ValueError(
                "MEMO_CHAT_SESSION_PREVIOUS_KEYS must not reuse MEMO_ADMIN_PASSWORD"
            )

        if active_secret and not active_key_id:
            raise ValueError(
                "MEMO_CHAT_SESSION_KEY_ID must be set when "
                "MEMO_CHAT_SESSION_SECRET is configured"
            )
        if "." in active_key_id:
            raise ValueError("MEMO_CHAT_SESSION_KEY_ID must not contain '.'")
        if active_key_id in previous_keys:
            raise ValueError(
                "MEMO_CHAT_SESSION_PREVIOUS_KEYS must not include the active "
                "MEMO_CHAT_SESSION_KEY_ID"
            )
        for key_id, secret in previous_keys.items():
            if not key_id or not secret:
                raise ValueError(
                    "MEMO_CHAT_SESSION_PREVIOUS_KEYS entries need a non-empty "
                    "key id and secret"
                )
            if "." in key_id:
                raise ValueError(
                    "MEMO_CHAT_SESSION_PREVIOUS_KEYS key ids must not contain '.'"
                )
        return self

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
