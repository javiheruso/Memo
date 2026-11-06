"""Public chat-session cookies must never become an admin signing oracle."""

import hashlib
import hmac
import uuid

import pytest
from starlette.requests import Request

from app.core.config import settings
from app.security import (
    CHAT_SESSION_COOKIE_NAME,
    issue_chat_session_cookie,
    read_chat_session_id,
    resolve_chat_session_id,
)


def _configure_chat_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    key_id: str,
    secret: str,
    previous_keys: dict[str, str] | None = None,
    admin_password: str = "admin-only-secret",
) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "admin_password", admin_password)
    monkeypatch.setattr(settings, "chat_session_secret", secret)
    monkeypatch.setattr(settings, "chat_session_key_id", key_id)
    monkeypatch.setattr(settings, "chat_session_previous_keys", previous_keys or {})


def _admin_signed_cookie(key_id: str, session_id: uuid.UUID, password: str) -> str:
    payload = f"memo-chat-session|{key_id}|{session_id}"
    signature = hmac.new(
        password.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"{key_id}.{session_id}.{signature}"


def _request_with_chat_cookie(value: str) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "path": "/chat",
            "raw_path": b"/chat",
            "query_string": b"",
            "headers": [
                (b"cookie", f"{CHAT_SESSION_COOKIE_NAME}={value}".encode("ascii"))
            ],
            "client": ("test", 12345),
            "server": ("test", 80),
            "scheme": "http",
        }
    )


def test_configured_chat_cookie_does_not_validate_with_admin_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_chat_session(
        monkeypatch,
        key_id="v1",
        secret="dedicated-chat-secret",
    )
    session_id = uuid.uuid4()

    value = issue_chat_session_cookie(session_id)
    assert value.startswith("v1.")
    assert read_chat_session_id(value) == session_id

    forged = _admin_signed_cookie("v1", session_id, "admin-only-secret")
    assert forged != value
    assert read_chat_session_id(forged) is None


def test_development_fallback_does_not_reuse_admin_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "admin_password", "admin-only-secret")
    monkeypatch.setattr(settings, "chat_session_secret", "")
    monkeypatch.setattr(settings, "chat_session_key_id", "v1")
    monkeypatch.setattr(settings, "chat_session_previous_keys", {})
    session_id = uuid.uuid4()

    value = issue_chat_session_cookie(session_id)
    key_id = value.split(".", 1)[0]
    assert key_id == "development"
    assert read_chat_session_id(value) == session_id

    forged = _admin_signed_cookie(key_id, session_id, "admin-only-secret")
    assert read_chat_session_id(forged) is None


def test_key_rotation_accepts_configured_previous_key_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = uuid.uuid4()
    _configure_chat_session(
        monkeypatch,
        key_id="v1",
        secret="first-secret",
    )
    old_cookie = issue_chat_session_cookie(session_id)

    _configure_chat_session(
        monkeypatch,
        key_id="v2",
        secret="second-secret",
        previous_keys={"v1": "first-secret"},
    )
    new_cookie = issue_chat_session_cookie(session_id)

    assert new_cookie.startswith("v2.")
    assert read_chat_session_id(new_cookie) == session_id
    assert read_chat_session_id(old_cookie) == session_id

    _configure_chat_session(
        monkeypatch,
        key_id="v2",
        secret="second-secret",
        previous_keys={},
    )
    assert read_chat_session_id(old_cookie) is None
    assert resolve_chat_session_id(_request_with_chat_cookie(old_cookie)) != session_id


def test_unknown_key_id_or_invalid_signature_mints_an_anonymous_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = uuid.uuid4()
    _configure_chat_session(
        monkeypatch,
        key_id="v2",
        secret="second-secret",
        previous_keys={"v1": "first-secret"},
    )
    valid = issue_chat_session_cookie(session_id)
    key_id, raw, signature = valid.split(".")
    unknown_key = _admin_signed_cookie("retired", session_id, "first-secret")
    tampered = f"{key_id}.{raw}.{'0' if signature[0] != '0' else '1'}{signature[1:]}"

    assert read_chat_session_id(unknown_key) is None
    assert read_chat_session_id(tampered) is None
    assert resolve_chat_session_id(_request_with_chat_cookie(unknown_key)) != session_id
    assert resolve_chat_session_id(_request_with_chat_cookie(tampered)) != session_id
    assert resolve_chat_session_id(_request_with_chat_cookie(valid)) == session_id
