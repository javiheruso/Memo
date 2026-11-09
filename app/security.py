"""Minimal shared admin authentication.

A single shared admin password gates both the server-rendered admin UI
(``/admin``) and the JSON admin API (``/api/v1/admin``).

- When ``MEMO_ADMIN_PASSWORD`` is unset or empty the surface is fail-open:
  every admin route works without credentials, preserving local/dev
  ergonomics and the default test suite.
- When set, the browser flow requires a signed session cookie issued by
  ``POST /admin/login``, and the API flow requires the password itself as an
  ``Authorization: Bearer <password>`` header.

There are no user accounts, no JWT, no database sessions and no new
dependencies: the cookie is an HMAC-SHA256-signed expiry timestamp keyed with
the admin password (stdlib ``hmac``), and the bearer token is the password
compared in constant time.

The public chat surface separately uses ``memo_chat_session`` to prove that a
browser owns a conversation. That cookie is signed with a dedicated
``MEMO_CHAT_SESSION_SECRET`` and never with the admin password, so the public
endpoints cannot be used as a signing oracle for the admin credential. Every
cookie carries an explicit key id so a deployment can rotate the signing key
while still accepting unexpired cookies signed by configured prior keys.
"""

import hashlib
import hmac
import time
import uuid
from typing import Annotated

from fastapi import Header, HTTPException, Request, Response

from app.core.config import settings

SESSION_COOKIE_NAME = "memo_admin_session"
SESSION_TTL_SECONDS = 12 * 60 * 60
CHAT_SESSION_COOKIE_NAME = "memo_chat_session"
CHAT_CONVERSATION_COOKIE_NAME = "memo_chat_conversation"
_DEV_CHAT_SESSION_KEY_ID = "development"
_DEV_CHAT_SESSION_SECRET = "memo-development-chat-session-secret"
_LOGIN_URL = "/admin/login"


class LoginRequired(Exception):
    """Raised by the UI dependency when a valid session cookie is missing.

    Converted into a redirect to ``/admin/login`` by an app-level exception
    handler so browser requests never see a bare 401/403.
    """


def admin_auth_enabled() -> bool:
    """True when a shared admin password is configured (auth is on)."""
    return bool(settings.admin_password)


def cookie_secure() -> bool:
    """True when session cookies should carry the ``Secure`` flag.

    Local development runs over plain HTTP, so ``Secure`` is omitted there;
    any non-development environment is assumed to sit behind TLS and gets the
    flag.
    """
    return settings.environment != "development"


def verify_password(submitted: str) -> bool:
    """Constant-time check of a submitted password against the configured one."""
    return _constant_time_eq(submitted, settings.admin_password)


def _constant_time_eq(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _sign(password: str, payload: str) -> str:
    return hmac.new(
        password.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def issue_session_cookie() -> str:
    """Build the signed session cookie value: ``<expiry>.<hmac-hex>``."""
    expiry = int(time.time()) + SESSION_TTL_SECONDS
    payload = f"memo-admin-session|{expiry}"
    return f"{expiry}.{_sign(settings.admin_password, payload)}"


def validate_session_cookie(value: str | None) -> bool:
    """Verify a session cookie against the admin password and its expiry."""
    if not admin_auth_enabled():
        return True
    if not value:
        return False
    parts = value.split(".")
    if len(parts) != 2:
        return False
    expiry_raw, signature = parts
    try:
        expiry = int(expiry_raw)
    except ValueError:
        return False
    if expiry < int(time.time()):
        return False
    expected = _sign(settings.admin_password, f"memo-admin-session|{expiry}")
    return _constant_time_eq(signature, expected)


def _active_chat_session_key() -> tuple[str, str]:
    """Return the active ``(key_id, secret)`` used to sign new chat cookies.

    A configured ``MEMO_CHAT_SESSION_SECRET`` always wins. Local development
    without one uses a fixed, non-production key so the public chat still works
    out of the box. That fallback deliberately ignores ``admin_password``: the
    admin credential must never become a public-cookie signing key.
    """
    secret = settings.chat_session_secret.strip()
    if secret:
        return settings.chat_session_key_id.strip(), secret
    if settings.environment.strip().lower() == "development":
        return _DEV_CHAT_SESSION_KEY_ID, _DEV_CHAT_SESSION_SECRET
    raise RuntimeError(
        "MEMO_CHAT_SESSION_SECRET must be set when MEMO_ENVIRONMENT is not "
        "'development'"
    )


def _chat_session_verification_secret(key_id: str) -> str | None:
    """Return the secret configured for ``key_id``, or ``None`` if unknown."""
    active_key_id, active_secret = _active_chat_session_key()
    if key_id == active_key_id:
        return active_secret
    for previous_key_id, previous_secret in settings.chat_session_previous_keys.items():
        if previous_key_id.strip() == key_id:
            secret = previous_secret.strip()
            return secret or None
    return None


def _chat_session_signature(key_id: str, secret: str, raw_session_id: str) -> str:
    payload = f"memo-chat-session|{key_id}|{raw_session_id}"
    return _sign(secret, payload)


def issue_chat_session_cookie(session_id: uuid.UUID) -> str:
    """Build a chat-session cookie: ``<key-id>.<uuid>.<hmac-hex>``."""
    key_id, secret = _active_chat_session_key()
    raw = str(session_id)
    signature = _chat_session_signature(key_id, secret, raw)
    return f"{key_id}.{raw}.{signature}"


def read_chat_session_id(value: str | None) -> uuid.UUID | None:
    """Return the signed chat-session id, or ``None`` when it is not valid.

    Cookies signed by an unknown or retired key id are rejected, as are
    malformed values and signatures that do not match the configured key.
    """
    if not value:
        return None
    parts = value.split(".")
    if len(parts) != 3:
        return None
    key_id, raw, signature = parts
    if not key_id:
        return None
    try:
        session_id = uuid.UUID(raw)
    except ValueError:
        return None
    secret = _chat_session_verification_secret(key_id)
    if secret is None:
        return None
    expected = _chat_session_signature(key_id, secret, raw)
    if not _constant_time_eq(signature, expected):
        return None
    return session_id


def resolve_chat_session_id(request: Request) -> uuid.UUID:
    """Reuse a valid signed chat session or mint a fresh opaque one."""
    return (
        read_chat_session_id(request.cookies.get(CHAT_SESSION_COOKIE_NAME))
        or uuid.uuid4()
    )


def set_chat_session_cookie(response: Response, session_id: uuid.UUID) -> None:
    response.set_cookie(
        CHAT_SESSION_COOKIE_NAME,
        issue_chat_session_cookie(session_id),
        httponly=True,
        samesite="lax",
        secure=cookie_secure(),
        path="/",
    )


def read_chat_conversation_id(value: str | None) -> uuid.UUID | None:
    """Return the conversation pointer cookie value, or ``None`` when invalid."""
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def set_chat_conversation_cookie(
    response: Response, conversation_id: uuid.UUID
) -> None:
    """Keep the browser pointed at the conversation it owns.

    The signed session cookie proves ownership; this plain pointer cookie only
    selects which conversation the public chat surface loads. Both the text
    fallback and the voice endpoint set it so a spoken turn continues the same
    thread.
    """
    response.set_cookie(
        CHAT_CONVERSATION_COOKIE_NAME,
        str(conversation_id),
        httponly=True,
        samesite="lax",
        secure=cookie_secure(),
        path="/",
    )


def require_admin_ui(request: Request) -> None:
    """Dependency for server-rendered admin pages: fail-open or signed cookie."""
    if not admin_auth_enabled():
        return
    if not validate_session_cookie(request.cookies.get(SESSION_COOKIE_NAME)):
        raise LoginRequired()


def require_admin_api(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Dependency for the JSON admin API: fail-open or bearer password."""
    if not admin_auth_enabled():
        return
    if authorization is None or not authorization.startswith("Bearer "):
        raise _unauthorized()
    token = authorization[len("Bearer ") :].strip()
    if not token or not _constant_time_eq(token, settings.admin_password):
        raise _unauthorized()


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )


__all__ = [
    "CHAT_CONVERSATION_COOKIE_NAME",
    "CHAT_SESSION_COOKIE_NAME",
    "SESSION_COOKIE_NAME",
    "SESSION_TTL_SECONDS",
    "LoginRequired",
    "admin_auth_enabled",
    "cookie_secure",
    "issue_chat_session_cookie",
    "issue_session_cookie",
    "read_chat_conversation_id",
    "read_chat_session_id",
    "require_admin_api",
    "require_admin_ui",
    "resolve_chat_session_id",
    "set_chat_conversation_cookie",
    "set_chat_session_cookie",
    "validate_session_cookie",
    "verify_password",
]
