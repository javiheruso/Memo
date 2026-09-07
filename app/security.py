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
"""

import hashlib
import hmac
import time
from typing import Annotated

from fastapi import Header, HTTPException, Request

from app.core.config import settings

SESSION_COOKIE_NAME = "memo_admin_session"
SESSION_TTL_SECONDS = 12 * 60 * 60
_LOGIN_URL = "/admin/login"


class LoginRequired(Exception):
    """Raised by the UI dependency when a valid session cookie is missing.

    Converted into a redirect to ``/admin/login`` by an app-level exception
    handler so browser requests never see a bare 401/403.
    """


def admin_auth_enabled() -> bool:
    """True when a shared admin password is configured (auth is on)."""
    return bool(settings.admin_password)


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
    "SESSION_COOKIE_NAME",
    "SESSION_TTL_SECONDS",
    "LoginRequired",
    "admin_auth_enabled",
    "issue_session_cookie",
    "require_admin_api",
    "require_admin_ui",
    "validate_session_cookie",
    "verify_password",
]
