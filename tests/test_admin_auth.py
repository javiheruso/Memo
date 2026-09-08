"""Admin authentication: fail-open default, signed-cookie browser flow, bearer API flow.

The default suite runs with ``MEMO_ADMIN_PASSWORD`` unset, so every test here
either pins auth on (via ``monkeypatch`` on the shared ``settings`` singleton)
or asserts the fail-open behavior explicitly.
"""

import pytest
from httpx import AsyncClient

from app.core.config import settings
from app.security import SESSION_COOKIE_NAME, cookie_secure

API = "/api/v1/admin"
UI = "/admin"
PASSWORD = "test-admin-password"


@pytest.fixture
def protected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "admin_password", PASSWORD)
    yield


async def _login(client: AsyncClient, password: str, *, follow: bool = False):
    return await client.post(
        f"{UI}/login", data={"password": password}, follow_redirects=follow
    )


# --------------------------------------------------------------------------
# Fail-open when the admin password is unset (default local/dev ergonomics)
# --------------------------------------------------------------------------


async def test_fail_open_admin_ui_without_password(app_client: AsyncClient) -> None:
    response = await app_client.get(UI)
    assert response.status_code == 200
    assert "Memo Admin" in response.text


async def test_fail_open_admin_api_without_password(app_client: AsyncClient) -> None:
    response = await app_client.get(f"{API}/persons")
    assert response.status_code == 200


async def test_login_page_redirects_to_admin_when_auth_disabled(
    app_client: AsyncClient,
) -> None:
    page = await app_client.get(f"{UI}/login")
    assert page.status_code == 303
    assert page.headers["location"] == "/admin"

    posted = await _login(app_client, "anything")
    assert posted.status_code == 303
    assert posted.headers["location"] == "/admin"


# --------------------------------------------------------------------------
# Protected behavior: UI redirects to login, API returns 401
# --------------------------------------------------------------------------


async def test_protected_ui_redirects_to_login(app_client: AsyncClient, protected) -> None:
    response = await app_client.get(UI)
    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"


async def test_protected_ui_subpage_redirects_to_login(
    app_client: AsyncClient, protected
) -> None:
    response = await app_client.get(f"{UI}/persons")
    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"


async def test_protected_api_rejects_missing_token(
    app_client: AsyncClient, protected
) -> None:
    response = await app_client.get(f"{API}/persons")
    assert response.status_code == 401


async def test_protected_api_rejects_wrong_token(
    app_client: AsyncClient, protected
) -> None:
    response = await app_client.get(
        f"{API}/persons", headers={"Authorization": "Bearer wrong-password"}
    )
    assert response.status_code == 401


async def test_protected_api_rejects_malformed_header(
    app_client: AsyncClient, protected
) -> None:
    response = await app_client.get(
        f"{API}/persons", headers={"Authorization": "Basic dXNlcjpwYXNz"}
    )
    assert response.status_code == 401


# --------------------------------------------------------------------------
# API bearer authentication
# --------------------------------------------------------------------------


async def test_protected_api_accepts_bearer_password(
    app_client: AsyncClient, protected
) -> None:
    response = await app_client.get(
        f"{API}/persons", headers={"Authorization": f"Bearer {PASSWORD}"}
    )
    assert response.status_code == 200


async def test_protected_api_accepts_bearer_on_write(
    app_client: AsyncClient, protected
) -> None:
    response = await app_client.post(
        f"{API}/persons",
        headers={"Authorization": f"Bearer {PASSWORD}"},
        json={"first_name": "Carmen"},
    )
    assert response.status_code == 201
    assert response.json()["display_name"] == "Carmen"


# --------------------------------------------------------------------------
# Browser login / logout flow
# --------------------------------------------------------------------------


async def test_login_page_renders_when_protected(
    app_client: AsyncClient, protected
) -> None:
    response = await app_client.get(f"{UI}/login")
    assert response.status_code == 200
    assert "Memo Admin" in response.text
    assert "Iniciar sesión" in response.text


async def test_login_success_sets_cookie_and_grants_access(
    app_client: AsyncClient, protected
) -> None:
    response = await _login(app_client, PASSWORD)
    assert response.status_code == 303
    assert response.headers["location"] == "/admin"
    assert SESSION_COOKIE_NAME in response.cookies

    dashboard = await app_client.get(UI)
    assert dashboard.status_code == 200
    assert "Memo Admin" in dashboard.text


async def test_login_failure_redirects_with_error(
    app_client: AsyncClient, protected
) -> None:
    response = await _login(app_client, "wrong-password")
    assert response.status_code == 303
    assert (
        response.headers["location"]
        == "/admin/login?error=Contrase%C3%B1a%20incorrecta"
    )
    assert SESSION_COOKIE_NAME not in response.cookies

    dashboard = await app_client.get(UI)
    assert dashboard.status_code == 303
    assert dashboard.headers["location"] == "/admin/login"


async def test_logout_clears_cookie(app_client: AsyncClient, protected) -> None:
    await _login(app_client, PASSWORD)
    assert (await app_client.get(UI)).status_code == 200

    logout = await app_client.post(f"{UI}/logout")
    assert logout.status_code == 303
    assert logout.headers["location"] == "/admin/login"

    dashboard = await app_client.get(UI)
    assert dashboard.status_code == 303
    assert dashboard.headers["location"] == "/admin/login"


# --------------------------------------------------------------------------
# Cookie integrity
# --------------------------------------------------------------------------


async def test_tampered_cookie_is_rejected(app_client: AsyncClient, protected) -> None:
    app_client.cookies.set(SESSION_COOKIE_NAME, "9999999999.deadbeef")
    response = await app_client.get(UI)
    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"


async def test_garbage_cookie_is_rejected(app_client: AsyncClient, protected) -> None:
    app_client.cookies.set(SESSION_COOKIE_NAME, "not-a-session-cookie")
    response = await app_client.get(UI)
    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"


async def test_expired_cookie_is_rejected(app_client: AsyncClient, protected) -> None:
    app_client.cookies.set(SESSION_COOKIE_NAME, "1.0")
    response = await app_client.get(UI)
    assert response.status_code == 303
    assert response.headers["location"] == "/admin/login"


# --------------------------------------------------------------------------
# Auth is orthogonal to the public surface
# --------------------------------------------------------------------------


async def test_health_stays_public_when_protected(
    app_client: AsyncClient, protected
) -> None:
    response = await app_client.get("/api/v1/health")
    assert response.status_code == 200


async def test_chat_stays_public_when_protected(
    app_client: AsyncClient, protected
) -> None:
    response = await app_client.post("/api/v1/chat", json={"message": "Hola"})
    assert response.status_code == 200


# --------------------------------------------------------------------------
# Secure-cookie flag across environments
# --------------------------------------------------------------------------


def test_cookie_secure_is_false_in_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "environment", "development")
    assert cookie_secure() is False


def test_cookie_secure_is_true_outside_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    assert cookie_secure() is True


async def test_production_login_sets_secure_cookie(
    app_client: AsyncClient, protected, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "environment", "production")
    response = await _login(app_client, PASSWORD)
    assert response.status_code == 303
    assert SESSION_COOKIE_NAME in response.cookies
    # httpx represents a Secure cookie with a leading "__Secure-" prefix marker.
    assert "__Secure-" in response.headers["set-cookie"] or "Secure" in response.headers["set-cookie"]
