import uuid

from httpx import AsyncClient

UI = "/admin"
API = "/api/v1/admin"


async def _new_session(client: AsyncClient) -> str:
    response = await client.post(f"{UI}/onboarding", follow_redirects=True)
    assert response.status_code == 200
    return response.url.path.rsplit("/", 1)[-1]


async def test_onboarding_list_and_create_page(app_client: AsyncClient) -> None:
    page = await app_client.get(f"{UI}/onboarding")
    assert page.status_code == 200
    assert "Onboarding familiar" in page.text
    assert "Iniciar onboarding" in page.text

    created = await app_client.post(f"{UI}/onboarding", follow_redirects=True)
    assert created.status_code == 200
    assert "Identidad y revisor" in created.text
    assert "Sin identificar" in created.text


async def test_onboarding_identity_and_preferences_are_resumable(
    app_client: AsyncClient,
) -> None:
    session_id = await _new_session(app_client)

    saved = await app_client.post(
        f"{UI}/onboarding/{session_id}/identity",
        data={
            "first_name": "Elena",
            "last_name": "Ruiz",
            "preferred_name": "Elenita",
            "language": "es-ES",
            "reviewer_name": "Javi",
            "reviewer_relationship": "hijo",
        },
        follow_redirects=True,
    )
    assert saved.status_code == 200
    assert "Identidad y revisor guardados" in saved.text

    prefs = await app_client.post(
        f"{UI}/onboarding/{session_id}/preferences",
        data={
            "positive_topics": "costura, familia",
            "never_do": "No hablar de política",
            "conversational_style": "Pausado",
        },
        follow_redirects=True,
    )
    assert prefs.status_code == 200
    assert "Preferencias guardadas" in prefs.text

    stored = (await app_client.get(f"{API}/onboarding/{session_id}")).json()
    assert stored["payload"]["identity"]["preferred_name"] == "Elenita"
    assert stored["payload"]["reviewer"]["name"] == "Javi"
    assert stored["payload"]["preferences"]["positive_topics"] == [
        "costura",
        "familia",
    ]


async def test_onboarding_unknown_session_is_404(app_client: AsyncClient) -> None:
    response = await app_client.get(f"{UI}/onboarding/{uuid.uuid4()}")
    assert response.status_code == 404
