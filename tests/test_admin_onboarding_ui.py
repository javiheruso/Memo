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


async def test_onboarding_entry_sections_are_saved_and_removed(
    app_client: AsyncClient,
) -> None:
    session_id = await _new_session(app_client)
    await app_client.post(
        f"{UI}/onboarding/{session_id}/identity",
        data={"first_name": "Elena", "reviewer_name": "Javi"},
    )
    base = f"{UI}/onboarding/{session_id}/entries"

    for section, data in (
        (
            "contributors",
            {"name": "Javi", "relationship": "hijo", "permission_note": "Aportar datos"},
        ),
        (
            "sources",
            {"key": "javi", "source_type": "family_member", "description": "Javi, hijo"},
        ),
        (
            "sensitive_topics",
            {
                "topic": "fallecimiento de Miguel",
                "handling": "No iniciar; redirigir con suavidad.",
                "triggers": "murió, funeral",
            },
        ),
        (
            "people",
            {
                "key": "miguel",
                "first_name": "Miguel",
                "relationship_type": "spouse_of",
                "certainty": "probable",
                "source_key": "javi",
            },
        ),
        (
            "facts",
            {
                "subject_key": "subject",
                "fact_type": "hobby",
                "value_text": "costura",
                "status": "verified",
                "source_key": "javi",
            },
        ),
        (
            "memories",
            {
                "title": "Tardes de costura",
                "narrative": "Elena cosía con Miguel los domingos.",
                "sensitivity_level": "low",
                "source_key": "javi",
                "people_keys": ["subject", "miguel"],
            },
        ),
    ):
        response = await app_client.post(
            f"{base}/{section}", data=data, follow_redirects=True
        )
        assert response.status_code == 200
        assert "Elemento añadido al borrador" in response.text

    stored = (await app_client.get(f"{API}/onboarding/{session_id}")).json()["payload"]
    assert [item["key"] for item in stored["sources"]] == ["javi"]
    assert [item["topic"] for item in stored["sensitive_topics"]] == [
        "fallecimiento de Miguel"
    ]
    assert stored["sensitive_topics"][0]["triggers"] == ["murió", "funeral"]
    assert [item["key"] for item in stored["people"]] == ["miguel"]
    assert [item["fact_type"] for item in stored["facts"]] == ["hobby"]
    assert stored["memories"][0]["people_keys"] == ["subject", "miguel"]

    removed = await app_client.post(
        f"{base}/facts/0/remove", follow_redirects=True
    )
    assert removed.status_code == 200
    assert "Elemento eliminado del borrador" in removed.text
    stored = (await app_client.get(f"{API}/onboarding/{session_id}")).json()["payload"]
    assert stored["facts"] == []


def _complete_package() -> dict:
    return {
        "identity": {"first_name": "Elena", "preferred_name": "Elenita"},
        "reviewer": {"name": "Javi", "relationship": "hijo"},
        "contributors": [{"name": "Javi", "relationship": "hijo"}],
        "sources": [
            {"key": "family", "source_type": "family_member", "description": "Javi"}
        ],
        "people": [
            {
                "key": "miguel",
                "first_name": "Miguel",
                "relationship_type": "spouse_of",
                "certainty": "probable",
                "source_key": "family",
            }
        ],
        "sensitive_topics": [
            {
                "topic": "fallecimiento de Miguel",
                "handling": "Do not initiate.",
                "triggers": ["murió"],
            }
        ],
        "facts": [
            {
                "fact_type": "hobby",
                "value_text": "costura",
                "status": "verified",
                "source_key": "family",
            }
        ],
        "preferences": {"positive_topics": ["costura"]},
    }


async def test_onboarding_consent_ui_gates_activation(app_client: AsyncClient) -> None:
    created = await app_client.post(f"{API}/onboarding", json=_complete_package())
    session_id = created.json()["id"]

    page = await app_client.get(f"{UI}/onboarding/{session_id}")
    assert page.status_code == 200
    assert "Registrar consentimiento" in page.text
    assert "Registra primero el consentimiento explícito" in page.text

    missing_confirm = await app_client.post(
        f"{UI}/onboarding/{session_id}/consent",
        data={"confirmed_by": "Javi", "scope": "Activar perfil"},
        follow_redirects=True,
    )
    assert "Marca la confirmación explícita" in missing_confirm.text
    stored = (await app_client.get(f"{API}/onboarding/{session_id}")).json()
    assert stored["consent_confirmed"] is False
    assert stored["status"] == "draft"

    consent = await app_client.post(
        f"{UI}/onboarding/{session_id}/consent",
        data={
            "confirmed": "on",
            "confirmed_by": "Javi",
            "scope": "Activar perfil",
        },
        follow_redirects=True,
    )
    assert "Consentimiento explícito registrado" in consent.text
    assert "Activar perfil conversacional" in consent.text
    stored = (await app_client.get(f"{API}/onboarding/{session_id}")).json()
    assert stored["consent_confirmed"] is True
    assert stored["status"] == "draft"

    activated = await app_client.post(
        f"{UI}/onboarding/{session_id}/activate", follow_redirects=True
    )
    assert activated.status_code == 200
    assert "Perfil conversacional activado" in activated.text
    stored = (await app_client.get(f"{API}/onboarding/{session_id}")).json()
    assert stored["status"] == "activated"
    assert stored["profile_id"] is not None
