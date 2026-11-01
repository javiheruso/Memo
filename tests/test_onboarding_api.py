import uuid

from httpx import AsyncClient

API = "/api/v1/admin"
ONBOARDING = f"{API}/onboarding"


async def test_onboarding_draft_is_saved_and_resumed(app_client: AsyncClient) -> None:
    created = await app_client.post(
        ONBOARDING,
        json={
            "identity": {
                "first_name": "Elena",
                "preferred_name": "Elenita",
                "language": "es-ES",
            },
            "reviewer": {"name": "Javi", "relationship": "hijo"},
        },
    )
    assert created.status_code == 201
    body = created.json()
    session_id = body["id"]
    assert body["status"] == "draft"
    assert body["payload"]["identity"]["first_name"] == "Elena"
    assert body["consent_confirmed"] is False

    resumed = await app_client.get(f"{ONBOARDING}/{session_id}")
    assert resumed.status_code == 200
    assert resumed.json()["payload"]["identity"]["preferred_name"] == "Elenita"

    saved = await app_client.put(
        f"{ONBOARDING}/{session_id}/draft",
        json={
            "identity": resumed.json()["payload"]["identity"],
            "reviewer": resumed.json()["payload"]["reviewer"],
            "contributors": [
                {
                    "name": "Javi",
                    "relationship": "hijo",
                    "permission_note": "Administer the profile",
                }
            ],
            "sources": [
                {
                    "key": "javi",
                    "source_type": "family_member",
                    "description": "Javi, son",
                }
            ],
            "sensitive_topics": [
                {
                    "topic": "fallecimiento de Miguel",
                    "handling": "Do not initiate; redirect gently.",
                    "triggers": ["murió"],
                }
            ],
        },
    )
    assert saved.status_code == 200
    payload = saved.json()["payload"]
    assert [item["name"] for item in payload["contributors"]] == ["Javi"]
    assert payload["sensitive_topics"][0]["triggers"] == ["murió"]

    listed = await app_client.get(ONBOARDING)
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [session_id]


async def test_onboarding_draft_rejects_duplicate_local_keys(
    app_client: AsyncClient,
) -> None:
    created = await app_client.post(ONBOARDING)
    session_id = created.json()["id"]

    response = await app_client.put(
        f"{ONBOARDING}/{session_id}/draft",
        json={
            "sources": [
                {"key": "family", "source_type": "family_member"},
                {"key": "family", "source_type": "document"},
            ]
        },
    )
    assert response.status_code == 422


async def test_onboarding_unknown_session_is_404(app_client: AsyncClient) -> None:
    response = await app_client.get(f"{ONBOARDING}/{uuid.uuid4()}")
    assert response.status_code == 404


def _first_day_package() -> dict:
    return {
        "identity": {"first_name": "Elena", "preferred_name": "Elenita"},
        "reviewer": {"name": "Javi", "relationship": "hijo"},
        "contributors": [{"name": "Javi", "relationship": "hijo"}],
        "sources": [
            {"key": "family", "source_type": "family_member", "description": "Javi"}
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


async def _draft_session(client: AsyncClient, payload: dict) -> str:
    created = await client.post(ONBOARDING, json=payload)
    assert created.status_code == 201
    session_id = created.json()["id"]
    saved = await client.put(f"{ONBOARDING}/{session_id}/draft", json=payload)
    assert saved.status_code == 200
    return session_id


async def test_activation_requires_explicit_consent(app_client: AsyncClient) -> None:
    session_id = await _draft_session(app_client, _first_day_package())

    blocked = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert blocked.status_code == 409
    assert "consent" in blocked.json()["detail"].lower()
    assert (await app_client.get(f"{ONBOARDING}/{session_id}")).json()[
        "status"
    ] == "draft"

    consent = await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={
            "confirmed_by": "Javi",
            "scope": "Activate Elena's conversational profile",
        },
    )
    assert consent.status_code == 200
    assert consent.json()["consent_confirmed"] is True
    assert consent.json()["status"] == "draft"

    activated = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert activated.status_code == 200
    body = activated.json()
    assert body["status"] == "activated"
    assert body["activated_at"] is not None
    assert body["subject_person_id"] is not None
    assert body["profile_id"] is not None


async def test_activation_rejects_an_incomplete_package(
    app_client: AsyncClient,
) -> None:
    created = await app_client.post(ONBOARDING)
    session_id = created.json()["id"]
    await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={"confirmed_by": "Javi", "scope": "Activate profile"},
    )

    response = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert response.status_code == 422
    problems = response.json()["detail"]
    assert "identity.first_name is required" in problems
    assert "reviewer.name is required" in problems
    assert "at least one permitted contributor is required" in problems


