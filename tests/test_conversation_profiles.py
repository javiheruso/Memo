import uuid

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ConversationProfile, ConversationProfileSensitiveTopic

API = "/api/v1/admin"
PROFILES = f"{API}/profiles"


async def _create_person(client: AsyncClient, first_name: str) -> str:
    response = await client.post(f"{API}/persons", json={"first_name": first_name})
    assert response.status_code == 201
    return response.json()["id"]


async def test_admin_profile_roundtrip_persists_sensitive_topics(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    person_id = await _create_person(app_client, "Elena")

    response = await app_client.post(
        PROFILES,
        json={
            "person_id": person_id,
            "preferred_name": "Elena",
            "language": "es-ES",
            "positive_topics": ["costura", "familia"],
            "sensitive_topics": [
                {
                    "topic": "fallecimiento de Miguel",
                    "handling": "Do not initiate; redirect gently if asked.",
                    "consent_review_required": True,
                    "triggers": ["murió", "fallecimiento"],
                }
            ],
            "data_origin": "Fictional for tests",
            "is_fictional": True,
            "is_active": True,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["person_id"] == person_id
    assert body["is_active"] is True
    assert body["positive_topics"] == ["costura", "familia"]
    assert body["sensitive_topics"][0]["topic"] == "fallecimiento de Miguel"
    assert body["sensitive_topics"][0]["triggers"] == ["murió", "fallecimiento"]

    listed = await app_client.get(PROFILES)
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [body["id"]]

    active = await app_client.get(f"{PROFILES}/active")
    assert active.status_code == 200
    assert active.json()["id"] == body["id"]

    profile = await db_session.get(ConversationProfile, uuid.UUID(body["id"]))
    assert profile is not None
    topics = (
        await db_session.execute(
            select(ConversationProfileSensitiveTopic).where(
                ConversationProfileSensitiveTopic.profile_id == profile.id
            )
        )
    ).scalars().all()
    assert [topic.topic for topic in topics] == ["fallecimiento de Miguel"]


async def test_activating_a_profile_deactivates_the_previous_one(
    app_client: AsyncClient,
) -> None:
    first_person = await _create_person(app_client, "Elena")
    second_person = await _create_person(app_client, "Carmen")

    first = (
        await app_client.post(
            PROFILES,
            json={"person_id": first_person, "is_active": True},
        )
    ).json()
    second = (
        await app_client.post(
            PROFILES,
            json={"person_id": second_person, "is_active": True},
        )
    ).json()

    active = (await app_client.get(f"{PROFILES}/active")).json()
    assert active["id"] == second["id"]
    assert active["is_active"] is True

    activated = await app_client.post(f"{PROFILES}/{first['id']}/activate")
    assert activated.status_code == 200
    assert activated.json()["id"] == first["id"]
    assert (await app_client.get(f"{PROFILES}/active")).json()["id"] == first["id"]


async def test_profile_requires_known_person_and_one_profile_per_person(
    app_client: AsyncClient,
) -> None:
    missing = await app_client.post(PROFILES, json={"person_id": str(uuid.uuid4())})
    assert missing.status_code == 404

    person_id = await _create_person(app_client, "Elena")
    created = await app_client.post(PROFILES, json={"person_id": person_id})
    assert created.status_code == 201

    duplicate = await app_client.post(PROFILES, json={"person_id": person_id})
    assert duplicate.status_code == 409
