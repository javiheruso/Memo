"""Media metadata vertical slice: repository, admin API and query flows.

Covers metadata-only media rows (no upload, no content interpretation), the
photo hint tier of the deterministic person-query responder, and the facts
tier that precedes it. Existing relation/memory flows are asserted here too so
the hint ladder order stays locked end to end.
"""

import uuid
from datetime import date

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import FactStatus, MediaType
from app.repositories.media import MediaRepository
from app.repositories.person import PersonRepository
from app.schemas.media import MediaCreate, MediaPersonCreate
from app.schemas.person import PersonCreate
from tests.fixtures.universe import SeedUniverse

API = "/api/v1/admin"


async def test_media_repository_roundtrip_and_for_person(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    repo = MediaRepository(db_session)
    media = await repo.create(
        MediaCreate(
            media_type=MediaType.photo,
            storage_path="albums/maria_1970.jpg",
            title="María de niña",
            description="Copia aportada por la familia.",
            approx_date=date(1970, 4, 12),
            place_text="Guardamar",
            source_id=seed.source_ids["old_photos"],
            people=[
                MediaPersonCreate(
                    person_id=seed.person_ids["maria"],
                    is_primary=True,
                    notes="retrato",
                )
            ],
        )
    )

    fetched = await repo.get(media.id)
    assert fetched is not None
    assert fetched.media_type == "photo"
    assert fetched.title == "María de niña"
    assert fetched.approx_date == date(1970, 4, 12)
    assert fetched.place_text == "Guardamar"
    assert fetched.description == "Copia aportada por la familia."

    listed = await repo.list()
    assert any(item.id == media.id for item in listed)

    person_media = await repo.for_person(seed.person_ids["maria"])
    assert [item.id for item in person_media] == [media.id]
    assert person_media[0].people[0].is_primary is True


async def test_media_repository_for_person_orders_primary_first(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    """When a person has several photos, the primary link comes first."""
    repo = MediaRepository(db_session)
    secondary = await repo.create(
        MediaCreate(
            media_type=MediaType.photo,
            storage_path="albums/secondary.jpg",
            approx_date=date(1960, 1, 1),
            people=[MediaPersonCreate(person_id=seed.person_ids["javi"])],
        )
    )
    primary = await repo.create(
        MediaCreate(
            media_type=MediaType.photo,
            storage_path="albums/primary.jpg",
            approx_date=date(1955, 1, 1),
            people=[
                MediaPersonCreate(
                    person_id=seed.person_ids["javi"], is_primary=True
                )
            ],
        )
    )
    person_media = await repo.for_person(seed.person_ids["javi"])
    assert [item.id for item in person_media] == [primary.id, secondary.id]


async def test_admin_media_create_list_metadata_only(
    app_client: AsyncClient,
) -> None:
    person = (
        await app_client.post(f"{API}/persons", json={"first_name": "Rosa"})
    ).json()

    response = await app_client.post(
        f"{API}/media",
        json={
            "media_type": "photo",
            "storage_path": "albums/rosa.jpg",
            "title": "Foto de Rosa",
            "approx_date": "1965-07-01",
            "place_text": "Santa Pola",
            "people": [{"person_id": person["id"], "is_primary": True}],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["media_type"] == "photo"
    assert body["title"] == "Foto de Rosa"
    assert body["approx_date"] == "1965-07-01"
    assert body["place_text"] == "Santa Pola"
    assert body["source_id"] is None
    assert body["people"] == [
        {"person_id": person["id"], "is_primary": True, "notes": None}
    ]

    listed = await app_client.get(f"{API}/media")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [body["id"]]


async def test_admin_media_rejects_non_photo_media_type_value(
    app_client: AsyncClient,
) -> None:
    response = await app_client.post(
        f"{API}/media",
        json={
            "media_type": "hologram",
            "storage_path": "albums/x.jpg",
        },
    )
    assert response.status_code == 422


async def test_admin_media_requires_storage_path(
    app_client: AsyncClient,
) -> None:
    response = await app_client.post(
        f"{API}/media",
        json={"media_type": "photo"},
    )
    assert response.status_code == 422


async def test_admin_person_primary_media_json_roundtrip_and_clear(
    app_client: AsyncClient,
) -> None:
    person = (
        await app_client.post(f"{API}/persons", json={"first_name": "Rosa"})
    ).json()
    media = (
        await app_client.post(
            f"{API}/media",
            json={"media_type": "photo", "storage_path": "albums/rosa.jpg"},
        )
    ).json()

    updated = await app_client.patch(
        f"{API}/persons/{person['id']}/primary-media",
        json={"media_id": media["id"]},
    )
    assert updated.status_code == 200
    assert updated.json()["primary_media_id"] == media["id"]

    fetched = await app_client.get(f"{API}/persons/{person['id']}")
    assert fetched.json()["primary_media_id"] == media["id"]

    cleared = await app_client.patch(
        f"{API}/persons/{person['id']}/primary-media",
        json={"media_id": None},
    )
    assert cleared.status_code == 200
    assert cleared.json()["primary_media_id"] is None


async def test_admin_person_primary_media_unknown_ids_404(
    app_client: AsyncClient,
) -> None:
    person = (
        await app_client.post(f"{API}/persons", json={"first_name": "Rosa"})
    ).json()

    unknown_media = await app_client.patch(
        f"{API}/persons/{person['id']}/primary-media",
        json={"media_id": str(uuid.uuid4())},
    )
    assert unknown_media.status_code == 404

    unknown_person = await app_client.patch(
        f"{API}/persons/{uuid.uuid4()}/primary-media",
        json={"media_id": None},
    )
    assert unknown_person.status_code == 404


async def test_chat_primary_media_reaches_photo_tier_without_link_row(
    app_client: AsyncClient,
) -> None:
    """A person linked to a photo only via ``primary_media_id`` still hits it.

    The photo row has no ``media_people`` link; retrieval must fall back to
    ``persons.primary_media_id`` so the deterministic photo tier fires.
    """
    person = (
        await app_client.post(f"{API}/persons", json={"first_name": "Genoveva"})
    ).json()
    media = (
        await app_client.post(
            f"{API}/media",
            json={
                "media_type": "photo",
                "storage_path": "albums/genoveva.jpg",
                "approx_date": "1958-06-01",
                "place_text": "Elche",
            },
        )
    ).json()
    set_response = await app_client.patch(
        f"{API}/persons/{person['id']}/primary-media",
        json={"media_id": media["id"]},
    )
    assert set_response.status_code == 200

    response = await app_client.post(
        "/api/v1/chat", json={"message": "¿Quién era Genoveva?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "person_query"
    assert body["answer"] == (
        "No tengo claro exactamente quién era Genoveva, "
        "pero tengo guardada una fotografía de 1958, en Elche."
    )
    assert any(
        s["entity_type"] == "media" and s["entity_id"] == media["id"]
        for s in body["sources"]
    )
    assert any(
        s["entity_type"] == "person"
        and s["entity_id"] == person["id"]
        and s["usage_type"] == "subject"
        for s in body["sources"]
    )


async def test_chat_primary_media_cleared_falls_through_to_no_details(
    app_client: AsyncClient,
) -> None:
    person = (
        await app_client.post(f"{API}/persons", json={"first_name": "Genoveva"})
    ).json()
    media = (
        await app_client.post(
            f"{API}/media",
            json={"media_type": "photo", "storage_path": "albums/genoveva.jpg"},
        )
    ).json()
    await app_client.patch(
        f"{API}/persons/{person['id']}/primary-media",
        json={"media_id": media["id"]},
    )
    cleared = await app_client.patch(
        f"{API}/persons/{person['id']}/primary-media",
        json={"media_id": None},
    )
    assert cleared.status_code == 200

    response = await app_client.post(
        "/api/v1/chat", json={"message": "¿Quién era Genoveva?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == (
        "Tengo guardado a Genoveva, pero todavía no tengo detalles."
    )
    assert {s["entity_type"] for s in body["sources"]} == {"person"}



async def test_chat_photo_only_person_reaches_photo_tier(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    response = await app_client.post(
        "/api/v1/chat", json={"message": "¿Quién era Luisa?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "person_query"
    assert body["ambiguity"] is None
    assert body["gap"] is None
    assert body["answer"] == (
        "No tengo claro exactamente quién era Luisa, "
        "pero tengo guardada una fotografía de 1965, en Santa Pola."
    )
    media_sources = [
        s for s in body["sources"] if s["entity_type"] == "media"
    ]
    assert media_sources == [
        {
            "entity_type": "media",
            "entity_id": str(seed.media_ids["luisa_photo"]),
            "usage_type": "media",
        }
    ]
    assert any(
        s["entity_type"] == "person"
        and s["entity_id"] == str(seed.person_ids["luisa"])
        and s["usage_type"] == "subject"
        for s in body["sources"]
    )


async def test_chat_relation_still_outranks_photo_tier(
    app_client: AsyncClient,
    db_session: AsyncSession,
    seed: SeedUniverse,
) -> None:
    """Attaching a photo to a known relation person does not change the answer."""
    repo = MediaRepository(db_session)
    await repo.create(
        MediaCreate(
            media_type=MediaType.photo,
            storage_path="albums/antonio.jpg",
            approx_date=date(1980, 1, 1),
            place_text="Guardamar",
            people=[MediaPersonCreate(person_id=seed.person_ids["antonio"])],
        )
    )

    response = await app_client.post(
        "/api/v1/chat", json={"message": "¿Quién era Antonio?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "pareja de Carmen" in body["answer"]
    assert "fotografía" not in body["answer"]
    assert {s["entity_type"] for s in body["sources"]} == {"person", "relationship"}


async def test_chat_facts_tier_fires_before_memories_for_relationless_person(
    app_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    person = await PersonRepository(db_session).create(
        PersonCreate(
            first_name="Rosa",
            last_name="Pérez",
            display_name="Rosa",
            description="No relation recorded yet.",
        )
    )
    source = (
        await app_client.post(
            f"{API}/sources", json={"source_type": "family_member"}
        )
    ).json()
    fact = (
        await app_client.post(
            f"{API}/facts",
            json={
                "subject_person_id": str(person.id),
                "fact_type": "residence",
                "value_text": "Vivió en Zaragoza",
                "status": FactStatus.verified.value,
                "source_id": source["id"],
            },
        )
    ).json()

    response = await app_client.post(
        "/api/v1/chat", json={"message": "¿Quién era Rosa?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "person_query"
    assert "Dato: 'Vivió en Zaragoza'" in body["answer"]
    assert any(
        s["entity_type"] == "fact" and s["entity_id"] == fact["id"]
        for s in body["sources"]
    )
