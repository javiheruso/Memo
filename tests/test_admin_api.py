import uuid

from httpx import AsyncClient

from tests.fixtures.universe import SeedUniverse

API = "/api/v1/admin"


async def test_admin_person_create_list_search_get(app_client: AsyncClient) -> None:
    response = await app_client.post(
        f"{API}/persons",
        json={
            "first_name": "Carmen",
            "last_name": "Gómez",
            "aliases": [{"alias": "Carmencita", "alias_type": "nickname"}],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["display_name"] == "Carmen Gómez"
    assert [a["alias"] for a in body["aliases"]] == ["Carmencita"]

    listed = await app_client.get(f"{API}/persons")
    assert listed.status_code == 200
    assert [p["id"] for p in listed.json()] == [body["id"]]

    search = await app_client.get(f"{API}/persons/search", params={"name": "carmen"})
    assert search.status_code == 200
    assert [p["id"] for p in search.json()] == [body["id"]]

    fetched = await app_client.get(f"{API}/persons/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["first_name"] == "Carmen"


async def test_admin_person_search_returns_both_pacos(app_client: AsyncClient, seed: SeedUniverse) -> None:
    response = await app_client.get(f"{API}/persons/search", params={"name": "paco"})
    assert response.status_code == 200
    ids = {p["id"] for p in response.json()}
    assert ids == {
        str(seed.person_ids["paco_navarro"]),
        str(seed.person_ids["paco_garcia"]),
    }


async def test_admin_person_not_found(app_client: AsyncClient) -> None:
    response = await app_client.get(f"{API}/persons/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_admin_source_roundtrip(app_client: AsyncClient) -> None:
    response = await app_client.post(
        f"{API}/sources", json={"source_type": "photo", "description": "album"}
    )
    assert response.status_code == 201
    source_id = response.json()["id"]

    listed = await app_client.get(f"{API}/sources")
    assert listed.status_code == 200
    assert [s["id"] for s in listed.json()] == [source_id]


async def test_admin_relationship_roundtrip(app_client: AsyncClient) -> None:
    carmen = (await app_client.post(f"{API}/persons", json={"first_name": "Carmen"})).json()
    antonio = (await app_client.post(f"{API}/persons", json={"first_name": "Antonio"})).json()
    source = (await app_client.post(f"{API}/sources", json={"source_type": "system"})).json()

    response = await app_client.post(
        f"{API}/relationships",
        json={
            "person_a_id": carmen["id"],
            "person_b_id": antonio["id"],
            "relationship_type": "spouse_of",
            "status": "verified",
            "source_id": source["id"],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["relationship_type"] == "spouse_of"
    assert body["status"] == "verified"

    listed = await app_client.get(f"{API}/relationships")
    assert listed.status_code == 200
    assert [r["id"] for r in listed.json()] == [body["id"]]


async def test_admin_relationship_rejects_self_reference(app_client: AsyncClient) -> None:
    person = (await app_client.post(f"{API}/persons", json={"first_name": "Carmen"})).json()
    response = await app_client.post(
        f"{API}/relationships",
        json={
            "person_a_id": person["id"],
            "person_b_id": person["id"],
            "relationship_type": "other",
        },
    )
    assert response.status_code == 400


async def test_admin_fact_roundtrip_defaults_to_pending(app_client: AsyncClient) -> None:
    person = (await app_client.post(f"{API}/persons", json={"first_name": "Antonio"})).json()
    source = (await app_client.post(f"{API}/sources", json={"source_type": "system"})).json()

    response = await app_client.post(
        f"{API}/facts",
        json={
            "subject_person_id": person["id"],
            "fact_type": "hobby",
            "value_text": "Fishing",
            "source_id": source["id"],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert body["value_text"] == "Fishing"

    listed = await app_client.get(f"{API}/facts")
    assert listed.status_code == 200
    assert [f["id"] for f in listed.json()] == [body["id"]]


async def test_admin_memory_roundtrip_with_people(app_client: AsyncClient) -> None:
    person = (await app_client.post(f"{API}/persons", json={"first_name": "Paco"})).json()
    source = (await app_client.post(f"{API}/sources", json={"source_type": "conversation"})).json()

    response = await app_client.post(
        f"{API}/memories",
        json={
            "narrative": "We went fishing together.",
            "source_id": source["id"],
            "people": [{"person_id": person["id"], "role": "friend"}],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["memory_status"] == "candidate"
    assert [m["person_id"] for m in body["people"]] == [person["id"]]

    listed = await app_client.get(f"{API}/memories")
    assert listed.status_code == 200
    assert [m["id"] for m in listed.json()] == [body["id"]]


async def test_admin_event_roundtrip_with_people(app_client: AsyncClient) -> None:
    person = (await app_client.post(f"{API}/persons", json={"first_name": "Carmen"})).json()
    response = await app_client.post(
        f"{API}/events",
        json={
            "title": "Cita médica",
            "description": "Revisión con el doctor.",
            "start_at": "2026-09-01T10:00:00Z",
            "status": "planned",
            "people": [
                {
                    "person_id": person["id"],
                    "role": "patient",
                }
            ],
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "planned"
    assert body["title"] == "Cita médica"
    assert [(p["person_id"], p["role"]) for p in body["people"]] == [
        (person["id"], "patient")
    ]

    listed = await app_client.get(f"{API}/events")
    assert listed.status_code == 200
    assert [e["id"] for e in listed.json()] == [body["id"]]

    fetched = await app_client.get(f"{API}/events/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["description"] == "Revisión con el doctor."


async def test_admin_event_not_found(app_client: AsyncClient) -> None:
    response = await app_client.get(f"{API}/events/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_admin_seed_loads_example_universe_on_empty_db(
    app_client: AsyncClient,
) -> None:
    response = await app_client.post(f"{API}/seed")
    assert response.status_code == 200
    assert response.json() == {
        "persons": 7,
        "relationships": 5,
        "facts": 5,
        "memories": 7,
        "media": 1,
    }

    listed = await app_client.get(f"{API}/persons")
    assert len(listed.json()) == 7


async def test_admin_seed_refuses_with_409_when_already_seeded(
    app_client: AsyncClient,
) -> None:
    first = await app_client.post(f"{API}/seed")
    assert first.status_code == 200

    second = await app_client.post(f"{API}/seed")
    assert second.status_code == 409
    assert second.json()["detail"] == "already seeded"

    listed = await app_client.get(f"{API}/persons")
    assert len(listed.json()) == 7
