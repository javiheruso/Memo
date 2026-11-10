"""Relationship status changes: promotion without duplicates or data loss.

The admin update path is an in-place status change. These tests pin the
boundary: the status enum is validated, unknown ids return 404, and provenance
(source), validity dates, type and endpoints survive the update. A promotion is
never an insert, so repeating it must not create a second relationship row.
"""

import uuid
from datetime import date

from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import RelationshipStatus, RelationshipType, SourceType
from app.models import Relationship
from app.repositories.person import PersonRepository
from app.repositories.relationship import RelationshipRepository
from app.repositories.source import SourceRepository
from app.schemas.person import PersonCreate
from app.schemas.relationship import RelationshipCreate
from app.schemas.source import SourceCreate

API = "/api/v1/admin"


async def _relationship(app_client: AsyncClient, *, status: str = "probable") -> dict:
    carmen = (
        await app_client.post(f"{API}/persons", json={"first_name": "Carmen"})
    ).json()
    antonio = (
        await app_client.post(f"{API}/persons", json={"first_name": "Antonio"})
    ).json()
    source = (
        await app_client.post(f"{API}/sources", json={"source_type": "family_member"})
    ).json()
    response = await app_client.post(
        f"{API}/relationships",
        json={
            "person_a_id": carmen["id"],
            "person_b_id": antonio["id"],
            "relationship_type": "spouse_of",
            "status": status,
            "source_id": source["id"],
            "valid_from": "1960-05-01",
            "valid_to": "2001-02-03",
        },
    )
    assert response.status_code == 201
    return response.json()


async def _relationship_count(db_session: AsyncSession) -> int:
    return (
        await db_session.execute(select(func.count()).select_from(Relationship))
    ).scalar_one()


async def test_patch_promotes_probable_relationship_in_place(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    created = await _relationship(app_client, status="probable")

    response = await app_client.patch(
        f"{API}/relationships/{created['id']}",
        json={"status": "verified"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["id"] == created["id"]
    assert body["status"] == "verified"
    # Provenance and temporal context are preserved exactly.
    assert body["source_id"] == created["source_id"]
    assert body["valid_from"] == "1960-05-01"
    assert body["valid_to"] == "2001-02-03"
    assert body["relationship_type"] == created["relationship_type"]
    assert body["person_a_id"] == created["person_a_id"]
    assert body["person_b_id"] == created["person_b_id"]
    assert body["created_at"] == created["created_at"]

    stored = list((await db_session.execute(select(Relationship))).scalars())
    assert len(stored) == 1
    assert stored[0].status is RelationshipStatus.verified
    assert stored[0].source_id == uuid.UUID(created["source_id"])


async def test_repeated_promotion_creates_no_duplicate_rows(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    created = await _relationship(app_client, status="probable")
    assert await _relationship_count(db_session) == 1

    for _ in range(3):
        response = await app_client.patch(
            f"{API}/relationships/{created['id']}",
            json={"status": "verified"},
        )
        assert response.status_code == 200

    listed = await app_client.get(f"{API}/relationships")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert await _relationship_count(db_session) == 1


async def test_patch_unknown_relationship_returns_404(
    app_client: AsyncClient,
) -> None:
    response = await app_client.patch(
        f"{API}/relationships/{uuid.uuid4()}", json={"status": "verified"}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Relationship not found"


async def test_patch_rejects_unknown_status_enum(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    created = await _relationship(app_client)

    response = await app_client.patch(
        f"{API}/relationships/{created['id']}", json={"status": "confirmed"}
    )
    assert response.status_code == 422

    stored = (
        await db_session.execute(
            select(Relationship).where(Relationship.id == uuid.UUID(created["id"]))
        )
    ).scalar_one()
    assert stored.status is RelationshipStatus.probable


async def test_patch_rejects_provenance_or_date_fields(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    created = await _relationship(app_client)

    response = await app_client.patch(
        f"{API}/relationships/{created['id']}",
        json={"status": "verified", "source_id": None, "valid_from": "1900-01-01"},
    )
    assert response.status_code == 422

    stored = (
        await db_session.execute(
            select(Relationship).where(Relationship.id == uuid.UUID(created["id"]))
        )
    ).scalar_one()
    assert stored.status is RelationshipStatus.probable
    assert stored.valid_from == date(1960, 5, 1)
    assert stored.source_id is not None


async def test_repository_set_status_preserves_row_identity(
    db_session: AsyncSession,
) -> None:
    persons = PersonRepository(db_session)
    carmen = await persons.create(PersonCreate(first_name="Carmen"))
    antonio = await persons.create(PersonCreate(first_name="Antonio"))
    source = await SourceRepository(db_session).create(
        SourceCreate(source_type=SourceType.system)
    )
    repo = RelationshipRepository(db_session)
    created = await repo.create(
        RelationshipCreate(
            person_a_id=carmen.id,
            person_b_id=antonio.id,
            relationship_type=RelationshipType.friend_of,
            status=RelationshipStatus.probable,
            source_id=source.id,
            valid_from=date(1990, 1, 1),
        )
    )
    count_before = await _relationship_count(db_session)

    updated = await repo.set_status(created.id, RelationshipStatus.verified)

    assert updated is not None
    assert updated.id == created.id
    assert updated.status is RelationshipStatus.verified
    assert updated.source_id == source.id
    assert updated.valid_from == date(1990, 1, 1)
    assert updated.person_a_id == carmen.id
    assert updated.person_b_id == antonio.id
    assert await _relationship_count(db_session) == count_before

    assert await repo.set_status(uuid.uuid4(), RelationshipStatus.verified) is None
