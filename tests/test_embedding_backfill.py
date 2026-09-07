"""Non-PostgreSQL-safe behavior for embedding persistence and backfill.

These tests assert that no vector is ever written on a dialect that cannot
store pgvector values (the default in-memory SQLite test path): the repository
refuses embedding writes, the backfill service refuses to run, and the admin
create/backfill paths leave embeddings untouched or reject the request. The
positive PostgreSQL paths are covered by the smoke tests in
``tests/test_postgres_profile.py``.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.domain.enums import MemoryStatus, SensitivityLevel
from app.embeddings.deterministic import DeterministicEmbedder
from app.models import Memory
from app.repositories.memory import MemoryRepository, UnsupportedVectorSearch
from app.schemas.memory import MemoryCreate, MemoryPersonCreate
from app.services.memory_embeddings import MemoryEmbeddingBackfillService
from tests.fixtures.universe import SeedUniverse

API = "/api/v1/admin"

_SKIP_ON_POSTGRES = pytest.mark.skipif(
    settings.test_database_url is not None,
    reason="relies on SQLite-only fail-safe behavior; covered by postgres smoke tests",
)


@_SKIP_ON_POSTGRES
async def test_create_with_embedding_rejected_on_sqlite(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    repo = MemoryRepository(db_session)
    with pytest.raises(UnsupportedVectorSearch):
        await repo.create(
            MemoryCreate(
                title="Fishing trips",
                narrative="Paco and Antonio used to go fishing.",
                memory_status=MemoryStatus.consolidated,
                sensitivity_level=SensitivityLevel.low,
                source_id=seed.source_ids["shared_by_carmen"],
                people=[MemoryPersonCreate(person_id=seed.person_ids["paco_navarro"])],
            ),
            embedding=[0.0] * 1536,
        )


@_SKIP_ON_POSTGRES
async def test_update_embeddings_rejected_on_sqlite(db_session: AsyncSession) -> None:
    with pytest.raises(UnsupportedVectorSearch):
        await MemoryRepository(db_session).update_embeddings(
            {uuid.uuid4(): [0.0] * 1536}
        )


@_SKIP_ON_POSTGRES
async def test_backfill_service_rejected_on_sqlite(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = MemoryEmbeddingBackfillService(
        embedder=DeterministicEmbedder(), memory_repo=MemoryRepository(db_session)
    )
    with pytest.raises(UnsupportedVectorSearch):
        await service.backfill()


@_SKIP_ON_POSTGRES
async def test_admin_create_memory_leaves_embedding_null_on_sqlite(
    app_client: AsyncClient, db_session: AsyncSession, seed: SeedUniverse
) -> None:
    response = await app_client.post(
        f"{API}/memories",
        json={
            "title": "Cooking with María",
            "narrative": "María and Carmen baked a cake for Javi's birthday.",
            "source_id": str(seed.source_ids["shared_by_carmen"]),
            "people": [{"person_id": str(seed.person_ids["maria"]), "role": "guest"}],
        },
    )
    assert response.status_code == 201
    stored = (
        await db_session.execute(
            select(Memory).where(Memory.id == uuid.UUID(response.json()["id"]))
        )
    ).scalar_one()
    assert stored.embedding is None


@_SKIP_ON_POSTGRES
async def test_admin_backfill_rejected_on_sqlite(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    response = await app_client.post(f"{API}/memories/backfill")
    assert response.status_code == 400
    assert "PostgreSQL" in response.json()["detail"]


@_SKIP_ON_POSTGRES
async def test_list_without_embeddings_returns_all_seed_memories_on_sqlite(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    pending = await MemoryRepository(db_session).list_without_embeddings(limit=100)
    assert len(pending) == 7
    assert all(memory.embedding is None for memory in pending)
