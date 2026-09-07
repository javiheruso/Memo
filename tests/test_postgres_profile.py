"""PostgreSQL-backed run smoke checks.

These tests only execute when ``MEMO_TEST_DATABASE_URL`` points at a running
PostgreSQL server; otherwise the whole module is skipped and the default suite
keeps running on in-memory SQLite. When the profile is enabled, the shared
conftest runs the ENTIRE suite against that server and truncates tables between
tests.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.domain.enums import MemoryStatus, SensitivityLevel
from app.embeddings.deterministic import DeterministicEmbedder
from app.models import Materialization, Memory
from app.repositories.memory import MemoryRepository
from app.retrieval.semantic_retriever import SemanticRetriever
from app.schemas.memory import MemoryCreate
from app.services.memory_embeddings import MemoryEmbeddingBackfillService
from tests.fixtures.universe import SeedUniverse

pytestmark = pytest.mark.postgres

_NEEDS_POSTGRES = pytest.mark.skipif(
    settings.test_database_url is None,
    reason="MEMO_TEST_DATABASE_URL is not set; running SQLite-backed suite",
)


@_NEEDS_POSTGRES
def test_profile_targets_postgresql() -> None:
    assert settings.test_database_url is not None
    assert "postgresql" in settings.test_database_url


@_NEEDS_POSTGRES
async def test_known_person_flow_runs_on_postgres(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    response = await app_client.post(
        "/api/v1/chat", json={"message": "¿Quién era Francisco?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "person_query"
    assert "amigo de Antonio" in body["answer"]
    assert body["sources"]


@_NEEDS_POSTGRES
async def test_event_query_runs_on_postgres(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    response = await app_client.post("/api/v1/chat", json={"message": "¿Viene María hoy?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "event_query"
    assert "confirmado" in body["answer"]
    assert any(
        source["entity_type"] == "event" for source in body["sources"]
    )


@_NEEDS_POSTGRES
async def test_memory_query_runs_on_postgres(app_client: AsyncClient) -> None:
    """The memory_query pipeline degrades honestly when no embeddings exist.

    The seed universe stores no embeddings (no backfill by design), so the
    pgvector path matches nothing and the responder admits the gap without
    inventing a memory.
    """
    response = await app_client.post(
        "/api/v1/chat", json={"message": "Do you remember when we went dancing?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "memory_query"
    assert body["answer"] == "No tengo guardado ese recuerdo todavía."
    assert body["sources"] == []


@_NEEDS_POSTGRES
async def test_semantic_retrieval_uses_pgvector_on_postgres(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    """Prove the pgvector ``<=>`` path returns stored-embedding memories."""
    embedder = DeterministicEmbedder()
    query = "Do you remember fishing with Paco?"
    memory = Memory(
        title="Fishing trips",
        narrative="Carmen remembers that Paco and Antonio used to go fishing together.",
        memory_status=MemoryStatus.consolidated,
        sensitivity_level=SensitivityLevel.low,
        source_id=seed.source_ids["shared_by_carmen"],
        embedding=await embedder.embed(query),
    )
    db_session.add(memory)
    await db_session.commit()

    repo = MemoryRepository(db_session)
    assert repo.supports_vector_search is True
    retriever = SemanticRetriever(embedder=embedder, memory_repo=repo)
    result = await retriever.search(query, limit=5)
    assert result.hits
    assert result.hits[0].memory.id == memory.id
    assert result.hits[0].score == pytest.approx(1.0)


@_NEEDS_POSTGRES
async def test_repo_persists_lists_and_batches_embeddings_on_postgres(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    """Prove create/list/update embedding persistence against pgvector."""
    embedder = DeterministicEmbedder()
    repo = MemoryRepository(db_session)

    embedded = await repo.create(
        MemoryCreate(
            title="Cooking with María",
            narrative="María and Carmen baked a cake for Javi's birthday.",
            memory_status=MemoryStatus.consolidated,
            sensitivity_level=SensitivityLevel.low,
            source_id=seed.source_ids["shared_by_carmen"],
        ),
        embedding=await embedder.embed("Cooking with María baked a cake"),
    )
    fetched = await repo.get(embedded.id)
    assert fetched is not None
    assert fetched.embedding is not None
    assert len(fetched.embedding) == 1536

    pending = await repo.list_without_embeddings(limit=100)
    assert all(memory.id != embedded.id for memory in pending)
    assert len(pending) == 7

    plain = await repo.create(
        MemoryCreate(
            title="Summer at the beach",
            narrative="The family spent a summer at the beach with little Javi.",
            memory_status=MemoryStatus.candidate,
            sensitivity_level=SensitivityLevel.low,
            source_id=seed.source_ids["old_photos"],
        )
    )
    updated = await repo.update_embeddings(
        {plain.id: await embedder.embed("Summer at the beach with Javi")}
    )
    assert updated == 1
    await db_session.refresh(plain)
    assert plain.embedding is not None


@_NEEDS_POSTGRES
async def test_backfill_service_runs_and_is_idempotent_on_postgres(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    embedder = DeterministicEmbedder()
    repo = MemoryRepository(db_session)
    service = MemoryEmbeddingBackfillService(
        embedder=embedder, memory_repo=repo
    )

    processed = await service.backfill(limit=100)
    assert processed == 7
    assert await repo.list_without_embeddings(limit=100) == []

    assert await service.backfill(limit=100) == 0


@_NEEDS_POSTGRES
async def test_admin_create_memory_embeds_on_postgres(
    app_client: AsyncClient, db_session: AsyncSession, seed: SeedUniverse
) -> None:
    response = await app_client.post(
        "/api/v1/admin/memories",
        json={
            "title": "Cooking with María",
            "narrative": "María and Carmen baked a cake for Javi's birthday.",
            "source_id": str(seed.source_ids["shared_by_carmen"]),
        },
    )
    assert response.status_code == 201
    stored = (
        await db_session.execute(
            select(Memory).where(Memory.id == uuid.UUID(response.json()["id"]))
        )
    ).scalar_one()
    assert stored.embedding is not None


@_NEEDS_POSTGRES
async def test_admin_backfill_endpoint_embeds_pending_memories_on_postgres(
    app_client: AsyncClient, db_session: AsyncSession, seed: SeedUniverse
) -> None:
    created = await app_client.post(
        "/api/v1/admin/memories",
        json={
            "title": "Bread from the bakery",
            "narrative": "Carmen remembers buying bread from Paco el panadero.",
            "source_id": str(seed.source_ids["recorded_conversation"]),
        },
    )
    assert created.status_code == 201

    response = await app_client.post("/api/v1/admin/memories/backfill")
    assert response.status_code == 200
    assert response.json()["processed"] == 7

    pending = await MemoryRepository(db_session).list_without_embeddings(limit=100)
    assert pending == []


@_NEEDS_POSTGRES
async def test_materialize_claim_group_memory_embeds_on_postgres(
    app_client: AsyncClient, db_session: AsyncSession, seed: SeedUniverse
) -> None:
    """Memory materialization reuses the embedding path on PostgreSQL."""
    response = await app_client.post(
        "/api/v1/chat", json={"message": "Íbamos a bailar los sábados."}
    )
    assert response.status_code == 200
    group = (await app_client.get("/api/v1/admin/claim-groups")).json()[0]

    materialized = await app_client.post(
        f"/api/v1/admin/claim-groups/{group['id']}/materialize-memory",
        json={"source_id": str(seed.source_ids["shared_by_carmen"])},
    )
    assert materialized.status_code == 201
    stored = (
        await db_session.execute(
            select(Memory).where(Memory.id == uuid.UUID(materialized.json()["id"]))
        )
    ).scalar_one()
    assert stored.embedding is not None
    assert len(stored.embedding) == 1536
    materialized_id = uuid.UUID(materialized.json()["id"])
    pending = await MemoryRepository(db_session).list_without_embeddings(limit=100)
    # The seed universe stores memories without embeddings; the materialized
    # memory must be embedded, so it must not appear among the pending rows.
    assert all(memory.id != materialized_id for memory in pending)


@_NEEDS_POSTGRES
async def test_materialize_claim_group_fact_runs_on_postgres(
    app_client: AsyncClient, db_session: AsyncSession, seed: SeedUniverse
) -> None:
    """Fact materialization creates a row, provenance and resolved group."""
    response = await app_client.post(
        "/api/v1/chat", json={"message": "Vivíamos en Madrid."}
    )
    assert response.status_code == 200
    group = (await app_client.get("/api/v1/admin/claim-groups")).json()[0]

    materialized = await app_client.post(
        f"/api/v1/admin/claim-groups/{group['id']}/materialize-fact",
        json={
            "subject_person_id": str(seed.person_ids["carmen"]),
            "fact_type": "residence",
            "source_id": str(seed.source_ids["shared_by_carmen"]),
        },
    )
    assert materialized.status_code == 201
    assert materialized.json()["value_text"] == "viviamos en madrid"

    provenance = (
        await db_session.execute(select(Materialization))
    ).scalars().all()
    assert len(provenance) == 1
    assert provenance[0].target_id == uuid.UUID(materialized.json()["id"])
