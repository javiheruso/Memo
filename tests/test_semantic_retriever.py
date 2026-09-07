import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.domain.enums import MemoryStatus, SensitivityLevel
from app.embeddings.deterministic import DeterministicEmbedder
from app.repositories.memory import MemoryRepository, UnsupportedVectorSearch
from app.retrieval.semantic_retriever import SemanticRetriever
from app.schemas.memory import MemoryCreate, MemoryPersonCreate
from tests.fixtures.universe import SeedUniverse

# The retriever behavior tests rely on the in-memory cosine fallback, which is
# the default SQLite path. Under the PostgreSQL profile the seed universe has
# no stored embeddings (no backfill), so those cases are covered by the
# dedicated smoke tests in test_postgres_profile.py instead.
_SKIP_ON_POSTGRES = pytest.mark.skipif(
    settings.test_database_url is not None,
    reason="relies on the SQLite semantic fallback; covered by postgres smoke tests",
)


@pytest.fixture
def retriever(db_session: AsyncSession) -> SemanticRetriever:
    return SemanticRetriever(
        embedder=DeterministicEmbedder(),
        memory_repo=MemoryRepository(db_session),
    )


@_SKIP_ON_POSTGRES
async def test_sqlite_repo_reports_no_vector_search(db_session: AsyncSession):
    assert MemoryRepository(db_session).supports_vector_search is False


@_SKIP_ON_POSTGRES
async def test_sqlite_repo_raises_for_pgvector_search(db_session: AsyncSession):
    repo = MemoryRepository(db_session)
    with pytest.raises(UnsupportedVectorSearch):
        await repo.similarity_search([0.0] * 1536)


@_SKIP_ON_POSTGRES
async def test_semantic_search_ranks_relevant_memory_first(
    retriever: SemanticRetriever, seed: SeedUniverse
):
    result = await retriever.search("Do you remember when we went dancing?")
    assert result.hits
    assert result.hits[0].memory.title == "Dancing evenings"
    assert result.hits[0].memory.memory_status == MemoryStatus.recurring


@_SKIP_ON_POSTGRES
async def test_semantic_search_without_match_returns_empty(
    retriever: SemanticRetriever, seed: SeedUniverse
):
    result = await retriever.search("¿Te acuerdas de cuando íbamos a bailar?")
    assert result.hits == []
    assert result.filtered_count == 0


@_SKIP_ON_POSTGRES
async def test_semantic_search_filters_sensitive_memories(
    retriever: SemanticRetriever, seed: SeedUniverse
):
    result = await retriever.search("Do you remember Antonio passing away?")
    assert result.filtered_count == 1
    assert all(
        hit.memory.sensitivity_level == SensitivityLevel.low for hit in result.hits
    )
    assert not any(hit.memory.title == "Antonio's passing" for hit in result.hits)


@_SKIP_ON_POSTGRES
async def test_semantic_fallback_embeds_memories_without_stored_embedding(
    retriever: SemanticRetriever, db_session: AsyncSession, seed: SeedUniverse
):
    repo = MemoryRepository(db_session)
    memory = await repo.create(
        MemoryCreate(
            title="Cooking with María",
            narrative="María and Carmen baked a cake together for Javi's birthday.",
            memory_status=MemoryStatus.consolidated,
            sensitivity_level=SensitivityLevel.low,
            source_id=seed.source_ids["shared_by_carmen"],
            people=[MemoryPersonCreate(person_id=seed.person_ids["maria"])],
        )
    )
    assert memory.embedding is None
    result = await retriever.search(
        "Do you remember when we baked a cake with María?"
    )
    assert any(hit.memory.id == memory.id for hit in result.hits)
