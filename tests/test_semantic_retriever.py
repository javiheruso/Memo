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
async def test_semantic_search_rejects_unrelated_shared_token_recall(
    retriever: SemanticRetriever, db_session: AsyncSession, seed: SeedUniverse
):
    """A single incidental shared token must not surface an unrelated memory.

    Reproduces "¿Te acuerdas de cuando fuimos juntos a Valencia?" against the
    Spanish dancing memory: only "juntos" overlaps, so the calibrated floor
    must keep it out. The same memory must still be recalled by a question
    about its actual content.
    """
    repo = MemoryRepository(db_session)
    dancing = await repo.create(
        MemoryCreate(
            title="Bailes de los sábados",
            narrative=(
                "Carmen y Antonio bailaban juntos en el salón del pueblo "
                "los sábados."
            ),
            memory_status=MemoryStatus.recurring,
            sensitivity_level=SensitivityLevel.low,
            source_id=seed.source_ids["shared_by_carmen"],
        )
    )

    unrelated = await retriever.search(
        "¿Te acuerdas de cuando fuimos juntos a Valencia?"
    )
    assert unrelated.hits == []

    recall = await retriever.search(
        "¿Te acuerdas de cuando bailábamos en el salón del pueblo?"
    )
    assert [hit.memory.id for hit in recall.hits] == [dancing.id]


@_SKIP_ON_POSTGRES
async def test_semantic_search_requires_calibrated_relevance_floor(
    retriever: SemanticRetriever, db_session: AsyncSession, seed: SeedUniverse
):
    """Weak topical overlap is not enough; a clear topical match is."""
    repo = MemoryRepository(db_session)
    dancing = await repo.create(
        MemoryCreate(
            title="Bailes de los sábados",
            narrative=(
                "Carmen y Antonio bailaban juntos en el salón del pueblo "
                "los sábados."
            ),
            memory_status=MemoryStatus.recurring,
            sensitivity_level=SensitivityLevel.low,
            source_id=seed.source_ids["shared_by_carmen"],
        )
    )

    weak = await retriever.search("¿Recuerdas aquella boda en el pueblo?")
    assert weak.hits == []

    strong = await retriever.search(
        "¿Recuerdas cuando bailaban en el salón del pueblo?"
    )
    assert [hit.memory.id for hit in strong.hits] == [dancing.id]
    assert strong.hits[0].score >= 0.20


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
async def test_semantic_search_filters_sensitivity_before_the_limit(
    retriever: SemanticRetriever, db_session: AsyncSession, seed: SeedUniverse
):
    """Protected memories must not consume the requested result slots.

    The high-sensitivity rows outrank the allowed ones on purpose. Filtering
    after truncation would return zero hits for ``limit=2``; filtering first
    must return both allowed memories.
    """
    repo = MemoryRepository(db_session)
    query = "lighthouse keeper secret diaries"
    for index in range(2):
        await repo.create(
            MemoryCreate(
                title=f"Protected lighthouse keeper secret diaries {index}",
                narrative="lighthouse keeper secret diaries",
                memory_status=MemoryStatus.consolidated,
                sensitivity_level=SensitivityLevel.high,
                source_id=seed.source_ids["family_records"],
            )
        )
    allowed_ids = set()
    for index in range(2):
        memory = await repo.create(
            MemoryCreate(
                title=f"Lighthouse keeper {index}",
                narrative="A quiet keeper memory.",
                memory_status=MemoryStatus.consolidated,
                sensitivity_level=SensitivityLevel.low,
                source_id=seed.source_ids["shared_by_carmen"],
            )
        )
        allowed_ids.add(memory.id)

    result = await retriever.search(query, limit=2)

    assert {hit.memory.id for hit in result.hits} == allowed_ids
    assert all(
        hit.memory.sensitivity_level == SensitivityLevel.low
        for hit in result.hits
    )
    assert result.filtered_count == 2


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
