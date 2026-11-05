"""Semantic memory retrieval seam.

Ranks memories by similarity to the user's phrasing. The storage path is
chosen by the repository: pgvector ``<=>`` on PostgreSQL, or a pure-Python
cosine fallback on other dialects (the default SQLite test path). The
deterministic embedder keeps the whole slice runnable without any real
embedding provider.

Trust-model rules stay in code: only memories at or below ``max_level``
sensitivity are returned, so a sensitive memory is never surfaced by a
similarity hit, and every returned item keeps its database status so the
context builder can tag it as a memory (never a fact).
"""

from dataclasses import dataclass

from app.domain.enums import SensitivityLevel
from app.embeddings.base import Embedder, cosine_similarity
from app.embeddings.memory_text import memory_search_text
from app.models import Memory
from app.policies.trust import (
    filter_sensitive_memories,
    sensitivity_levels_at_or_below,
)
from app.repositories.memory import MemoryRepository

_FALLBACK_CORPUS_LIMIT = 1000


@dataclass(frozen=True)
class SemanticHit:
    memory: Memory
    score: float


@dataclass(frozen=True)
class SemanticSearchResult:
    hits: list[SemanticHit]
    filtered_count: int


class SemanticRetriever:
    """Rank memories by similarity, applying the sensitivity policy in code."""

    def __init__(
        self, *, embedder: Embedder, memory_repo: MemoryRepository
    ) -> None:
        self._embedder = embedder
        self._memory_repo = memory_repo

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        max_level: SensitivityLevel = SensitivityLevel.low,
    ) -> SemanticSearchResult:
        query_vector = await self._embedder.embed(query)
        if self._memory_repo.supports_vector_search:
            # The SQL path filters sensitivity in the query itself, before the
            # neighbour limit, so protected rows never consume safe slots.
            scored = await self._memory_repo.similarity_search(
                query_vector,
                limit=limit,
                sensitivity_levels=sensitivity_levels_at_or_below(max_level),
            )
            safe, filtered = filter_sensitive_memories(
                [memory for memory, _ in scored], max_level
            )
            safe_ids = {memory.id for memory in safe}
            hits = [
                SemanticHit(memory=memory, score=score)
                for memory, score in scored
                if memory.id in safe_ids
            ][:limit]
            return SemanticSearchResult(
                hits=hits, filtered_count=len(filtered)
            )

        scored, filtered = await self._fallback(
            query_vector, limit=limit, max_level=max_level
        )
        return SemanticSearchResult(
            hits=[
                SemanticHit(memory=memory, score=score)
                for memory, score in scored
            ],
            filtered_count=len(filtered),
        )

    async def _fallback(
        self,
        query_vector: list[float],
        *,
        limit: int,
        max_level: SensitivityLevel,
    ) -> tuple[list[tuple[Memory, float]], list[Memory]]:
        """Score memories in Python when pgvector is not available (SQLite).

        Memories with a stored embedding use it; memories created through the
        repository (``embedding`` is ``NULL``) are embedded on the fly from
        their title and narrative, so the default test path never needs a
        vector-capable database.

        The sensitivity filter runs over the full ranked list *before*
        ``[:limit]``, so high-sensitivity memories cannot crowd out allowed
        ones. The filtered rows are still returned for policy reporting.
        """
        candidates = await self._memory_repo.list(limit=_FALLBACK_CORPUS_LIMIT)
        scored: list[tuple[Memory, float]] = []
        for memory in candidates:
            vector = memory.embedding
            if vector is None:
                vector = await self._embedder.embed(
                    memory_search_text(
                        title=memory.title, narrative=memory.narrative
                    )
                )
            score = cosine_similarity(query_vector, vector)
            if score > 0.0:
                scored.append((memory, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        safe, filtered = filter_sensitive_memories(
            [memory for memory, _ in scored], max_level
        )
        safe_ids = {memory.id for memory in safe}
        return (
            [
                (memory, score)
                for memory, score in scored
                if memory.id in safe_ids
            ][:limit],
            filtered,
        )
