"""Backfill service for memories whose embedding is ``NULL``.

Narrative memories are the vectorizable content (dossier doc 03): relationships,
facts, events and trust state stay structured and are never embedded. The
service only runs against a PostgreSQL dialect because that is the only path
where pgvector can persist the vectors; on any other dialect it raises
``UnsupportedVectorSearch`` so a caller can never silently leave half-written
data or write a vector into a database that cannot store it.
"""

from app.embeddings.base import Embedder
from app.embeddings.memory_text import memory_search_text
from app.repositories.memory import MemoryRepository, UnsupportedVectorSearch


class MemoryEmbeddingBackfillService:
    """Embed and persist embeddings for memories that currently have none."""

    def __init__(
        self, *, embedder: Embedder, memory_repo: MemoryRepository
    ) -> None:
        self._embedder = embedder
        self._memory_repo = memory_repo

    async def backfill(self, *, limit: int = 100) -> int:
        """Embed up to ``limit`` memories with a ``NULL`` embedding.

        Returns the number of memories embedded. Embeddings are computed from
        the same memory text used by the retrieval fallback, so a stored vector
        and an on-the-fly vector for the same memory always agree.
        """
        if not self._memory_repo.supports_vector_search:
            raise UnsupportedVectorSearch(
                "embedding backfill requires a PostgreSQL dialect for pgvector writes"
            )
        pending = await self._memory_repo.list_without_embeddings(limit=limit)
        if not pending:
            return 0
        vectors = {
            memory.id: await self._embedder.embed(
                memory_search_text(title=memory.title, narrative=memory.narrative)
            )
            for memory in pending
        }
        await self._memory_repo.update_embeddings(vectors)
        return len(pending)
