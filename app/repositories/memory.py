import uuid

from sqlalchemy import bindparam, select, update
from sqlalchemy.orm import selectinload

from app.models import Memory, MemoryPerson
from app.repositories.base import SQLRepository
from app.schemas.memory import MemoryCreate


class UnsupportedVectorSearch(ValueError):
    """Raised when pgvector I/O (similarity search or embedding writes) is
    requested on a non-PostgreSQL dialect. Semantic retrieval falls back to
    in-memory scoring on SQLite, and embedding writes are refused so no vector
    is ever persisted on a dialect that cannot store it."""


class MemoryRepository(SQLRepository[Memory]):
    model = Memory
    load_options = (selectinload(Memory.people),)

    @property
    def supports_vector_search(self) -> bool:
        """True when the bound dialect can run pgvector ``<=>`` queries."""
        return self._session.get_bind().dialect.name == "postgresql"

    async def similarity_search(
        self, query_vector: list[float], *, limit: int = 5
    ) -> list[tuple[Memory, float]]:
        """Nearest-neighbor memories on PostgreSQL via pgvector.

        Returns ``(memory, similarity)`` pairs ordered by descending
        similarity. ``similarity`` is derived from the L2 distance so higher
        is better, matching the cosine scoring used by the SQLite fallback.
        Memories without a stored embedding are never matched.
        """
        if not self.supports_vector_search:
            raise UnsupportedVectorSearch(
                "pgvector similarity search requires a PostgreSQL dialect"
            )
        distance_expr = Memory.embedding.l2_distance(query_vector)
        stmt = (
            select(Memory, distance_expr.label("distance"))
            .where(Memory.embedding.is_not(None))
            .options(*self.load_options)
            .order_by(distance_expr)
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            (memory, 1.0 / (1.0 + float(distance))) for memory, distance in rows
        ]

    async def create(
        self, data: MemoryCreate, *, embedding: list[float] | None = None
    ) -> Memory:
        if embedding is not None and not self.supports_vector_search:
            raise UnsupportedVectorSearch(
                "pgvector embedding writes require a PostgreSQL dialect"
            )
        return await self._create(
            lambda: Memory(
                title=data.title,
                narrative=data.narrative,
                approx_start_date=data.approx_start_date,
                approx_end_date=data.approx_end_date,
                approx_period_text=data.approx_period_text,
                place_text=data.place_text,
                memory_status=data.memory_status,
                sensitivity_level=data.sensitivity_level,
                source_id=data.source_id,
                embedding=embedding,
                people=[
                    MemoryPerson(person_id=person.person_id, role=person.role)
                    for person in data.people
                ],
            )
        )

    async def list_without_embeddings(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[Memory]:
        """Rows whose ``embedding`` is ``NULL``, for backfill and monitoring."""
        stmt = (
            select(Memory)
            .where(Memory.embedding.is_(None))
            .options(*self.load_options)
            .order_by(Memory.created_at)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def update_embeddings(self, vectors: dict[uuid.UUID, list[float]]) -> int:
        """Persist embeddings for the given memory ids in one batched statement.

        Returns the number of rows updated. Refuses to run on a dialect that
        cannot store pgvector values (the SQLite test path).

        The statement targets the core ``Memory.__table__`` rather than the ORM
        entity: an ORM ``update(Memory)`` with a WHERE on the primary key is
        interpreted by SQLAlchemy as a per-row "bulk update by primary key",
        which requires the primary-key value under the column key and rejects a
        named ``bindparam``. A core update sidesteps that machinery entirely.
        """
        if not vectors:
            return 0
        if not self.supports_vector_search:
            raise UnsupportedVectorSearch(
                "pgvector embedding writes require a PostgreSQL dialect"
            )
        table = Memory.__table__
        stmt = (
            update(table)
            .where(table.c.id == bindparam("memory_id"))
            .values(embedding=bindparam("embedding"))
        )
        result = await self._session.execute(
            stmt,
            [
                {"memory_id": memory_id, "embedding": vector}
                for memory_id, vector in vectors.items()
            ],
        )
        rowcount = result.rowcount or 0
        await self._session.commit()
        return rowcount

    async def for_person(self, person_id: uuid.UUID) -> list[Memory]:
        stmt = (
            select(Memory)
            .join(MemoryPerson, MemoryPerson.memory_id == Memory.id)
            .where(MemoryPerson.person_id == person_id)
            .options(selectinload(Memory.people))
            .order_by(Memory.created_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())
