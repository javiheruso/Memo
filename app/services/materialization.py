"""Explicit admin-only materialization of reviewed evidence into facts/memories.

Materialization is the only promotion-like write in the system, and it is
always explicit, human-initiated and audited. For every fact/memory created
from a claim group or a knowledge gap the service:

- refuses ``ignored`` sources and duplicate attempts,
- writes one append-only ``materializations`` provenance row,
- transitions the source out of the review queue (claim group or gap to
  ``resolved``) so it is never re-flagged or re-materialized by accident,
- never mutates the source evidence: member claims keep their ``candidate``
  status and verbatim text, gap occurrences are untouched, and contradiction
  rows are left alone.

The whole unit (target row, provenance row, source transition) commits once, so
a failure can never leave a fact/memory without provenance. Memory
materialization reuses the existing embedding behavior (``memory_search_text``
+ the PostgreSQL-only pgvector write) and on any other dialect keeps the
embedding ``NULL``, exactly like the manual memory creation path.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import (
    ClaimGroupStatus,
    KnowledgeGapStatus,
    MaterializationSourceType,
    MaterializationTargetType,
)
from app.embeddings.base import Embedder
from app.embeddings.memory_text import memory_search_text
from app.models import (
    ClaimGroup,
    Fact,
    KnowledgeGap,
    Materialization,
    Memory,
    MemoryPerson,
    Source,
)
from app.repositories.fact import FactRepository
from app.repositories.memory import MemoryRepository
from app.schemas.materialization import FactMaterializeRequest, MemoryMaterializeRequest


class MaterializationError(Exception):
    """Base error for the materialization service; mapped to HTTP by the API."""


class UnknownSourceError(MaterializationError):
    """The claim group or knowledge gap being materialized does not exist."""


class IgnoredSourceError(MaterializationError):
    """An ``ignored`` source is explicitly out of the queue and cannot be
    materialized."""


class DuplicateMaterializationError(MaterializationError):
    """The source was already materialized into the same target kind."""


class UnknownSourceRefError(MaterializationError):
    """The evidence ``source_id`` supplied for the target row does not exist."""


class MaterializationService:
    def __init__(
        self, session: AsyncSession, embedder: Embedder | None = None
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._fact_repo = FactRepository(session)
        self._memory_repo = MemoryRepository(session)

    # ------------------------------------------------------------------
    # Public materialization entrypoints
    # ------------------------------------------------------------------

    async def materialize_claim_group_fact(
        self, group_id: uuid.UUID, data: FactMaterializeRequest
    ) -> Fact:
        group = await self._session.get(ClaimGroup, group_id)
        if group is None:
            raise UnknownSourceError("Claim group not found")
        self._guard_not_ignored(
            group.status,
            message=f"cannot materialize claim group {group_id}: it is ignored",
        )
        await self._guard_no_duplicate(
            MaterializationSourceType.claim_group,
            group.id,
            MaterializationTargetType.fact,
        )
        fact = await self._build_fact(data, default_text=group.canonical_text)
        self._session.add(fact)
        await self._session.flush()
        self._add_provenance(
            MaterializationSourceType.claim_group,
            group.id,
            MaterializationTargetType.fact,
            fact.id,
        )
        self._resolve_claim_group(group, materialized_label=f"fact {fact.id}")
        return await self._commit_and_refetch(fact)

    async def materialize_claim_group_memory(
        self, group_id: uuid.UUID, data: MemoryMaterializeRequest
    ) -> Memory:
        group = await self._session.get(ClaimGroup, group_id)
        if group is None:
            raise UnknownSourceError("Claim group not found")
        self._guard_not_ignored(
            group.status,
            message=f"cannot materialize claim group {group_id}: it is ignored",
        )
        await self._guard_no_duplicate(
            MaterializationSourceType.claim_group,
            group.id,
            MaterializationTargetType.memory,
        )
        memory = await self._build_memory(data, default_text=group.canonical_text)
        self._session.add(memory)
        await self._session.flush()
        self._add_provenance(
            MaterializationSourceType.claim_group,
            group.id,
            MaterializationTargetType.memory,
            memory.id,
        )
        self._resolve_claim_group(group, materialized_label=f"memory {memory.id}")
        return await self._commit_and_refetch(memory)

    async def materialize_gap_fact(
        self, gap_id: uuid.UUID, data: FactMaterializeRequest
    ) -> Fact:
        gap = await self._session.get(KnowledgeGap, gap_id)
        if gap is None:
            raise UnknownSourceError("Knowledge gap not found")
        self._guard_not_ignored(
            gap.status,
            message=f"cannot materialize knowledge gap {gap_id}: it is ignored",
        )
        await self._guard_no_duplicate(
            MaterializationSourceType.knowledge_gap,
            gap.id,
            MaterializationTargetType.fact,
        )
        fact = await self._build_fact(data, default_text=gap.canonical_reference)
        self._session.add(fact)
        await self._session.flush()
        self._add_provenance(
            MaterializationSourceType.knowledge_gap,
            gap.id,
            MaterializationTargetType.fact,
            fact.id,
        )
        self._resolve_gap(gap, person_id=data.subject_person_id, fact_id=fact.id)
        return await self._commit_and_refetch(fact)

    async def materialize_gap_memory(
        self, gap_id: uuid.UUID, data: MemoryMaterializeRequest
    ) -> Memory:
        gap = await self._session.get(KnowledgeGap, gap_id)
        if gap is None:
            raise UnknownSourceError("Knowledge gap not found")
        self._guard_not_ignored(
            gap.status,
            message=f"cannot materialize knowledge gap {gap_id}: it is ignored",
        )
        await self._guard_no_duplicate(
            MaterializationSourceType.knowledge_gap,
            gap.id,
            MaterializationTargetType.memory,
        )
        memory = await self._build_memory(
            data, default_text=gap.canonical_reference
        )
        self._session.add(memory)
        await self._session.flush()
        self._add_provenance(
            MaterializationSourceType.knowledge_gap,
            gap.id,
            MaterializationTargetType.memory,
            memory.id,
        )
        person_id = data.people[0].person_id if data.people else None
        self._resolve_gap(gap, person_id=person_id)
        return await self._commit_and_refetch(memory)

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _guard_not_ignored(self, status: object, *, message: str) -> None:
        if str(getattr(status, "value", status)) == "ignored":
            raise IgnoredSourceError(message)

    async def _guard_no_duplicate(
        self,
        source_type: MaterializationSourceType,
        source_id: uuid.UUID,
        target_type: MaterializationTargetType,
    ) -> None:
        stmt = select(Materialization).where(
            Materialization.source_type == source_type,
            Materialization.source_id == source_id,
            Materialization.target_type == target_type,
        )
        existing = (await self._session.execute(stmt)).scalar_one_or_none()
        if existing is not None:
            raise DuplicateMaterializationError(
                f"{source_type.value} {source_id} was already materialized as "
                f"{target_type.value} {existing.target_id}"
            )

    async def _guard_source_ref_exists(self, source_id: uuid.UUID) -> None:
        if await self._session.get(Source, source_id) is None:
            raise UnknownSourceRefError(f"source {source_id} not found")

    # ------------------------------------------------------------------
    # Target row builders
    # ------------------------------------------------------------------

    async def _build_fact(
        self, data: FactMaterializeRequest, *, default_text: str
    ) -> Fact:
        await self._guard_source_ref_exists(data.source_id)
        return Fact(
            subject_person_id=data.subject_person_id,
            fact_type=data.fact_type,
            value_text=data.value_text or default_text,
            value_date=data.value_date,
            value_number=data.value_number,
            related_person_id=data.related_person_id,
            place_text=data.place_text,
            valid_from=data.valid_from,
            valid_to=data.valid_to,
            status=data.status,
            source_id=data.source_id,
            notes=data.notes,
        )

    async def _build_memory(
        self, data: MemoryMaterializeRequest, *, default_text: str
    ) -> Memory:
        await self._guard_source_ref_exists(data.source_id)
        title = data.title or default_text
        narrative = data.narrative or default_text
        embedding = None
        if self._memory_repo.supports_vector_search and self._embedder is not None:
            embedding = await self._embedder.embed(
                memory_search_text(title=title, narrative=narrative)
            )
        return Memory(
            title=title,
            narrative=narrative,
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

    # ------------------------------------------------------------------
    # Provenance and source-state transitions
    # ------------------------------------------------------------------

    def _add_provenance(
        self,
        source_type: MaterializationSourceType,
        source_id: uuid.UUID,
        target_type: MaterializationTargetType,
        target_id: uuid.UUID,
    ) -> None:
        self._session.add(
            Materialization(
                source_type=source_type,
                source_id=source_id,
                target_type=target_type,
                target_id=target_id,
            )
        )

    def _resolve_claim_group(self, group: ClaimGroup, *, materialized_label: str) -> None:
        group.status = ClaimGroupStatus.resolved
        group.needs_review = False
        group.resolved_at = datetime.now(UTC)
        note = f"Materialized as {materialized_label}"
        group.resolution_note = (
            f"{group.resolution_note} | {note}"
            if group.resolution_note
            else note
        )

    def _resolve_gap(
        self,
        gap: KnowledgeGap,
        *,
        person_id: uuid.UUID | None = None,
        fact_id: uuid.UUID | None = None,
    ) -> None:
        gap.status = KnowledgeGapStatus.resolved
        gap.resolved_person_id = person_id
        gap.resolved_fact_id = fact_id

    async def _commit_and_refetch(self, target: Fact | Memory) -> Fact | Memory:
        """Commit the atomic unit and reload the target row.

        The reload mirrors ``SQLRepository._create``: a fresh SELECT (with the
        repository's eager loads) populates server-defaulted columns such as
        ``created_at`` and relationship collections so the row is safe to
        serialize outside an async context.
        """
        await self._session.commit()
        if isinstance(target, Memory):
            fresh = await self._memory_repo.get(target.id)
        else:
            fresh = await self._fact_repo.get(target.id)
        if fresh is None:
            raise RuntimeError(f"{type(target).__name__} disappeared after commit")
        return fresh
