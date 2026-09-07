import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.domain.enums import KnowledgeGapStatus
from app.models import KnowledgeGap, KnowledgeGapConversation
from app.repositories.base import SQLRepository


class KnowledgeGapRepository(SQLRepository[KnowledgeGap]):
    model = KnowledgeGap
    load_options = (selectinload(KnowledgeGap.sessions),)

    async def list_recent(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        status: KnowledgeGapStatus | None = None,
    ) -> list[KnowledgeGap]:
        stmt = select(KnowledgeGap)
        if status is not None:
            stmt = stmt.where(KnowledgeGap.status == status)
        stmt = stmt.order_by(KnowledgeGap.last_seen_at.desc(), KnowledgeGap.id)
        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def set_status(
        self,
        gap_id: uuid.UUID,
        *,
        status: KnowledgeGapStatus,
        resolved_person_id: uuid.UUID | None = None,
        resolved_fact_id: uuid.UUID | None = None,
    ) -> KnowledgeGap | None:
        """Apply a manual status transition to a knowledge gap.

        Uses the existing ``knowledge_gap_status`` model. Resolving optionally
        links the gap to the person or fact that closed it (the audit trail);
        any non-``resolved`` transition clears those links. No promotion logic
        lives here — this only changes the gap's own status.
        """
        gap = await self.get(gap_id)
        if gap is None:
            return None
        gap.status = status
        if status is KnowledgeGapStatus.resolved:
            gap.resolved_person_id = resolved_person_id
            gap.resolved_fact_id = resolved_fact_id
        else:
            gap.resolved_person_id = None
            gap.resolved_fact_id = None
        await self._session.commit()
        # ``updated_at`` is computed by ``onupdate`` on the server side, so the
        # flush expires it; reload so the row is safe to serialize outside an
        # async context.
        await self._session.refresh(gap)
        return gap

    async def upsert(
        self,
        *,
        gap_type: str,
        canonical_reference: str,
        conversation_id: uuid.UUID,
        description: str | None = None,
        now: datetime | None = None,
    ) -> tuple[KnowledgeGap, bool]:
        """Record an occurrence of a knowledge gap.

        The gap is keyed by ``(gap_type, canonical_reference)``. A new
        occurrence increments ``occurrence_count``; a first occurrence from a
        conversation that has not seen the gap before also increments
        ``distinct_sessions``. Returns the gap and whether it was created.
        """
        moment = now or datetime.now(UTC)
        stmt = select(KnowledgeGap).where(
            KnowledgeGap.gap_type == gap_type,
            KnowledgeGap.canonical_reference == canonical_reference,
        )
        gap = (await self._session.execute(stmt)).scalar_one_or_none()

        if gap is None:
            gap = KnowledgeGap(
                gap_type=gap_type,
                canonical_reference=canonical_reference,
                description=description,
                first_seen_at=moment,
                last_seen_at=moment,
                occurrence_count=1,
                distinct_sessions=1,
                status=KnowledgeGapStatus.observed,
            )
            self._session.add(gap)
            await self._session.flush()
            self._session.add(
                KnowledgeGapConversation(gap_id=gap.id, conversation_id=conversation_id)
            )
            await self._session.commit()
            return gap, True

        gap.occurrence_count += 1
        gap.last_seen_at = moment
        existing = await self._session.execute(
            select(KnowledgeGapConversation).where(
                KnowledgeGapConversation.gap_id == gap.id,
                KnowledgeGapConversation.conversation_id == conversation_id,
            )
        )
        if existing.scalar_one_or_none() is None:
            self._session.add(
                KnowledgeGapConversation(gap_id=gap.id, conversation_id=conversation_id)
            )
            gap.distinct_sessions += 1
        await self._session.commit()
        return gap, False
