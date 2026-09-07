import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import ClaimGroupContradiction
from app.repositories.base import SQLRepository


class ClaimGroupContradictionRepository(SQLRepository[ClaimGroupContradiction]):
    """Async data access for contradiction signals.

    Read-only except for ``upsert``, which records a signal append-only: a
    repeated detection on the same (group, fact, detector) key refreshes the
    detection timestamps and counter instead of creating new evidence rows.
    """

    model = ClaimGroupContradiction

    async def upsert(
        self,
        *,
        claim_group_id: uuid.UUID,
        fact_id: uuid.UUID,
        detector_name: str,
        matched_terms: str | None,
        now: datetime | None = None,
    ) -> ClaimGroupContradiction:
        now = now or datetime.now(UTC)
        stmt = select(ClaimGroupContradiction).where(
            ClaimGroupContradiction.claim_group_id == claim_group_id,
            ClaimGroupContradiction.fact_id == fact_id,
            ClaimGroupContradiction.detector_name == detector_name,
        )
        existing = (await self._session.execute(stmt)).scalar_one_or_none()
        if existing is None:
            existing = ClaimGroupContradiction(
                claim_group_id=claim_group_id,
                fact_id=fact_id,
                detector_name=detector_name,
                matched_terms=matched_terms,
                first_detected_at=now,
                last_detected_at=now,
                detection_count=1,
            )
            self._session.add(existing)
        else:
            existing.last_detected_at = now
            existing.matched_terms = matched_terms
            existing.detection_count += 1
        return existing

    async def list_recent(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[ClaimGroupContradiction]:
        stmt = (
            select(ClaimGroupContradiction)
            .options(
                selectinload(ClaimGroupContradiction.group),
                selectinload(ClaimGroupContradiction.fact),
            )
            .order_by(
                ClaimGroupContradiction.last_detected_at.desc(),
                ClaimGroupContradiction.id,
            )
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())
