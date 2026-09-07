import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.domain.enums import ClaimGroupStatus
from app.models import (
    Claim,
    ClaimGroup,
    ClaimGroupContradiction,
    ClaimGroupMember,
    ClaimGroupRememberedKey,
)
from app.repositories.base import SQLRepository


class ClaimGroupRepository(SQLRepository[ClaimGroup]):
    model = ClaimGroup

    async def get_by_canonical_text(self, canonical_text: str) -> ClaimGroup | None:
        stmt = select(ClaimGroup).where(ClaimGroup.canonical_text == canonical_text)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def attach(
        self,
        claim: Claim,
        group_key: str,
        *,
        now: datetime | None = None,
    ) -> ClaimGroup:
        """Link a claim to the group identified by ``group_key``, creating the
        group when needed, and refresh the denormalized recurrence counters.

        A claim can belong to at most one group (enforced by a unique
        constraint on ``claim_group_members.claim_id``), so attaching an already
        grouped claim is a no-op for membership but still refreshes the metrics.

        Group resolution is: an existing group with that exact canonical text
        wins; otherwise, if the canonical text is a remembered key left behind by
        a human merge, the claim attaches to the surviving group that absorbed
        that wording; only when neither matches is a new group created. This is
        how absorbed claim wording routes future claims to the survivor.
        """
        now = now or datetime.now(UTC)
        group = await self._resolve_group(group_key, now)

        existing = await self._session.get(ClaimGroupMember, (group.id, claim.id))
        if existing is None:
            member = ClaimGroupMember(claim_group_id=group.id)
            member.claim = claim
            self._session.add(member)
            await self._session.flush()

        await self._refresh_metrics(group)
        return group

    async def _resolve_group(self, group_key: str, now: datetime) -> ClaimGroup:
        group = await self.get_by_canonical_text(group_key)
        if group is not None:
            return group

        stmt = select(ClaimGroupRememberedKey).where(
            ClaimGroupRememberedKey.canonical_text == group_key
        )
        remembered = (await self._session.execute(stmt)).scalar_one_or_none()
        if remembered is not None:
            survivor = await self.get(remembered.claim_group_id)
            if survivor is not None:
                return survivor

        group = ClaimGroup(
            canonical_text=group_key,
            first_seen_at=now,
            last_seen_at=now,
            total_mentions=0,
            distinct_conversations=0,
            distinct_days=0,
        )
        self._session.add(group)
        await self._session.flush()
        return group

    async def _refresh_metrics(self, group: ClaimGroup) -> None:
        member_stmt = select(ClaimGroupMember.claim_id).where(
            ClaimGroupMember.claim_group_id == group.id
        )
        member_ids = list((await self._session.execute(member_stmt)).scalars())
        if not member_ids:
            return
        claims_stmt = select(Claim).where(Claim.id.in_(member_ids))
        claims = list((await self._session.execute(claims_stmt)).scalars())
        group.total_mentions = len(claims)
        group.distinct_conversations = len({claim.conversation_id for claim in claims})
        group.distinct_days = len({claim.first_seen_at.date() for claim in claims})
        group.first_seen_at = min(claim.first_seen_at for claim in claims)
        group.last_seen_at = max(claim.first_seen_at for claim in claims)

    async def list_recent(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        needs_review: bool | None = None,
        status: ClaimGroupStatus | None = None,
    ) -> list[ClaimGroup]:
        stmt = select(ClaimGroup)
        if needs_review is not None:
            stmt = stmt.where(ClaimGroup.needs_review == needs_review)
        if status is not None:
            stmt = stmt.where(ClaimGroup.status == status)
        stmt = stmt.order_by(ClaimGroup.last_seen_at.desc(), ClaimGroup.id)
        stmt = stmt.limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def set_status(
        self,
        group_id: uuid.UUID,
        *,
        status: ClaimGroupStatus,
        resolution_note: str | None = None,
        now: datetime | None = None,
    ) -> ClaimGroup | None:
        """Apply a manual review state to a group.

        ``resolved`` and ``ignored`` are terminal: they clear ``needs_review``
        and stamp ``resolved_at`` so the resolution is auditable, and
        contradiction detection skips the group afterwards. Reopening to
        ``active`` clears the audit stamps and puts the group back in the
        review queue. The group's evidence rows and member claims are never
        touched.
        """
        group = await self.get(group_id)
        if group is None:
            return None
        group.status = status
        group.resolution_note = resolution_note
        group.needs_review = False
        group.resolved_at = None if status is ClaimGroupStatus.active else (now or datetime.now(UTC))
        await self._session.commit()
        return group

    async def get_with_members(self, group_id: uuid.UUID) -> ClaimGroup | None:
        stmt = (
            select(ClaimGroup)
            .where(ClaimGroup.id == group_id)
            .options(
                selectinload(ClaimGroup.members)
                .selectinload(ClaimGroupMember.claim)
                .selectinload(Claim.membership),
                selectinload(ClaimGroup.members)
                .selectinload(ClaimGroupMember.claim)
                .selectinload(Claim.message),
                selectinload(ClaimGroup.contradictions).selectinload(
                    ClaimGroupContradiction.fact
                ),
            )
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def refresh_group_metrics(self, group_id: uuid.UUID) -> ClaimGroup | None:
        """Recompute the denormalized recurrence counters of a group.

        Used after the merge transaction moves memberships between groups, so
        the surviving group's counters reflect its new member set without going
        through ``attach`` again.
        """
        group = await self.get(group_id)
        if group is None:
            return None
        await self._refresh_metrics(group)
        return group
