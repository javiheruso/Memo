import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.claims.base import ClaimCandidate
from app.domain.enums import ClaimStatus
from app.models import Claim
from app.repositories.base import SQLRepository


class ClaimRepository(SQLRepository[Claim]):
    model = Claim

    async def add_many(
        self,
        candidates: list[ClaimCandidate],
        *,
        conversation_id: uuid.UUID,
        message_id: uuid.UUID,
        speaker_person_id: uuid.UUID | None = None,
        first_seen_at: datetime | None = None,
    ) -> list[Claim]:
        """Persist candidate claims as append-only rows tied to a message."""
        now = first_seen_at or datetime.now(UTC)
        created = [
            Claim(
                speaker_person_id=speaker_person_id,
                canonical_subject=candidate.canonical_subject,
                canonical_predicate=candidate.canonical_predicate,
                canonical_object=candidate.canonical_object,
                original_text=candidate.original_text,
                conversation_id=conversation_id,
                message_id=message_id,
                first_seen_at=now,
                status=ClaimStatus.candidate,
            )
            for candidate in candidates
        ]
        self._session.add_all(created)
        await self._session.commit()
        return created

    async def list_for_message(self, message_id: uuid.UUID) -> list[Claim]:
        stmt = (
            select(Claim)
            .where(Claim.message_id == message_id)
            .order_by(Claim.created_at, Claim.id)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def list_for_conversation(self, conversation_id: uuid.UUID) -> list[Claim]:
        stmt = (
            select(Claim)
            .where(Claim.conversation_id == conversation_id)
            .order_by(Claim.created_at, Claim.id)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def list_recent(self, *, limit: int = 100, offset: int = 0) -> list[Claim]:
        stmt = (
            select(Claim)
            .options(selectinload(Claim.membership))
            .order_by(Claim.created_at.desc(), Claim.id)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())
