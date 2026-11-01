import uuid
from datetime import datetime

from sqlalchemy import select, update

from app.domain.enums import OnboardingStatus
from app.models import OnboardingSession
from app.repositories.base import SQLRepository


class OnboardingRepository(SQLRepository[OnboardingSession]):
    model = OnboardingSession

    async def list_recent(self, *, limit: int = 100) -> list[OnboardingSession]:
        stmt = (
            select(OnboardingSession)
            .order_by(OnboardingSession.created_at.desc(), OnboardingSession.id)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def get_for_update(self, session_id: uuid.UUID) -> OnboardingSession | None:
        """Load a session while holding its row lock for the transaction.

        PostgreSQL serializes concurrent activations here. SQLite ignores the
        clause, so ``claim_activation`` provides the durable compare-and-set
        that actually decides the winner on every backend.
        """
        stmt = (
            select(OnboardingSession)
            .where(OnboardingSession.id == session_id)
            .with_for_update()
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def claim_activation(
        self, session_id: uuid.UUID, *, activated_at: datetime
    ) -> bool:
        """Atomically move a draft to activated and report whether we won.

        The conditional update is the durable idempotency guard: only one
        transaction can change a draft row to activated, so a double submit
        cannot materialize the package twice.
        """
        stmt = (
            update(OnboardingSession)
            .where(
                OnboardingSession.id == session_id,
                OnboardingSession.status == OnboardingStatus.draft,
            )
            .values(
                status=OnboardingStatus.activated,
                activated_at=activated_at,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(stmt)
        return result.rowcount == 1
