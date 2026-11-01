from sqlalchemy import select

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
