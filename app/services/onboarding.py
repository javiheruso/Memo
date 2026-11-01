"""Admin/family-led onboarding workflow.

The service owns the draft lifecycle. Drafts are resumable JSON validated by
the onboarding schemas; activation is a separate, consent-gated step that will
materialize the real rows through the existing repositories in a later work
unit.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import OnboardingStatus
from app.models import OnboardingSession
from app.repositories.onboarding import OnboardingRepository
from app.schemas.onboarding import OnboardingDraft


class OnboardingError(Exception):
    """Base class for onboarding workflow failures."""


class OnboardingNotFoundError(OnboardingError):
    """The requested onboarding session does not exist."""


class OnboardingNotDraftError(OnboardingError):
    """The session is already activated and can no longer be edited."""


class OnboardingService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._onboarding = OnboardingRepository(session)

    async def create(
        self, draft: OnboardingDraft | None = None
    ) -> OnboardingSession:
        row = OnboardingSession(
            status=OnboardingStatus.draft,
            payload=(draft or OnboardingDraft()).model_dump(mode="json"),
        )
        self._session.add(row)
        await self._session.commit()
        return await self._require(row.id)

    async def get(self, session_id: uuid.UUID) -> OnboardingSession:
        return await self._require(session_id)

    async def list(self, *, limit: int = 100) -> list[OnboardingSession]:
        return await self._onboarding.list_recent(limit=limit)

    async def save_draft(
        self, session_id: uuid.UUID, draft: OnboardingDraft
    ) -> OnboardingSession:
        row = await self._require(session_id)
        if row.status is not OnboardingStatus.draft:
            raise OnboardingNotDraftError(
                "Activated onboarding sessions cannot be edited"
            )
        row.payload = draft.model_dump(mode="json")
        await self._session.commit()
        return await self._require(session_id)

    def draft_of(self, row: OnboardingSession) -> OnboardingDraft:
        return OnboardingDraft.model_validate(row.payload or {})

    async def _require(self, session_id: uuid.UUID) -> OnboardingSession:
        row = await self._onboarding.get(session_id)
        if row is None:
            raise OnboardingNotFoundError("Onboarding session not found")
        return row
