from app.domain.enums import OnboardingStatus
from app.models import OnboardingSession
from app.repositories.onboarding import OnboardingRepository
from app.schemas.onboarding import OnboardingDraft
from sqlalchemy.ext.asyncio import AsyncSession


async def test_onboarding_repository_persists_draft_payload(
    db_session: AsyncSession,
) -> None:
    draft = OnboardingDraft.model_validate(
        {
            "identity": {"first_name": "Elena", "preferred_name": "Elenita"},
            "sources": [{"key": "family", "source_type": "family_member"}],
        }
    )
    row = OnboardingSession(
        status=OnboardingStatus.draft, payload=draft.model_dump(mode="json")
    )
    db_session.add(row)
    await db_session.commit()

    repo = OnboardingRepository(db_session)
    stored = await repo.get(row.id)
    assert stored is not None
    assert stored.status is OnboardingStatus.draft
    assert OnboardingDraft.model_validate(stored.payload).identity.first_name == "Elena"

    listed = await repo.list_recent()
    assert [item.id for item in listed] == [row.id]
