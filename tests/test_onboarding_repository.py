from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import OnboardingStatus
from app.models import OnboardingSession
from app.repositories.onboarding import OnboardingRepository
from app.schemas.onboarding import OnboardingDraft


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


async def test_claim_activation_is_a_one_way_compare_and_set(
    db_session: AsyncSession,
) -> None:
    row = OnboardingSession(status=OnboardingStatus.draft, payload={})
    db_session.add(row)
    await db_session.commit()

    repo = OnboardingRepository(db_session)
    activated_at = datetime.now(UTC)

    assert await repo.claim_activation(row.id, activated_at=activated_at) is True
    await db_session.commit()
    assert await repo.claim_activation(row.id, activated_at=activated_at) is False
