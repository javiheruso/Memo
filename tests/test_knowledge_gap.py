import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import KnowledgeGap, KnowledgeGapConversation
from app.repositories.conversation import ConversationRepository
from app.repositories.knowledge_gap import KnowledgeGapRepository


async def _conversation(db_session: AsyncSession) -> uuid.UUID:
    conversation = await ConversationRepository(db_session).get_or_create(None)
    return conversation.id


async def _gap_count(db_session: AsyncSession) -> int:
    result = await db_session.execute(select(func.count()).select_from(KnowledgeGap))
    return result.scalar_one()


async def test_upsert_creates_gap_with_occurrence(db_session: AsyncSession):
    conversation_id = await _conversation(db_session)
    gap, created = await KnowledgeGapRepository(db_session).upsert(
        gap_type="person_unknown",
        canonical_reference="eustaquio",
        conversation_id=conversation_id,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert created is True
    assert gap.occurrence_count == 1
    assert gap.distinct_sessions == 1
    assert gap.first_seen_at == gap.last_seen_at


async def test_upsert_increments_occurrence_within_same_conversation(
    db_session: AsyncSession,
):
    conversation_id = await _conversation(db_session)
    repo = KnowledgeGapRepository(db_session)
    await repo.upsert(
        gap_type="person_unknown",
        canonical_reference="eustaquio",
        conversation_id=conversation_id,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    gap, created = await repo.upsert(
        gap_type="person_unknown",
        canonical_reference="eustaquio",
        conversation_id=conversation_id,
        now=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert created is False
    assert gap.occurrence_count == 2
    assert gap.distinct_sessions == 1
    assert gap.last_seen_at == datetime(2026, 1, 2, tzinfo=UTC)


async def test_upsert_counts_distinct_conversations(db_session: AsyncSession):
    first = await _conversation(db_session)
    second = await _conversation(db_session)
    assert first != second
    repo = KnowledgeGapRepository(db_session)
    await repo.upsert(
        gap_type="person_unknown",
        canonical_reference="eustaquio",
        conversation_id=first,
    )
    gap, _ = await repo.upsert(
        gap_type="person_unknown",
        canonical_reference="eustaquio",
        conversation_id=second,
    )
    assert gap.occurrence_count == 2
    assert gap.distinct_sessions == 2


async def test_upsert_tracks_session_associations(db_session: AsyncSession):
    conversation_id = await _conversation(db_session)
    repo = KnowledgeGapRepository(db_session)
    gap, _ = await repo.upsert(
        gap_type="person_unknown",
        canonical_reference="eustaquio",
        conversation_id=conversation_id,
    )
    result = await db_session.execute(
        select(KnowledgeGapConversation).where(
            KnowledgeGapConversation.gap_id == gap.id
        )
    )
    associations = list(result.scalars())
    assert len(associations) == 1
    assert associations[0].conversation_id == conversation_id


async def test_different_references_create_separate_gaps(db_session: AsyncSession):
    conversation_id = await _conversation(db_session)
    repo = KnowledgeGapRepository(db_session)
    await repo.upsert(
        gap_type="person_unknown",
        canonical_reference="eustaquio",
        conversation_id=conversation_id,
    )
    await repo.upsert(
        gap_type="person_unknown",
        canonical_reference="rigoberto",
        conversation_id=conversation_id,
    )
    assert await _gap_count(db_session) == 2
