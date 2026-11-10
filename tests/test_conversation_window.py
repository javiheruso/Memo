"""Read-only conversation windows and their narrow continuity rules.

The window is the single bounded snapshot the service derives continuity from.
These tests pin its ordering, limit, source order, ownership scoping and the
strict semantics that preserve the existing ambiguous/unrelated-turn behavior.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.conversation_window import (
    ConversationWindow,
    SourceSnapshot,
    TurnSnapshot,
)
from app.domain.enums import Intent
from app.models import Conversation, Message, ResponseSource
from app.repositories.conversation import (
    DEFAULT_CONVERSATION_WINDOW_LIMIT,
    MAX_CONVERSATION_WINDOW_LIMIT,
    ConversationRepository,
    ResponseSourceRef,
)


async def _conversation(
    db_session: AsyncSession, *, owner_session_id: str | None = None
) -> Conversation:
    return await ConversationRepository(db_session).get_or_create(
        None, owner_session_id=owner_session_id
    )


async def _message(
    db_session: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    speaker: str,
    text: str,
    created_at: datetime,
    intent: str | None = None,
    sources: tuple[tuple[str, uuid.UUID, str | None], ...] = (),
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        speaker=speaker,
        text=text,
        intent=intent,
        created_at=created_at,
    )
    db_session.add(message)
    await db_session.flush()
    for index, (entity_type, entity_id, usage_type) in enumerate(sources):
        db_session.add(
            ResponseSource(
                message_id=message.id,
                source_entity_type=entity_type,
                source_entity_id=entity_id,
                usage_type=usage_type,
                created_at=created_at + timedelta(microseconds=index + 1),
            )
        )
    await db_session.commit()
    return message


def _turn(
    *,
    speaker: str = "assistant",
    intent: str | None = None,
    sources: tuple[SourceSnapshot, ...] = (),
) -> TurnSnapshot:
    return TurnSnapshot(
        message_id=uuid.uuid4(),
        speaker=speaker,
        text="text",
        intent=intent,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        sources=sources,
    )


def _source(
    entity_id: uuid.UUID, *, usage_type: str, entity_type: str = "person"
) -> SourceSnapshot:
    return SourceSnapshot(
        entity_type=entity_type,
        entity_id=entity_id,
        usage_type=usage_type,
    )


async def test_window_loads_only_its_conversation_oldest_to_newest(
    db_session: AsyncSession,
) -> None:
    repository = ConversationRepository(db_session)
    current = await _conversation(db_session)
    other = await _conversation(db_session)
    start = datetime(2026, 1, 1, tzinfo=UTC)

    first = await _message(
        db_session, current.id, speaker="user", text="first", created_at=start
    )
    second = await _message(
        db_session,
        current.id,
        speaker="assistant",
        text="second",
        created_at=start + timedelta(seconds=1),
    )
    third = await _message(
        db_session,
        current.id,
        speaker="user",
        text="third",
        created_at=start + timedelta(seconds=2),
    )
    await _message(
        db_session,
        other.id,
        speaker="assistant",
        text="leaked",
        created_at=start + timedelta(seconds=3),
    )

    window = await repository.load_window(current.id)

    assert window is not None
    assert window.conversation_id == current.id
    assert [turn.message_id for turn in window.turns] == [
        first.id,
        second.id,
        third.id,
    ]
    assert [turn.text for turn in window.turns] == ["first", "second", "third"]
    assert all(turn.text != "leaked" for turn in window.turns)


async def test_window_limit_keeps_the_most_recent_messages_in_ascending_order(
    db_session: AsyncSession,
) -> None:
    conversation = await _conversation(db_session)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    messages = [
        await _message(
            db_session,
            conversation.id,
            speaker="user" if index % 2 == 0 else "assistant",
            text=f"message-{index}",
            created_at=start + timedelta(seconds=index),
        )
        for index in range(6)
    ]

    window = await ConversationRepository(db_session).load_window(
        conversation.id, limit=3
    )

    assert window is not None
    assert [turn.text for turn in window.turns] == [
        "message-3",
        "message-4",
        "message-5",
    ]
    assert [turn.message_id for turn in window.turns] == [
        message.id for message in messages[-3:]
    ]


async def test_window_limit_is_positive_and_hard_capped(
    db_session: AsyncSession,
) -> None:
    conversation = await _conversation(db_session)
    repository = ConversationRepository(db_session)

    assert DEFAULT_CONVERSATION_WINDOW_LIMIT <= MAX_CONVERSATION_WINDOW_LIMIT
    with pytest.raises(ValueError, match="positive"):
        await repository.load_window(conversation.id, limit=0)

    start = datetime(2026, 1, 1, tzinfo=UTC)
    extra = 5
    for index in range(MAX_CONVERSATION_WINDOW_LIMIT + extra):
        db_session.add(
            Message(
                conversation_id=conversation.id,
                speaker="user",
                text=f"message-{index}",
                created_at=start + timedelta(microseconds=index),
            )
        )
    await db_session.commit()

    window = await repository.load_window(
        conversation.id, limit=MAX_CONVERSATION_WINDOW_LIMIT + 100
    )

    assert window is not None
    assert len(window.turns) == MAX_CONVERSATION_WINDOW_LIMIT
    assert window.turns[0].text == f"message-{extra}"


async def test_window_source_snapshots_keep_the_presented_order(
    db_session: AsyncSession,
) -> None:
    repository = ConversationRepository(db_session)
    conversation = await _conversation(db_session)
    assistant = await repository.add_message(
        conversation.id, speaker="assistant", text="answer"
    )
    subject_id = uuid.uuid4()
    candidate_one = uuid.uuid4()
    candidate_two = uuid.uuid4()
    provenance_id = uuid.uuid4()

    await repository.add_response_sources(
        assistant.id,
        [
            ResponseSourceRef("person", subject_id, "subject"),
            ResponseSourceRef("person", candidate_one, "candidate"),
            ResponseSourceRef("person", candidate_two, "candidate"),
            ResponseSourceRef("source", provenance_id, "provenance"),
        ],
    )

    window = await repository.load_window(conversation.id)

    assert window is not None
    assert len(window.turns) == 1
    turn = window.turns[0]
    assert [
        (source.entity_type, source.entity_id, source.usage_type)
        for source in turn.sources
    ] == [
        ("person", subject_id, "subject"),
        ("person", candidate_one, "candidate"),
        ("person", candidate_two, "candidate"),
        ("source", provenance_id, "provenance"),
    ]


async def test_window_is_owner_session_safe_and_scoped(
    db_session: AsyncSession,
) -> None:
    repository = ConversationRepository(db_session)
    owned = await _conversation(db_session, owner_session_id="owner-a")
    await _message(
        db_session,
        owned.id,
        speaker="assistant",
        text="private",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        intent=Intent.person_query.value,
        sources=(("person", uuid.uuid4(), "subject"),),
    )

    assert (
        await repository.load_window(owned.id, owner_session_id="owner-a") is not None
    )
    assert await repository.load_window(owned.id, owner_session_id="owner-b") is None
    assert await repository.load_window(uuid.uuid4()) is None

    # Callers that do not run an owned session (legacy/internal paths) still
    # get the same conversation they already validated.
    assert await repository.load_window(owned.id) is not None


def test_contract_keeps_ambiguous_and_unrelated_turns_narrow() -> None:
    subject = uuid.uuid4()
    candidate_one = uuid.uuid4()
    candidate_two = uuid.uuid4()

    ambiguous = ConversationWindow(
        conversation_id=uuid.uuid4(),
        turns=(
            _turn(
                intent=Intent.person_query.value,
                sources=(
                    _source(subject, usage_type="subject"),
                    _source(candidate_one, usage_type="candidate"),
                    _source(candidate_two, usage_type="candidate"),
                ),
            ),
        ),
    )
    assert ambiguous.pending_person_candidates() == (
        candidate_one,
        candidate_two,
    )
    assert ambiguous.last_fact_subject() is None
    assert ambiguous.last_resolved_person() is None

    unrelated = ConversationWindow(
        conversation_id=uuid.uuid4(),
        turns=(
            _turn(
                intent=Intent.general_chat.value,
                sources=(_source(subject, usage_type="subject"),),
            ),
        ),
    )
    assert unrelated.pending_person_candidates() == ()
    assert unrelated.last_fact_subject() is None
    assert unrelated.last_resolved_person() is None


def test_contract_distinguishes_person_from_fact_subjects() -> None:
    fact_subject = uuid.uuid4()
    person_subject = uuid.uuid4()

    fact_window = ConversationWindow(
        conversation_id=uuid.uuid4(),
        turns=(
            _turn(
                intent=Intent.fact_query.value,
                sources=(_source(fact_subject, usage_type="subject"),),
            ),
        ),
    )
    assert fact_window.last_fact_subject() == fact_subject
    assert fact_window.last_resolved_person() == fact_subject

    person_window = ConversationWindow(
        conversation_id=uuid.uuid4(),
        turns=(
            _turn(
                intent=Intent.person_query.value,
                sources=(_source(person_subject, usage_type="subject"),),
            ),
        ),
    )
    assert person_window.last_fact_subject() is None
    assert person_window.last_resolved_person() == person_subject

    # A later user message must not hide the previous assistant turn.
    with_trailing_user = ConversationWindow(
        conversation_id=uuid.uuid4(),
        turns=(
            _turn(
                intent=Intent.fact_query.value,
                sources=(_source(fact_subject, usage_type="subject"),),
            ),
            _turn(speaker="user", intent=Intent.general_chat.value),
        ),
    )
    assert with_trailing_user.last_fact_subject() == fact_subject
