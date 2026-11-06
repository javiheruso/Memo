import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.domain.enums import Intent
from app.models import Conversation, Message, ResponseSource
from app.repositories.base import SQLRepository


@dataclass(frozen=True)
class ResponseSourceRef:
    entity_type: str
    entity_id: uuid.UUID
    usage_type: str | None = None


class ConversationAccessDenied(Exception):
    """Raised when a caller tries to use a conversation it does not own."""


class ConversationRepository(SQLRepository[Conversation]):
    model = Conversation

    async def get_or_create(
        self,
        conversation_id: uuid.UUID | None,
        *,
        owner_session_id: str | None = None,
        commit: bool = True,
    ) -> Conversation:
        if conversation_id is not None:
            existing = await self.get(conversation_id)
            if existing is not None:
                if (
                    owner_session_id is not None
                    and existing.owner_session_id != owner_session_id
                ):
                    raise ConversationAccessDenied
                return existing
            if owner_session_id is not None:
                raise ConversationAccessDenied
        return await self._create(
            lambda: Conversation(
                started_at=datetime.now(UTC),
                initiated_by="user",
                initiation_reason=None,
                owner_session_id=owner_session_id,
            ),
            commit=commit,
        )

    async def get_owned(
        self, conversation_id: uuid.UUID, owner_session_id: str
    ) -> Conversation | None:
        conversation = await self.get(conversation_id)
        if (
            conversation is None
            or conversation.owner_session_id != owner_session_id
        ):
            return None
        return conversation

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        *,
        speaker: str,
        text: str,
        intent: str | None = None,
        commit: bool = True,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            speaker=speaker,
            text=text,
            intent=intent,
            created_at=datetime.now(UTC),
        )
        self._session.add(message)
        if commit:
            await self._session.commit()
        else:
            await self._session.flush()
        return message

    async def add_response_sources(
        self,
        message_id: uuid.UUID,
        sources: list[ResponseSourceRef],
        *,
        commit: bool = True,
    ) -> list[ResponseSource]:
        """Persist response sources preserving the given order.

        The rows are stamped with strictly increasing timestamps so a later
        ``(created_at, id)`` read returns them in insertion order. This keeps
        the ambiguity candidates a follow-up message can pick from ordered
        exactly as the assistant listed them, regardless of the random uuid
        primary keys.
        """
        base = datetime.now(UTC)
        created = [
            ResponseSource(
                message_id=message_id,
                source_entity_type=source.entity_type,
                source_entity_id=source.entity_id,
                usage_type=source.usage_type,
                created_at=base + timedelta(microseconds=index),
            )
            for index, source in enumerate(sources)
        ]
        self._session.add_all(created)
        if commit:
            await self._session.commit()
        else:
            await self._session.flush()
        return created

    async def commit_turn(self) -> None:
        """Commit the atomic persistence unit of one conversation turn."""
        await self._session.commit()

    async def rollback_turn(self) -> None:
        """Roll back a failed turn so no partial message/source pair remains."""
        await self._session.rollback()

    async def last_assistant_candidates(
        self, conversation_id: uuid.UUID
    ) -> list[uuid.UUID]:
        """Return the pending person ambiguity candidates of the latest reply.

        Loads the most recent assistant ``Message`` of the conversation and
        returns the ``source_entity_id`` values of its ``person`` rows whose
        ``usage_type`` is ``candidate``, in the order the assistant presented
        them. Returns ``[]`` when there is no assistant message or no
        candidate sources.
        """
        latest = (
            await self._session.execute(
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.speaker == "assistant",
                )
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if latest is None:
            return []
        rows = (
            await self._session.execute(
                select(ResponseSource)
                .where(
                    ResponseSource.message_id == latest.id,
                    ResponseSource.source_entity_type == "person",
                    ResponseSource.usage_type == "candidate",
                )
                .order_by(ResponseSource.created_at, ResponseSource.id)
            )
        ).scalars()
        return [row.source_entity_id for row in rows]

    async def last_fact_subject(self, conversation_id: uuid.UUID) -> uuid.UUID | None:
        """Return the subject of the immediately previous fact answer.

        Follow-up continuity is deliberately narrow: the most recent assistant
        turn must be a resolved ``fact_query`` answer with exactly one
        ``subject`` person source. An ambiguous answer (``candidate`` rows) or
        any other intent returns ``None``, so a subject is never carried across
        unrelated turns or an unresolved ambiguity.
        """
        latest = (
            await self._session.execute(
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.speaker == "assistant",
                )
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if latest is None or latest.intent != Intent.fact_query.value:
            return None
        rows = list(
            (
                await self._session.execute(
                    select(ResponseSource).where(
                        ResponseSource.message_id == latest.id,
                        ResponseSource.source_entity_type == "person",
                    )
                )
            ).scalars()
        )
        if any(row.usage_type == "candidate" for row in rows):
            return None
        subjects = {row.source_entity_id for row in rows if row.usage_type == "subject"}
        if len(subjects) != 1:
            return None
        return next(iter(subjects))

    async def list_messages(self, conversation_id: uuid.UUID) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .options(selectinload(Message.response_sources))
            .order_by(Message.created_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())
