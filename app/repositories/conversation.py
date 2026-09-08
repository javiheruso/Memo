import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import Conversation, Message, ResponseSource
from app.repositories.base import SQLRepository


@dataclass(frozen=True)
class ResponseSourceRef:
    entity_type: str
    entity_id: uuid.UUID
    usage_type: str | None = None


class ConversationRepository(SQLRepository[Conversation]):
    model = Conversation

    async def get_or_create(self, conversation_id: uuid.UUID | None) -> Conversation:
        if conversation_id is not None:
            existing = await self.get(conversation_id)
            if existing is not None:
                return existing
        return await self._create(
            lambda: Conversation(
                started_at=datetime.now(UTC),
                initiated_by="user",
                initiation_reason=None,
            )
        )

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        *,
        speaker: str,
        text: str,
        intent: str | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            speaker=speaker,
            text=text,
            intent=intent,
            created_at=datetime.now(UTC),
        )
        self._session.add(message)
        await self._session.commit()
        return message

    async def add_response_sources(
        self, message_id: uuid.UUID, sources: list[ResponseSourceRef]
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
        await self._session.commit()
        return created

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

    async def list_messages(self, conversation_id: uuid.UUID) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .options(selectinload(Message.response_sources))
            .order_by(Message.created_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())
