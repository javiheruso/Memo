import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.types import UTCDateTime


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime())
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    initiated_by: Mapped[str] = mapped_column(String(50))
    initiation_reason: Mapped[str | None] = mapped_column(String(500))
    device_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now()
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"))
    speaker: Mapped[str] = mapped_column(String(50))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now()
    )
    intent: Mapped[str | None] = mapped_column(String(100))
    audio_path: Mapped[str | None] = mapped_column(String(500))

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
    response_sources: Mapped[list["ResponseSource"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )


class ResponseSource(Base):
    __tablename__ = "response_sources"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("messages.id"))
    source_entity_type: Mapped[str] = mapped_column(String(50))
    source_entity_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    usage_type: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now()
    )

    message: Mapped["Message"] = relationship(back_populates="response_sources")
