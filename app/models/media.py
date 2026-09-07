import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Date, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.memory import EMBEDDING_DIMENSIONS, Memory


class Media(Base):
    __tablename__ = "media"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    media_type: Mapped[str] = mapped_column(String(50))
    storage_path: Mapped[str] = mapped_column(String(500))
    title: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    approx_date: Mapped[date | None] = mapped_column(Date)
    place_text: Mapped[str | None] = mapped_column(String(500))
    source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sources.id"))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    source: Mapped["Source | None"] = relationship()
    people: Mapped[list["MediaPerson"]] = relationship(
        back_populates="media", cascade="all, delete-orphan"
    )
    memories: Mapped[list["MemoryMedia"]] = relationship(
        back_populates="media", cascade="all, delete-orphan"
    )


class MediaPerson(Base):
    __tablename__ = "media_people"

    media_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("media.id"), primary_key=True)
    person_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("persons.id"), primary_key=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text)

    media: Mapped["Media"] = relationship(back_populates="people")
    person: Mapped["Person"] = relationship()


class MemoryMedia(Base):
    __tablename__ = "memory_media"

    memory_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("memories.id"), primary_key=True)
    media_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("media.id"), primary_key=True)
    relation_type: Mapped[str | None] = mapped_column(String(50))

    memory: Mapped["Memory"] = relationship(back_populates="media")
    media: Mapped["Media"] = relationship(back_populates="memories")
