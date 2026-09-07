import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.domain.enums import EventStatus


class Event(Base):
    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[EventStatus] = mapped_column(
        Enum(
            EventStatus,
            name="event_status",
            values_callable=lambda values: [item.value for item in values],
        )
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sources.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    source: Mapped["Source | None"] = relationship()
    people: Mapped[list["EventPerson"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )


class EventPerson(Base):
    __tablename__ = "event_people"

    event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("events.id"), primary_key=True)
    person_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("persons.id"), primary_key=True)
    role: Mapped[str | None] = mapped_column(String(50))

    event: Mapped["Event"] = relationship(back_populates="people")
    person: Mapped["Person"] = relationship()
