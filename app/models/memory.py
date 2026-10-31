import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.types import UTCDateTime
from app.domain.enums import MemoryStatus, SensitivityLevel

EMBEDDING_DIMENSIONS = 1536


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(255))
    narrative: Mapped[str] = mapped_column(Text)
    approx_start_date: Mapped[date | None] = mapped_column(Date)
    approx_end_date: Mapped[date | None] = mapped_column(Date)
    approx_period_text: Mapped[str | None] = mapped_column(String(255))
    place_text: Mapped[str | None] = mapped_column(String(500))
    memory_status: Mapped[MemoryStatus] = mapped_column(
        Enum(
            MemoryStatus,
            name="memory_status",
            values_callable=lambda values: [item.value for item in values],
        )
    )
    sensitivity_level: Mapped[SensitivityLevel] = mapped_column(
        Enum(
            SensitivityLevel,
            name="sensitivity_level",
            values_callable=lambda values: [item.value for item in values],
        )
    )
    source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sources.id"))
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now(), onupdate=func.now()
    )

    source: Mapped["Source"] = relationship()
    people: Mapped[list["MemoryPerson"]] = relationship(
        back_populates="memory", cascade="all, delete-orphan"
    )
    media: Mapped[list["MemoryMedia"]] = relationship(
        back_populates="memory", cascade="all, delete-orphan"
    )


class MemoryPerson(Base):
    __tablename__ = "memory_people"

    memory_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("memories.id"), primary_key=True
    )
    person_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("persons.id"), primary_key=True)
    role: Mapped[str | None] = mapped_column(String(50))

    memory: Mapped["Memory"] = relationship(back_populates="people")
    person: Mapped["Person"] = relationship()
