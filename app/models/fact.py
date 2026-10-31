import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Date,
    Enum,
    ForeignKey,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.types import UTCDateTime
from app.domain.enums import FactStatus


class Fact(Base):
    __tablename__ = "facts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    subject_person_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("persons.id"))
    fact_type: Mapped[str] = mapped_column(String(50))
    value_text: Mapped[str | None] = mapped_column(Text)
    value_date: Mapped[date | None] = mapped_column(Date)
    value_number: Mapped[Decimal | None] = mapped_column(Numeric)
    related_person_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("persons.id"))
    place_text: Mapped[str | None] = mapped_column(String(500))
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)
    status: Mapped[FactStatus] = mapped_column(
        Enum(
            FactStatus,
            name="fact_status",
            values_callable=lambda values: [item.value for item in values],
        )
    )
    source_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sources.id"))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now(), onupdate=func.now()
    )

    subject_person: Mapped["Person"] = relationship(foreign_keys=[subject_person_id])
    related_person: Mapped["Person | None"] = relationship(foreign_keys=[related_person_id])
    source: Mapped["Source"] = relationship()
