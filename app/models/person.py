import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.domain.enums import RelationshipStatus, RelationshipType


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    first_name: Mapped[str] = mapped_column(String(255))
    last_name: Mapped[str | None] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(255))
    birth_date: Mapped[date | None] = mapped_column(Date)
    death_date: Mapped[date | None] = mapped_column(Date)
    is_deceased: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str | None] = mapped_column(Text)
    primary_media_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("media.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    aliases: Mapped[list["PersonAlias"]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )
    primary_media: Mapped["Media | None"] = relationship(foreign_keys=[primary_media_id])


class PersonAlias(Base):
    __tablename__ = "person_aliases"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    person_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("persons.id"))
    alias: Mapped[str] = mapped_column(String(255))
    alias_type: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    person: Mapped["Person"] = relationship(back_populates="aliases")


class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    person_a_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("persons.id"))
    person_b_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("persons.id"))
    relationship_type: Mapped[RelationshipType] = mapped_column(
        Enum(
            RelationshipType,
            name="relationship_type",
            values_callable=lambda values: [item.value for item in values],
        )
    )
    status: Mapped[RelationshipStatus] = mapped_column(
        Enum(
            RelationshipStatus,
            name="relationship_status",
            values_callable=lambda values: [item.value for item in values],
        ),
        default=RelationshipStatus.verified,
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sources.id"))
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    person_a: Mapped["Person"] = relationship(foreign_keys=[person_a_id])
    person_b: Mapped["Person"] = relationship(foreign_keys=[person_b_id])
