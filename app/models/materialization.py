import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.domain.enums import (
    MaterializationSourceType,
    MaterializationTargetType,
)


class Materialization(Base):
    """Append-only provenance record for an explicit admin materialization.

    A row is written for every fact/memory created from a claim group or a
    knowledge gap. ``source_type``/``source_id`` identify the reviewed evidence
    and ``target_type``/``target_id`` the created fact or memory. The unique
    ``(source_type, source_id, target_type)`` key makes materialization
    idempotent per source/target pair: the same source can become one fact and
    one memory, but never two of the same kind. Rows are never deleted, so the
    provenance history is append-only. The polymorphic ids intentionally carry
    no foreign keys (a source id can point at either a claim group or a gap),
    so referential integrity is enforced by the service, which loads the source
    and target rows before writing.
    """

    __tablename__ = "materializations"
    __table_args__ = (
        UniqueConstraint(
            "source_type",
            "source_id",
            "target_type",
            name="uq_materializations_source_target_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source_type: Mapped[MaterializationSourceType] = mapped_column(
        Enum(
            MaterializationSourceType,
            name="materialization_source_type",
            values_callable=lambda values: [item.value for item in values],
        ),
        nullable=False,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, index=True)
    target_type: Mapped[MaterializationTargetType] = mapped_column(
        Enum(
            MaterializationTargetType,
            name="materialization_target_type",
            values_callable=lambda values: [item.value for item in values],
        ),
        nullable=False,
    )
    target_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
