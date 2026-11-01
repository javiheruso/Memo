import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.types import UTCDateTime
from app.domain.enums import ClaimStatus

if TYPE_CHECKING:
    from app.models.conversation import Message


class Claim(Base):
    """A candidate claim extracted from a user message, persisted verbatim.

    Append-only by design: ``original_text`` is never mutated once stored and
    nothing here ever promotes a claim to a fact or a memory. The canonical
    subject/predicate/object triplet stays ``NULL`` in this baseline;
    canonicalization and grouping belong to the claim-grouping phase (dossier
    doc 04), which is intentionally out of scope.
    """

    __tablename__ = "claims"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    speaker_person_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("persons.id"))
    canonical_subject: Mapped[str | None] = mapped_column(String(255))
    canonical_predicate: Mapped[str | None] = mapped_column(String(255))
    canonical_object: Mapped[str | None] = mapped_column(String(500))
    original_text: Mapped[str] = mapped_column(Text)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"))
    message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("messages.id"))
    first_seen_at: Mapped[datetime] = mapped_column(UTCDateTime())
    status: Mapped[ClaimStatus] = mapped_column(
        Enum(
            ClaimStatus,
            name="claim_status",
            values_callable=lambda values: [item.value for item in values],
        )
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now(), onupdate=func.now()
    )

    membership: Mapped["ClaimGroupMember | None"] = relationship(
        back_populates="claim", uselist=False, cascade="all, delete-orphan"
    )
    message: Mapped["Message"] = relationship()

    @property
    def claim_group_id(self) -> uuid.UUID | None:
        """Group this claim currently belongs to, if any (None in the baseline
        before grouping runs, or when grouping failed safely)."""
        return self.membership.claim_group_id if self.membership is not None else None
