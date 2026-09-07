import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.domain.enums import ClaimGroupStatus


class ClaimGroup(Base):
    """Groups equivalent candidate claims.

    Baseline grouping is deterministic and conservative: members share the same
    normalized claim text (``canonical_text``), so paraphrases do not merge.
    Denormalized recurrence counters are refreshed from the linked claims on
    every attach, keeping the read path trivial for the admin surface. No
    promotion or family-review logic lives here; that belongs to later phases
    (dossier doc 04). ``needs_review`` is a pure signal flag: it is only ever
    set to ``True`` by contradiction detection and never promotes or rejects
    anything.

    ``status`` is the manual review state (default ``active``). An admin sets a
    group to ``resolved`` or ``ignored`` through the admin surface; such groups
    leave the review queue and contradiction detection skips them, so they are
    never re-flagged immediately. ``resolved_at`` and ``resolution_note`` keep
    the resolution auditable.
    """

    __tablename__ = "claim_groups"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    canonical_text: Mapped[str] = mapped_column(String(500), unique=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    total_mentions: Mapped[int] = mapped_column(Integer, default=0)
    distinct_conversations: Mapped[int] = mapped_column(Integer, default=0)
    distinct_days: Mapped[int] = mapped_column(Integer, default=0)
    needs_review: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, server_default=false()
    )
    status: Mapped[ClaimGroupStatus] = mapped_column(
        Enum(
            ClaimGroupStatus,
            name="claim_group_status",
            values_callable=lambda values: [item.value for item in values],
        ),
        default=ClaimGroupStatus.active,
        server_default=ClaimGroupStatus.active.value,
        nullable=False,
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    members: Mapped[list["ClaimGroupMember"]] = relationship(
        back_populates="group", cascade="all, delete-orphan"
    )
    contradictions: Mapped[list["ClaimGroupContradiction"]] = relationship(
        back_populates="group", cascade="all, delete-orphan"
    )

    @property
    def claims(self) -> list["Claim"]:
        """Member claims, read through the association (loaded eagerly by the repo)."""
        return [member.claim for member in self.members]


class ClaimGroupMember(Base):
    __tablename__ = "claim_group_members"

    claim_group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claim_groups.id"), primary_key=True
    )
    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claims.id"), primary_key=True, unique=True
    )
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    group: Mapped["ClaimGroup"] = relationship(back_populates="members")
    claim: Mapped["Claim"] = relationship(back_populates="membership")


class ClaimGroupContradiction(Base):
    """A contradiction signal between a claim group and a verified fact.

    Append-only evidence about a detected signal: rows are never deleted and the
    unique ``(claim_group_id, fact_id, detector_name)`` key means a repeated
    detection refreshes ``last_detected_at`` / ``detection_count`` instead of
    duplicating evidence. The fact surface is referenced by id and eager-loaded
    for summaries; the raw fact is never modified here and no claim is ever
    promoted or rejected.
    """

    __tablename__ = "claim_group_contradictions"
    __table_args__ = (
        UniqueConstraint(
            "claim_group_id",
            "fact_id",
            "detector_name",
            name="uq_claim_group_contradictions_group_fact_detector",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    claim_group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claim_groups.id"), index=True
    )
    fact_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("facts.id"), index=True)
    detector_name: Mapped[str] = mapped_column(String(50))
    matched_terms: Mapped[str | None] = mapped_column(Text)
    first_detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    detection_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    group: Mapped["ClaimGroup"] = relationship(back_populates="contradictions")
    fact: Mapped["Fact"] = relationship()


class ClaimGroupRememberedKey(Base):
    """Route map left behind when a claim group is absorbed into a survivor.

    When a human merges a source claim group into a surviving group, the source
    group's canonical text is recorded here so that a future claim using that
    absorbed wording routes to the survivor instead of recreating the deleted
    group (resolved in ``ClaimGroupRepository.attach``). Rows are keyed by the
    absorbed canonical text, so each absorbed wording maps to exactly one
    survivor; when the survivor is itself merged later, its remembered keys are
    re-pointed to the new survivor before the group row is deleted. The rows are
    append-only in spirit but the ``claim_group_id`` target is intentionally
    mutable so a chain of merges keeps routing to the live survivor.
    """

    __tablename__ = "claim_group_remembered_keys"

    canonical_text: Mapped[str] = mapped_column(String(500), primary_key=True)
    claim_group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claim_groups.id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    group: Mapped["ClaimGroup"] = relationship()
