import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.domain.enums import KnowledgeGapStatus


class KnowledgeGap(Base):
    __tablename__ = "knowledge_gaps"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    gap_type: Mapped[str] = mapped_column(String(50))
    canonical_reference: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    occurrence_count: Mapped[int] = mapped_column(Integer, default=0)
    distinct_sessions: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[KnowledgeGapStatus] = mapped_column(
        Enum(
            KnowledgeGapStatus,
            name="knowledge_gap_status",
            values_callable=lambda values: [item.value for item in values],
        )
    )
    resolved_person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("persons.id")
    )
    resolved_fact_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("facts.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    resolved_person: Mapped["Person | None"] = relationship(
        foreign_keys=[resolved_person_id]
    )
    resolved_fact: Mapped["Fact | None"] = relationship(foreign_keys=[resolved_fact_id])
    sessions: Mapped[list["KnowledgeGapConversation"]] = relationship(
        back_populates="gap", cascade="all, delete-orphan"
    )


class KnowledgeGapConversation(Base):
    __tablename__ = "knowledge_gap_conversations"

    gap_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_gaps.id"), primary_key=True
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    gap: Mapped["KnowledgeGap"] = relationship(back_populates="sessions")
