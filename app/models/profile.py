"""Conversational profile and configured sensitive-topic policy.

A profile belongs to the person using the companion and carries the durable
preferences the conversation pipeline needs: how to address them, what to talk
about, and which topics must never be initiated spontaneously. Sensitive
topics are stored as their own rows so the policy can be queried and audited
without parsing free text.

The profile is deliberately small. It is not a full onboarding model: consent
revocation, roles, media and broader preference features are deferred.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ConversationProfile(Base):
    __tablename__ = "conversation_profiles"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    person_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("persons.id"), unique=True
    )
    preferred_name: Mapped[str | None] = mapped_column(String(255))
    language: Mapped[str | None] = mapped_column(String(50))
    conversational_style: Mapped[str | None] = mapped_column(Text)
    positive_topics: Mapped[list[str]] = mapped_column(JSON, default=list)
    beliefs_policy: Mapped[str | None] = mapped_column(Text)
    never_do: Mapped[list[str]] = mapped_column(JSON, default=list)
    orientation: Mapped[str | None] = mapped_column(Text)
    data_origin: Mapped[str | None] = mapped_column(Text)
    source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sources.id"))
    is_fictional: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    sensitive_topics: Mapped[list["ConversationProfileSensitiveTopic"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )
    source: Mapped["Source | None"] = relationship()


class ConversationProfileSensitiveTopic(Base):
    __tablename__ = "conversation_profile_sensitive_topics"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "topic",
            name="uq_conversation_profile_sensitive_topics_profile_topic",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    profile_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversation_profiles.id")
    )
    topic: Mapped[str] = mapped_column(String(255))
    handling: Mapped[str | None] = mapped_column(Text)
    consent_review_required: Mapped[bool] = mapped_column(Boolean, default=False)
    triggers: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    profile: Mapped["ConversationProfile"] = relationship(
        back_populates="sensitive_topics"
    )
