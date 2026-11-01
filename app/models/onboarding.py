"""Admin-led onboarding draft and consent record.

The session is the resumable staging area for the first-day package. It holds
a JSON payload validated by the onboarding schemas; no authority-bearing row
(person, fact, memory or profile) is written until activation. Consent is kept
in explicit columns rather than inside the draft so activation can gate on it
without parsing free-form content. The consent fingerprint binds that consent
to the exact canonical payload the operator reviewed.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.types import UTCDateTime
from app.domain.enums import OnboardingStatus


class OnboardingSession(Base):
    __tablename__ = "onboarding_sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    status: Mapped[OnboardingStatus] = mapped_column(
        Enum(
            OnboardingStatus,
            name="onboarding_status",
            values_callable=lambda values: [item.value for item in values],
        ),
        default=OnboardingStatus.draft,
        index=True,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    consent_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_confirmed_by: Mapped[str | None] = mapped_column(String(255))
    consent_confirmed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    consent_scope: Mapped[str | None] = mapped_column(Text)
    consent_payload_fingerprint: Mapped[str | None] = mapped_column(String(64))
    subject_person_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("persons.id")
    )
    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversation_profiles.id")
    )
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), server_default=func.now(), onupdate=func.now()
    )
