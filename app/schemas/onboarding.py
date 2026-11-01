"""Schemas for the admin/family-led onboarding draft.

The draft is permissive enough to save incomplete work: every field is
optional and cross-references between sources, people, facts and memories are
checked only when the operator activates the profile. Individual entry shapes
still reject malformed values immediately so a resumed draft never carries
data the activation step cannot interpret.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import (
    FactStatus,
    OnboardingStatus,
    RelationshipStatus,
    RelationshipType,
    SensitivityLevel,
    SourceType,
)
from app.schemas.profile import SensitiveTopicCreate


class OnboardingModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class OnboardingIdentity(OnboardingModel):
    # Limits mirror the target PersonCreate/ConversationProfileCreate schemas.
    # A draft may be incomplete, but it must never carry a value that the
    # activation materialization step would reject with a validation error.
    first_name: str = Field(default="", max_length=255)
    last_name: str | None = Field(default=None, max_length=255)
    preferred_name: str | None = Field(default=None, max_length=255)
    language: str | None = Field(default="es-ES", max_length=50)


class OnboardingReviewer(OnboardingModel):
    name: str = ""
    relationship: str | None = None


class OnboardingContributor(OnboardingModel):
    name: str = Field(min_length=1, max_length=255)
    relationship: str | None = None
    permission_note: str | None = None


class OnboardingSource(OnboardingModel):
    key: str = Field(min_length=1, max_length=50)
    source_type: SourceType = SourceType.family_member
    description: str | None = None


class OnboardingPerson(OnboardingModel):
    key: str = Field(min_length=1, max_length=50)
    first_name: str = Field(min_length=1, max_length=255)
    last_name: str | None = Field(default=None, max_length=255)
    relationship_type: RelationshipType = RelationshipType.other
    certainty: RelationshipStatus = RelationshipStatus.probable
    source_key: str | None = None
    notes: str | None = None


class OnboardingFact(OnboardingModel):
    subject_key: str = "subject"
    fact_type: str = Field(min_length=1, max_length=50)
    value_text: str | None = None
    status: FactStatus = FactStatus.probable
    source_key: str = Field(min_length=1, max_length=50)
    notes: str | None = None


class OnboardingMemory(OnboardingModel):
    title: str | None = Field(default=None, max_length=255)
    narrative: str = Field(min_length=1)
    sensitivity_level: SensitivityLevel = SensitivityLevel.low
    source_key: str = Field(min_length=1, max_length=50)
    people_keys: list[str] = Field(default_factory=list)


class OnboardingPreferences(OnboardingModel):
    positive_topics: list[str] = Field(default_factory=list)
    never_do: list[str] = Field(default_factory=list)
    conversational_style: str | None = None


class OnboardingDraft(OnboardingModel):
    identity: OnboardingIdentity = Field(default_factory=OnboardingIdentity)
    reviewer: OnboardingReviewer = Field(default_factory=OnboardingReviewer)
    contributors: list[OnboardingContributor] = Field(default_factory=list)
    sources: list[OnboardingSource] = Field(default_factory=list)
    people: list[OnboardingPerson] = Field(default_factory=list)
    facts: list[OnboardingFact] = Field(default_factory=list)
    memories: list[OnboardingMemory] = Field(default_factory=list)
    sensitive_topics: list[SensitiveTopicCreate] = Field(default_factory=list)
    preferences: OnboardingPreferences = Field(
        default_factory=OnboardingPreferences
    )

    @model_validator(mode="after")
    def _unique_local_keys(self) -> "OnboardingDraft":
        for label, items in (
            ("sources", self.sources),
            ("people", self.people),
        ):
            keys = [item.key for item in items]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{label} keys must be unique")
            if "subject" in keys:
                raise ValueError(
                    f"'{label}' cannot use the reserved key 'subject'"
                )
        return self


class OnboardingConsentRequest(OnboardingModel):
    confirmed_by: str = Field(min_length=1, max_length=255)
    scope: str = Field(min_length=1)


class OnboardingSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: OnboardingStatus
    payload: OnboardingDraft
    consent_confirmed: bool
    consent_confirmed_by: str | None
    consent_confirmed_at: datetime | None
    consent_scope: str | None
    subject_person_id: uuid.UUID | None
    profile_id: uuid.UUID | None
    activated_at: datetime | None
    created_at: datetime
    updated_at: datetime
