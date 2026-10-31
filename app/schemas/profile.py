"""Request/response schemas for conversational profiles."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SensitiveTopicCreate(BaseModel):
    topic: str = Field(min_length=1, max_length=255)
    handling: str | None = None
    consent_review_required: bool = False
    triggers: list[str] = Field(default_factory=list)


class SensitiveTopicRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    topic: str
    handling: str | None
    consent_review_required: bool
    triggers: list[str]


class ConversationProfileCreate(BaseModel):
    person_id: uuid.UUID
    preferred_name: str | None = Field(default=None, max_length=255)
    language: str | None = Field(default=None, max_length=50)
    conversational_style: str | None = None
    positive_topics: list[str] = Field(default_factory=list)
    sensitive_topics: list[SensitiveTopicCreate] = Field(default_factory=list)
    beliefs_policy: str | None = None
    never_do: list[str] = Field(default_factory=list)
    orientation: str | None = None
    data_origin: str | None = None
    source_id: uuid.UUID | None = None
    is_fictional: bool = False
    is_active: bool = False

    @model_validator(mode="after")
    def _reject_duplicate_sensitive_topics(self) -> "ConversationProfileCreate":
        topics = [item.topic.strip().lower() for item in self.sensitive_topics]
        if len(topics) != len(set(topics)):
            raise ValueError("sensitive_topics must not contain duplicates")
        return self


class ConversationProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    person_id: uuid.UUID
    preferred_name: str | None
    language: str | None
    conversational_style: str | None
    positive_topics: list[str]
    sensitive_topics: list[SensitiveTopicRead]
    beliefs_policy: str | None
    never_do: list[str]
    orientation: str | None
    data_origin: str | None
    source_id: uuid.UUID | None
    is_fictional: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime
