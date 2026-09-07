import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import EventStatus


class EventPersonCreate(BaseModel):
    person_id: uuid.UUID
    role: str | None = Field(default=None, max_length=50)


class EventCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = None
    start_at: datetime
    end_at: datetime | None = None
    status: EventStatus = EventStatus.planned
    source_id: uuid.UUID | None = None
    people: list[EventPersonCreate] = Field(default_factory=list)


class EventPersonRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    person_id: uuid.UUID
    role: str | None


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    description: str | None
    start_at: datetime
    end_at: datetime | None
    status: EventStatus
    source_id: uuid.UUID | None
    created_at: datetime
    people: list[EventPersonRead] = Field(default_factory=list)
