import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import MemoryStatus, SensitivityLevel


class MemoryPersonCreate(BaseModel):
    person_id: uuid.UUID
    role: str | None = Field(default=None, max_length=50)


class MemoryCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    narrative: str = Field(min_length=1)
    approx_start_date: date | None = None
    approx_end_date: date | None = None
    approx_period_text: str | None = Field(default=None, max_length=255)
    place_text: str | None = Field(default=None, max_length=500)
    memory_status: MemoryStatus = MemoryStatus.candidate
    sensitivity_level: SensitivityLevel = SensitivityLevel.low
    source_id: uuid.UUID
    people: list[MemoryPersonCreate] = Field(default_factory=list)


class MemoryPersonRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    person_id: uuid.UUID
    role: str | None


class MemoryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    narrative: str
    approx_start_date: date | None
    approx_end_date: date | None
    approx_period_text: str | None
    place_text: str | None
    memory_status: MemoryStatus
    sensitivity_level: SensitivityLevel
    source_id: uuid.UUID
    created_at: datetime
    people: list[MemoryPersonRead] = Field(default_factory=list)


class MemoryBackfillRead(BaseModel):
    processed: int
