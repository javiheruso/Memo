import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import MediaType


class MediaPersonCreate(BaseModel):
    person_id: uuid.UUID
    is_primary: bool = False
    notes: str | None = Field(default=None, max_length=500)


class MediaCreate(BaseModel):
    """Metadata-only media input. No upload, no image bytes, no visual data.

    ``description`` is free-text caption metadata the family may have written;
    it is never treated as an authoritative description of the image content.
    """

    media_type: MediaType
    storage_path: str = Field(min_length=1, max_length=500)
    title: str | None = Field(default=None, max_length=255)
    description: str | None = None
    approx_date: date | None = None
    place_text: str | None = Field(default=None, max_length=500)
    source_id: uuid.UUID | None = None
    people: list[MediaPersonCreate] = Field(default_factory=list)


class MediaPersonRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    person_id: uuid.UUID
    is_primary: bool
    notes: str | None


class MediaRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    media_type: str
    storage_path: str
    title: str | None
    description: str | None
    approx_date: date | None
    place_text: str | None
    source_id: uuid.UUID | None
    created_at: datetime
    people: list[MediaPersonRead] = Field(default_factory=list)
