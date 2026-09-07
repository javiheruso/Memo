import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import SourceType


class SourceCreate(BaseModel):
    source_type: SourceType
    person_id: uuid.UUID | None = None
    description: str | None = None
    external_reference: str | None = Field(default=None, max_length=500)


class SourceRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_type: SourceType
    person_id: uuid.UUID | None
    description: str | None
    external_reference: str | None
    created_at: datetime
