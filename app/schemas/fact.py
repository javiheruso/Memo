import uuid
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import FactStatus


class FactCreate(BaseModel):
    subject_person_id: uuid.UUID
    fact_type: str = Field(max_length=50)
    value_text: str | None = None
    value_date: date | None = None
    value_number: Decimal | None = None
    related_person_id: uuid.UUID | None = None
    place_text: str | None = Field(default=None, max_length=500)
    valid_from: date | None = None
    valid_to: date | None = None
    status: FactStatus = FactStatus.pending
    source_id: uuid.UUID
    notes: str | None = None


class FactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    subject_person_id: uuid.UUID
    fact_type: str
    value_text: str | None
    value_date: date | None
    value_number: Decimal | None
    related_person_id: uuid.UUID | None
    place_text: str | None
    valid_from: date | None
    valid_to: date | None
    status: FactStatus
    source_id: uuid.UUID
    notes: str | None
    created_at: datetime
