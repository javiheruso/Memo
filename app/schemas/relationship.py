import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from app.domain.enums import RelationshipStatus, RelationshipType


class RelationshipCreate(BaseModel):
    person_a_id: uuid.UUID
    person_b_id: uuid.UUID
    relationship_type: RelationshipType
    status: RelationshipStatus = RelationshipStatus.verified
    source_id: uuid.UUID | None = None
    valid_from: date | None = None
    valid_to: date | None = None


class RelationshipRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    person_a_id: uuid.UUID
    person_b_id: uuid.UUID
    relationship_type: RelationshipType
    status: RelationshipStatus
    source_id: uuid.UUID | None
    valid_from: date | None
    valid_to: date | None
    created_at: datetime
