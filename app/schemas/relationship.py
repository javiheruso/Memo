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


class RelationshipStatusUpdate(BaseModel):
    """Change only a relationship's trust status.

    Extra fields are rejected on purpose: provenance (``source_id``), validity
    dates, relationship type and endpoints are immutable through this update.
    The status enum is validated by Pydantic, so an unknown value fails before
    any database write.
    """

    model_config = ConfigDict(extra="forbid")

    status: RelationshipStatus


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
