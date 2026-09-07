import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from app.domain.enums import KnowledgeGapStatus


class KnowledgeGapRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    gap_type: str
    canonical_reference: str
    description: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    occurrence_count: int
    distinct_sessions: int
    status: KnowledgeGapStatus
    resolved_person_id: uuid.UUID | None
    resolved_fact_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class KnowledgeGapStatusUpdate(BaseModel):
    """Manual status transition for a knowledge gap.

    Uses the existing ``knowledge_gap_status`` model. Resolution may optionally
    link the gap to the person or fact that closed it; that linkage is only
    meaningful when the gap is being resolved.
    """

    status: KnowledgeGapStatus
    resolved_person_id: uuid.UUID | None = None
    resolved_fact_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def _resolution_requires_reference(self) -> "KnowledgeGapStatusUpdate":
        if (
            self.status is not KnowledgeGapStatus.resolved
            and (self.resolved_person_id is not None or self.resolved_fact_id is not None)
        ):
            raise ValueError(
                "resolved_person_id/resolved_fact_id require status 'resolved'"
            )
        return self
