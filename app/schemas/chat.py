import uuid

from pydantic import BaseModel, Field

from app.domain.enums import Intent


class ChatRequest(BaseModel):
    conversation_id: uuid.UUID | None = None
    message: str = Field(min_length=1, max_length=2000)


class SourceRef(BaseModel):
    entity_type: str
    entity_id: uuid.UUID
    usage_type: str | None = None


class CandidateRef(BaseModel):
    id: uuid.UUID
    display_name: str


class AmbiguityInfo(BaseModel):
    term: str
    candidates: list[CandidateRef]


class GapRef(BaseModel):
    id: uuid.UUID
    gap_type: str
    canonical_reference: str
    status: str


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    intent: Intent
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    ambiguity: AmbiguityInfo | None = None
    gap: GapRef | None = None
