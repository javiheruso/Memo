import uuid

from pydantic import BaseModel, Field

from app.domain.enums import Intent
from app.schemas.chat import AmbiguityInfo, GapRef, SourceRef


class VoiceResponse(BaseModel):
    conversation_id: uuid.UUID
    intent: Intent
    transcript: str | None = None
    text: str | None = None
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    audio: str | None = None
    audio_content_type: str | None = None
    ambiguity: AmbiguityInfo | None = None
    gap: GapRef | None = None
