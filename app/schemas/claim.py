import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import ClaimGroupStatus, ClaimStatus, FactStatus


class ClaimRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_text: str
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    speaker_person_id: uuid.UUID | None
    first_seen_at: datetime
    status: ClaimStatus
    created_at: datetime
    claim_group_id: uuid.UUID | None


class ClaimMessageRead(BaseModel):
    """The conversational message a member claim was extracted from.

    Gives a reviewer the full utterance that produced the claim (it may contain
    several claims) plus who spoke it, without loading the whole conversation.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    text: str
    speaker: str
    created_at: datetime


class ClaimGroupClaimRead(ClaimRead):
    """A member claim in a group-detail response, with its source message."""

    message: ClaimMessageRead | None = None


class ClaimGroupRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    canonical_text: str
    total_mentions: int
    distinct_conversations: int
    distinct_days: int
    needs_review: bool
    status: ClaimGroupStatus
    resolved_at: datetime | None
    resolution_note: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    created_at: datetime
    updated_at: datetime


class ClaimGroupResolve(BaseModel):
    """Manual review-state transition for a claim group.

    ``active`` reopens the group; ``resolved`` and ``ignored`` take it out of
    the review queue so contradiction detection stops flagging it.
    """

    status: ClaimGroupStatus
    resolution_note: str | None = Field(default=None, max_length=500)


class ClaimGroupContradictionFactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    fact_type: str
    value_text: str | None
    place_text: str | None
    status: FactStatus


class ClaimGroupContradictionGroupRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    canonical_text: str
    needs_review: bool
    status: ClaimGroupStatus


class ClaimGroupContradictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    claim_group_id: uuid.UUID
    fact_id: uuid.UUID
    detector_name: str
    matched_terms: str | None
    first_detected_at: datetime
    last_detected_at: datetime
    detection_count: int
    created_at: datetime
    updated_at: datetime
    group: ClaimGroupContradictionGroupRead
    fact: ClaimGroupContradictionFactRead


class ClaimGroupDetail(ClaimGroupRead):
    claims: list[ClaimGroupClaimRead]
    contradictions: list[ClaimGroupContradictionRead]


class ClaimGroupMergeSuggestionRead(BaseModel):
    """A deterministic merge target shown to the reviewer."""

    model_config = ConfigDict(from_attributes=True)

    group_id: uuid.UUID
    canonical_text: str
    total_mentions: int
    needs_review: bool
    shared_terms: int
    opposite_polarity: bool


class ClaimGroupMergeRequest(BaseModel):
    """Human-instructed canonical merge of one claim group into a survivor.

    ``target_group_id`` is the surviving group that absorbs ``group_id``. The
    opposite-polarity flag must be set explicitly when folding a negated wording
    into a non-negated survivor (or vice versa); the service refuses those
    merges otherwise.
    """

    target_group_id: uuid.UUID
    confirm_opposite_polarity: bool = False
