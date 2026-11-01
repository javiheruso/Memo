from enum import Enum


class FactStatus(str, Enum):
    verified = "verified"
    probable = "probable"
    pending = "pending"
    rejected = "rejected"


class MemoryStatus(str, Enum):
    candidate = "candidate"
    recurring = "recurring"
    consolidated = "consolidated"
    archived = "archived"


class EventStatus(str, Enum):
    planned = "planned"
    confirmed = "confirmed"
    cancelled = "cancelled"
    completed = "completed"


class KnowledgeGapStatus(str, Enum):
    observed = "observed"
    recurrent = "recurrent"
    family_review = "family_review"
    resolved = "resolved"
    ignored = "ignored"


class ClaimStatus(str, Enum):
    candidate = "candidate"
    rejected = "rejected"


class ClaimGroupStatus(str, Enum):
    """Manual review state of a claim group.

    ``active`` is the default: the group is part of the review queue and
    contradiction detection may lift ``needs_review`` on it. ``resolved`` and
    ``ignored`` are terminal, human-set states that take the group out of the
    queue: detection never re-flags them. No auto-promotion exists here.
    """

    active = "active"
    resolved = "resolved"
    ignored = "ignored"


class RelationshipStatus(str, Enum):
    verified = "verified"
    probable = "probable"
    pending = "pending"
    rejected = "rejected"


class RelationshipType(str, Enum):
    parent_of = "parent_of"
    child_of = "child_of"
    grandparent_of = "grandparent_of"
    grandchild_of = "grandchild_of"
    sibling_of = "sibling_of"
    spouse_of = "spouse_of"
    partner_of = "partner_of"
    friend_of = "friend_of"
    other = "other"


class OnboardingStatus(str, Enum):
    """Lifecycle of an admin-led onboarding draft.

    ``draft`` is resumable and mutable. ``activated`` is terminal: the
    workflow has created the real person, facts, memories and conversational
    profile, and further draft edits are refused.
    """

    draft = "draft"
    activated = "activated"


class SourceType(str, Enum):
    family_member = "family_member"
    user = "user"
    conversation = "conversation"
    document = "document"
    photo = "photo"
    calendar = "calendar"
    system = "system"
    other = "other"


class SensitivityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class MediaType(str, Enum):
    """Kind of stored media.

    The ``media.media_type`` column is a plain string, so this enum only
    standardizes the values the admin surface accepts. No upload or serving
    exists in this slice: rows carry metadata only.
    """

    photo = "photo"
    video = "video"
    audio = "audio"
    document = "document"
    other = "other"


class MaterializationSourceType(str, Enum):
    """What kind of reviewed evidence a materialization came from."""

    claim_group = "claim_group"
    knowledge_gap = "knowledge_gap"


class MaterializationTargetType(str, Enum):
    """What kind of entity an explicit materialization produced."""

    fact = "fact"
    memory = "memory"


class Intent(str, Enum):
    general_chat = "general_chat"
    person_query = "person_query"
    memory_query = "memory_query"
    fact_query = "fact_query"
    event_query = "event_query"
    orientation_query = "orientation_query"
    unknown = "unknown"


class EpistemicKind(str, Enum):
    verified_fact = "verified_fact"
    probable_fact = "probable_fact"
    memory = "memory"
    claim = "claim"
    unknown = "unknown"
    policy = "policy"


RELATIONSHIP_INVERSE: dict[RelationshipType, RelationshipType] = {
    RelationshipType.parent_of: RelationshipType.child_of,
    RelationshipType.child_of: RelationshipType.parent_of,
    RelationshipType.grandparent_of: RelationshipType.grandchild_of,
    RelationshipType.grandchild_of: RelationshipType.grandparent_of,
    RelationshipType.sibling_of: RelationshipType.sibling_of,
    RelationshipType.spouse_of: RelationshipType.spouse_of,
    RelationshipType.partner_of: RelationshipType.partner_of,
    RelationshipType.friend_of: RelationshipType.friend_of,
    RelationshipType.other: RelationshipType.other,
}


def invert_relationship_type(relationship_type: RelationshipType) -> RelationshipType:
    """Return the inverse of a directed relationship type.

    Used to phrase a relationship from the perspective of either endpoint.
    Symmetric types map to themselves.
    """
    return RELATIONSHIP_INVERSE.get(relationship_type, relationship_type)
