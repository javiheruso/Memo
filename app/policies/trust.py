"""Trust-model policy rules enforced in code (not just prompt instructions).

These pure functions encode the dossier's non-negotiable rules that apply to
the retrieval/context layer: no invention, no promotion of unverified data,
and no silent collapse of ambiguity. They keep the epistemology explicit so a
generative responder can never merge a memory into a verified fact.
"""

from app.domain.enums import (
    EpistemicKind,
    FactStatus,
    RelationshipStatus,
    SensitivityLevel,
)
from app.models import Memory

_SENSITIVITY_RANK = {
    SensitivityLevel.low: 0,
    SensitivityLevel.medium: 1,
    SensitivityLevel.high: 2,
}


def epistemic_kind_for_fact(status: FactStatus) -> EpistemicKind:
    if status == FactStatus.verified:
        return EpistemicKind.verified_fact
    return EpistemicKind.probable_fact


def epistemic_kind_for_relationship(status: RelationshipStatus) -> EpistemicKind:
    if status == RelationshipStatus.verified:
        return EpistemicKind.verified_fact
    return EpistemicKind.probable_fact


def filter_sensitive_memories(
    memories: list[Memory],
    max_level: SensitivityLevel = SensitivityLevel.low,
) -> tuple[list[Memory], list[Memory]]:
    """Split memories into safe-to-surface and filtered-out items.

    Memories above ``max_level`` are excluded from the response context rather
    than surfaced by a generic "who is" answer.
    """
    threshold = _SENSITIVITY_RANK[max_level]
    safe: list[Memory] = []
    filtered: list[Memory] = []
    for memory in memories:
        rank = _SENSITIVITY_RANK[memory.sensitivity_level]
        (safe if rank <= threshold else filtered).append(memory)
    return safe, filtered
