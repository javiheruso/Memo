"""Trust-model policy rules enforced in code (not just prompt instructions).

These pure functions encode the dossier's non-negotiable rules that apply to
the retrieval/context layer: no invention, no promotion of unverified data,
and no silent collapse of ambiguity. They keep the epistemology explicit so a
generative responder can never merge a memory into a verified fact.

Only ``verified`` and ``probable`` facts and relationships are answerable.
``rejected`` rows are excluded at the repository boundary, and ``pending``
rows are withheld from answer contexts until a human reviews them. Neither
state is ever re-labelled as ``probable``.
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

ANSWERABLE_FACT_STATUS_VALUES = frozenset(
    {FactStatus.verified.value, FactStatus.probable.value}
)
ANSWERABLE_RELATIONSHIP_STATUS_VALUES = frozenset(
    {RelationshipStatus.verified.value, RelationshipStatus.probable.value}
)


def is_answerable_fact_status(status: FactStatus | str) -> bool:
    """True only for fact states that may enter an answer.

    ``pending`` is not answerable yet and ``rejected`` is never answerable, so
    both are withheld instead of being softened into a probable statement.
    """
    value = status.value if isinstance(status, FactStatus) else status
    return value in ANSWERABLE_FACT_STATUS_VALUES


def is_answerable_relationship_status(
    status: RelationshipStatus | str,
) -> bool:
    """True only for relationship states that may enter an answer."""
    value = status.value if isinstance(status, RelationshipStatus) else status
    return value in ANSWERABLE_RELATIONSHIP_STATUS_VALUES


def epistemic_kind_for_fact(status: FactStatus) -> EpistemicKind:
    if status == FactStatus.verified:
        return EpistemicKind.verified_fact
    if status == FactStatus.probable:
        return EpistemicKind.probable_fact
    raise ValueError(
        f"Fact status '{status.value}' is not answerable; "
        "pending and rejected facts must be withheld"
    )


def epistemic_kind_for_relationship(status: RelationshipStatus) -> EpistemicKind:
    if status == RelationshipStatus.verified:
        return EpistemicKind.verified_fact
    if status == RelationshipStatus.probable:
        return EpistemicKind.probable_fact
    raise ValueError(
        f"Relationship status '{status.value}' is not answerable; "
        "pending and rejected relationships must be withheld"
    )


def sensitivity_levels_at_or_below(
    max_level: SensitivityLevel = SensitivityLevel.low,
) -> tuple[SensitivityLevel, ...]:
    """Return the levels allowed by ``max_level``, for query-side filtering.

    Retrieval must filter sensitivity *before* applying its final limit;
    otherwise protected rows consume slots and the caller receives fewer
    allowed memories than requested. SQL engines can use this tuple directly.
    """
    threshold = _SENSITIVITY_RANK[max_level]
    return tuple(
        level for level, rank in _SENSITIVITY_RANK.items() if rank <= threshold
    )


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
