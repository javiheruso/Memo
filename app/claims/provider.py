"""Claim extractor, grouper and contradiction detector wiring through settings.

Mirror of ``app/llm/provider.py`` and ``app/embeddings/provider.py``: the
conversation pipeline receives its ``ClaimExtractor``, ``ClaimGrouper`` and
``ContradictionDetector`` from this provider, so a real extractor, a semantic
grouper or a semantic contradiction engine can be registered here and selected
via ``MEMO_CLAIM_EXTRACTOR`` / ``MEMO_CLAIM_GROUPER`` /
``MEMO_CLAIM_CONTRADICTION_DETECTOR`` without touching routing, retrieval or
the conversation service.
"""

from app.claims.base import ClaimExtractor
from app.claims.contradiction import (
    ContradictionDetector,
    DeterministicContradictionDetector,
)
from app.claims.deterministic import DeterministicClaimExtractor
from app.claims.grouping import ClaimGrouper, DeterministicClaimGrouper
from app.core.config import get_settings

EXTRACTORS: dict[str, type[ClaimExtractor]] = {
    "deterministic": DeterministicClaimExtractor,
}

GROUPERS: dict[str, type[ClaimGrouper]] = {
    "deterministic": DeterministicClaimGrouper,
}

CONTRADICTION_DETECTORS: dict[str, type[ContradictionDetector]] = {
    "deterministic": DeterministicContradictionDetector,
}


def get_claim_extractor() -> ClaimExtractor:
    settings = get_settings()
    try:
        extractor_cls = EXTRACTORS[settings.claim_extractor]
    except KeyError as exc:
        raise ValueError(
            f"Unknown claim extractor '{settings.claim_extractor}'. "
            f"Known: {', '.join(sorted(EXTRACTORS))}."
        ) from exc
    return extractor_cls()


def get_claim_grouper() -> ClaimGrouper:
    settings = get_settings()
    try:
        grouper_cls = GROUPERS[settings.claim_grouper]
    except KeyError as exc:
        raise ValueError(
            f"Unknown claim grouper '{settings.claim_grouper}'. "
            f"Known: {', '.join(sorted(GROUPERS))}."
        ) from exc
    return grouper_cls()


def get_claim_contradiction_detector() -> ContradictionDetector:
    settings = get_settings()
    try:
        detector_cls = CONTRADICTION_DETECTORS[settings.claim_contradiction_detector]
    except KeyError as exc:
        raise ValueError(
            f"Unknown claim contradiction detector "
            f"'{settings.claim_contradiction_detector}'. "
            f"Known: {', '.join(sorted(CONTRADICTION_DETECTORS))}."
        ) from exc
    return detector_cls()
