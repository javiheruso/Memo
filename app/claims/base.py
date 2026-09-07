from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ClaimCandidate:
    """A candidate claim extracted from a single user message.

    ``original_text`` is the verbatim, append-only source sentence. The
    canonical triplet stays ``None`` in the ingestion baseline: canonicalization
    and grouping are deferred to the claim-grouping phase (dossier doc 04).
    """

    original_text: str
    canonical_subject: str | None = None
    canonical_predicate: str | None = None
    canonical_object: str | None = None


class ClaimExtractor(Protocol):
    """Replaceable candidate-claim extractor.

    Implementations must be conservative: they may miss candidates, but should
    never turn trivial or query content into claims. A provider failure must
    never affect the conversation pipeline, which catches extraction errors
    before persisting anything.
    """

    def extract(self, text: str) -> list[ClaimCandidate]: ...
