"""Claim-vs-fact contradiction detector seam.

Conservative deterministic baseline: a claim group contradicts a verified fact
only when the claim carries an explicit negation marker AND every content token
of the fact surface appears in the claim. No semantics and no canonicalization
are involved, so differently-worded contradictions (paraphrases, implicit
opposites, differently conjugated verbs) are missed on purpose — false
negatives are preferred over noise, mirroring the claim extractor philosophy.

A semantic detector can be registered in ``app/claims/provider.py`` behind the
same ``ContradictionDetector`` seam and swapped in via
``MEMO_CLAIM_CONTRADICTION_DETECTOR`` without touching the pipeline.
"""

from dataclasses import dataclass
from typing import Protocol

from app.claims.grouping import normalize_claim_text

_NEGATION_MARKERS = frozenset({"no", "nunca", "jamas", "tampoco"})
_NEGATION_PHRASES = ("ya no", "ni siquiera")

# High-frequency framing words that carry no meaning about the fact's content.
# Removed from both sides before the overlap check so that "recuerdo que no
# vivíamos en Madrid" still matches a fact surface about "vivimos en Madrid".
_STOPWORDS = frozenset(
    {
        "a",
        "al",
        "con",
        "cuando",
        "como",
        "de",
        "del",
        "donde",
        "e",
        "el",
        "en",
        "es",
        "era",
        "fue",
        "habia",
        "hay",
        "la",
        "las",
        "le",
        "lo",
        "los",
        "me",
        "mas",
        "mi",
        "mis",
        "muy",
        "nos",
        "o",
        "para",
        "pero",
        "por",
        "que",
        "quien",
        "recuerdo",
        "recuerda",
        "acuerdo",
        "acordaba",
        "se",
        "ser",
        "su",
        "sus",
        "te",
        "tu",
        "un",
        "una",
        "unos",
        "unas",
        "y",
        "ya",
    }
)


@dataclass(frozen=True)
class ContradictionSignal:
    """A conservative contradiction signal between a claim and a fact surface.

    ``matched_terms`` lists the fact content tokens the claim repeats, which is
    what the deterministic detector found suspicious when combined with the
    claim's negation marker.
    """

    matched_terms: tuple[str, ...]


class ContradictionDetector(Protocol):
    """Replaceable claim-vs-fact contradiction detector.

    Implementations compare a claim group's canonical text against the textual
    surface of a single verified fact and return a signal when they consider
    the claim to contradict the fact. Returning ``None`` means "no signal".
    """

    name: str

    def detect(self, claim_text: str, fact_text: str) -> ContradictionSignal | None: ...


def fact_surface_text(*, value_text: str | None = None, place_text: str | None = None) -> str:
    """Assemble the comparable textual surface of a fact.

    Only the fact's own narrative fields are used; structured fields (dates,
    numbers, related people) are excluded because they carry no surface wording
    a Spanish claim could repeat.
    """
    return " ".join(part for part in (value_text, place_text) if part).strip()


class DeterministicContradictionDetector:
    """Rules-only negation + content-overlap detector: conservative, no network."""

    name = "deterministic"

    def detect(self, claim_text: str, fact_text: str) -> ContradictionSignal | None:
        if not self._has_negation(claim_text):
            return None
        fact_tokens = self._content_tokens(fact_text)
        if not fact_tokens:
            return None
        claim_tokens = self._content_tokens(claim_text)
        matched = tuple(sorted(token for token in fact_tokens if token in claim_tokens))
        if len(matched) != len(fact_tokens):
            return None
        return ContradictionSignal(matched_terms=matched)

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {token for token in normalize_claim_text(text).split() if token not in _STOPWORDS}

    @staticmethod
    def _has_negation(text: str) -> bool:
        normalized = normalize_claim_text(text)
        tokens = set(normalized.split())
        if any(marker in tokens for marker in _NEGATION_MARKERS):
            return True
        return any(phrase in normalized for phrase in _NEGATION_PHRASES)
