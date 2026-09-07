"""Claim grouping seam.

Baseline grouping is deliberately conservative: two candidates share a group
only when their normalized claim text is identical (case, accents, punctuation
and whitespace folded). Paraphrases with the same meaning but different wording
do NOT merge here — merging them is semantic grouping, which belongs to a later
phase and would be implemented behind this same ``ClaimGrouper`` seam, selected
through ``MEMO_CLAIM_GROUPER`` (see ``app/claims/provider.py``).
"""

import re
import unicodedata
from typing import Protocol


class ClaimGrouper(Protocol):
    """Replaceable claim-to-group mapping.

    Implementations return a stable group key for a claim text; claims whose
    keys are equal land in the same group. A semantic implementation can be
    registered in ``app/claims/provider.py`` and swapped in without touching the
    pipeline.
    """

    def group_key(self, text: str) -> str: ...


def normalize_claim_text(text: str) -> str:
    """Fold a claim to a comparison key: lowercase, accent-stripped, tokens only.

    Punctuation and whitespace collapse to single spaces between tokens, so
    "Vivíamos en Madrid." and "viviamos en madrid" map to the same key while
    different wording stays different.
    """
    folded = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in folded if unicodedata.category(char) != "Mn")
    tokens = re.findall(r"[^\W_]+", stripped.lower(), flags=re.UNICODE)
    return " ".join(tokens)


class DeterministicClaimGrouper:
    """Exact normalized-text grouping: conservative, deterministic, no network."""

    def group_key(self, text: str) -> str:
        return normalize_claim_text(text)
