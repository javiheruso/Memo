"""Embedding provider seam.

The retrieval pipeline receives its ``Embedder`` from
``app/embeddings/provider`` (mirroring the responder wiring) so neither the
endpoint nor ``ConversationService`` hardcodes a concrete implementation. A
real provider (for example an OpenAI-compatible embeddings endpoint) can be
swapped in later without touching routing or retrieval.
"""

import math
from typing import Protocol


class Embedder(Protocol):
    """Replaceable text-to-vector embedder sized to the schema dimension."""

    async def embed(self, text: str) -> list[float]: ...


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns a value in ``[0.0, 1.0]`` for the non-negative vectors produced by
    the deterministic embedder; a zero vector scores ``0.0`` against anything.
    """
    if len(a) != len(b):
        raise ValueError(
            f"vector dimension mismatch: {len(a)} != {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
