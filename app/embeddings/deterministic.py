"""Deterministic embedder used by default and in the test suite.

No real embeddings provider exists yet. This implementation is stable across
processes and Python versions (it hashes with ``blake2b``, not the randomized
built-in ``hash``) and produces L2-normalized bag-of-token vectors sized to the
schema's vector dimension (``EMBEDDING_DIMENSIONS``), so memories that share
words with the query rank higher. Single-character tokens are dropped so
function words like "a" do not create spurious matches. It is a deterministic
placeholder for real semantic embeddings, not a semantic model.
"""

import hashlib
import math
import re

from app.embeddings.base import Embedder
from app.models.memory import EMBEDDING_DIMENSIONS

_TOKEN = re.compile(r"[a-záéíóúñü0-9]+")


class DeterministicEmbedder(Embedder):
    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in _TOKEN.findall(text.lower()):
            if len(token) < 2:
                continue
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
            vector[int(digest, 16) % self._dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]
