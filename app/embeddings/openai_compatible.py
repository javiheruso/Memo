"""Real OpenAI-compatible embedder behind the provider registry, fail-closed.

Mirror of ``app/llm/openai_compatible.py``: calls any OpenAI-compatible
``/embeddings`` endpoint with the text, validates that the returned vector
matches the schema dimension, and on any invalid output or provider/network
problem falls back to the deterministic embedder for that call. A model or
endpoint mistake can therefore degrade a single embedding to the deterministic
placeholder but can never inject a wrong-dimension vector into a pgvector
column or crash the retrieval pipeline.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings
from app.embeddings.base import Embedder
from app.embeddings.deterministic import DeterministicEmbedder
from app.models.memory import EMBEDDING_DIMENSIONS


class InvalidEmbeddingOutput(ValueError):
    """Raised when an embeddings response cannot be turned into a trusted vector."""


class OpenAICompatibleEmbedder(Embedder):
    """Embedder backed by any OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        dimensions: int = EMBEDDING_DIMENSIONS,
        client: httpx.AsyncClient | None = None,
        fallback: Embedder | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required for the OpenAI-compatible embedder")
        if not model:
            raise ValueError("model is required for the OpenAI-compatible embedder")
        if dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._dimensions = dimensions
        self._client = client
        self._fallback = fallback or DeterministicEmbedder(dimensions=dimensions)

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAICompatibleEmbedder:
        return cls(
            base_url=settings.embedder_base_url,
            model=settings.embedder_model,
            api_key=settings.embedder_api_key,
            timeout_seconds=settings.embedder_timeout_seconds,
        )

    async def embed(self, text: str) -> list[float]:
        try:
            return await self._call_endpoint(text)
        except (httpx.HTTPError, InvalidEmbeddingOutput):
            # Fail closed: the deterministic embedder handles this call.
            return await self._fallback.embed(text)

    async def _call_endpoint(self, text: str) -> list[float]:
        url = f"{self._base_url}/embeddings"
        payload: dict[str, Any] = {"model": self._model, "input": text}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if self._client is not None:
            response = await self._client.post(url, headers=headers, json=payload)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()

        try:
            embedding = response.json()["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise InvalidEmbeddingOutput(
                "embedding response did not match the expected shape"
            ) from exc
        if not isinstance(embedding, list) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in embedding
        ):
            raise InvalidEmbeddingOutput("embedding is not a numeric vector")
        if len(embedding) != self._dimensions:
            raise InvalidEmbeddingOutput(
                f"embedding dimension {len(embedding)} != expected {self._dimensions}"
            )
        return [float(value) for value in embedding]
