import math

import httpx
import pytest

from app.embeddings.deterministic import DeterministicEmbedder
from app.embeddings.openai_compatible import OpenAICompatibleEmbedder
from app.models.memory import EMBEDDING_DIMENSIONS


def _client_for(
    embedding: list[float] | None = None,
    *,
    status: int = 200,
    error: Exception | None = None,
    capture: list | None = None,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        if error is not None:
            raise error
        body = {"data": [{"embedding": embedding or [0.0] * EMBEDDING_DIMENSIONS}]}
        return httpx.Response(status, json=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _embedder(client: httpx.AsyncClient, **kwargs) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        base_url=kwargs.pop("base_url", "http://embed.test"),
        model=kwargs.pop("model", "test-embed"),
        api_key=kwargs.pop("api_key", "secret"),
        client=client,
        **kwargs,
    )


def _zero_vector() -> list[float]:
    return [0.0] * EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_valid_embedding_returns_the_provider_vector() -> None:
    vector = [0.1] * EMBEDDING_DIMENSIONS
    async with _client_for(vector) as client:
        result = await _embedder(client).embed("hola")
    assert result == vector
    assert len(result) == EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_request_targets_embeddings_with_model_and_auth() -> None:
    import json

    capture: list[httpx.Request] = []
    async with _client_for(capture=capture) as client:
        await _embedder(client).embed("¿Te acuerdas de bailar?")
    request = capture[0]
    assert request.url == "http://embed.test/embeddings"
    body = json.loads(request.content)
    assert body["model"] == "test-embed"
    assert body["input"] == "¿Te acuerdas de bailar?"
    assert request.headers["Authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_wrong_dimension_falls_back_to_deterministic() -> None:
    async with _client_for([0.25, 0.75]) as client:
        result = await _embedder(client).embed("hola")
    assert len(result) == EMBEDDING_DIMENSIONS
    norm = math.sqrt(sum(value * value for value in result))
    assert norm == pytest.approx(1.0)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_malformed_response_falls_back_to_deterministic() -> None:
    async with _client_for(["not", "numbers"]) as client:
        result = await _embedder(client).embed("hola")
    assert len(result) == EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_empty_data_falls_back_to_deterministic() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _embedder(client).embed("hola")
    assert len(result) == EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_http_error_falls_back_to_deterministic() -> None:
    async with _client_for(status=500) as client:
        result = await _embedder(client).embed("hola")
    assert len(result) == EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_network_error_falls_back_to_deterministic() -> None:
    async with _client_for(error=httpx.ConnectError("connection refused")) as client:
        result = await _embedder(client).embed("hola")
    assert len(result) == EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_timeout_falls_back_to_deterministic() -> None:
    async with _client_for(error=httpx.TimeoutException("request timed out")) as client:
        result = await _embedder(client).embed("hola")
    assert len(result) == EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_fallback_matches_deterministic_for_same_text() -> None:
    deterministic = await DeterministicEmbedder().embed("hola")
    async with _client_for(error=httpx.ConnectError("refused")) as client:
        result = await _embedder(client).embed("hola")
    assert result == deterministic


@pytest.mark.asyncio
async def test_fallback_dimension_is_validated() -> None:
    async with _client_for(error=httpx.ConnectError("refused")) as client:
        embedder = OpenAICompatibleEmbedder(
            base_url="http://embed.test",
            model="m",
            dimensions=64,
            client=client,
            fallback=DeterministicEmbedder(dimensions=64),
        )
        result = await embedder.embed("hola")
    assert len(result) == 64


def test_base_url_model_and_dimensions_are_required() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleEmbedder(base_url="", model="m")
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleEmbedder(base_url="http://embed.test", model="")
    with pytest.raises(ValueError, match="dimensions"):
        OpenAICompatibleEmbedder(base_url="http://embed.test", model="m", dimensions=0)


def test_from_settings_maps_config() -> None:
    from app.core.config import Settings

    embedder = OpenAICompatibleEmbedder.from_settings(
        Settings(
            embedder="openai_compatible",
            embedder_base_url="http://local",
            embedder_model="local-embed",
            embedder_api_key=None,
            embedder_timeout_seconds=5,
        )
    )
    assert embedder._base_url == "http://local"
    assert embedder._model == "local-embed"
    assert embedder._api_key is None
    assert embedder._timeout == 5
    assert embedder._dimensions == EMBEDDING_DIMENSIONS
