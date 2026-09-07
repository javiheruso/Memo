import math

import pytest

from app.embeddings.base import cosine_similarity
from app.embeddings.deterministic import DeterministicEmbedder
from app.embeddings.provider import EMBEDDERS, get_embedder
from app.models.memory import EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_embedder_dimension_matches_schema():
    vector = await DeterministicEmbedder().embed("Hola")
    assert len(vector) == EMBEDDING_DIMENSIONS


@pytest.mark.asyncio
async def test_embedder_is_deterministic_across_calls():
    embedder = DeterministicEmbedder()
    assert await embedder.embed("Fishing trips with Antonio") == await embedder.embed(
        "Fishing trips with Antonio"
    )


@pytest.mark.asyncio
async def test_embedder_is_l2_normalized():
    vector = await DeterministicEmbedder().embed("una frase cualquiera")
    norm = math.sqrt(sum(value * value for value in vector))
    assert norm == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_shared_tokens_rank_higher():
    embedder = DeterministicEmbedder()
    query = await embedder.embed("Do you remember when we went dancing?")
    dancing = await embedder.embed(
        "Carmen and Antonio went dancing at the local hall"
    )
    fishing = await embedder.embed(
        "Carmen remembers that Paco and Antonio went fishing"
    )
    assert cosine_similarity(query, dancing) > cosine_similarity(query, fishing)
    assert cosine_similarity(query, dancing) > 0.0


def test_cosine_similarity_zero_for_zero_vector():
    assert cosine_similarity([0.0, 0.0], [0.5, 0.5]) == 0.0


def test_cosine_similarity_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([0.1, 0.2], [0.1])


def test_provider_defaults_to_deterministic():
    assert "deterministic" in EMBEDDERS
    assert isinstance(get_embedder(), DeterministicEmbedder)


def test_provider_rejects_unknown_embedder(monkeypatch):
    class _FakeSettings:
        embedder = "nonexistent"

    monkeypatch.setattr(
        "app.embeddings.provider.get_settings", lambda: _FakeSettings()
    )
    with pytest.raises(ValueError, match="Unknown embedder 'nonexistent'"):
        get_embedder()


def test_provider_registers_openai_compatible():
    from app.embeddings.openai_compatible import OpenAICompatibleEmbedder

    assert "openai_compatible" in EMBEDDERS
    assert EMBEDDERS["openai_compatible"] is OpenAICompatibleEmbedder


def test_provider_selects_openai_compatible_from_settings(monkeypatch):
    from app.embeddings.openai_compatible import OpenAICompatibleEmbedder

    class _FakeSettings:
        embedder = "openai_compatible"
        embedder_base_url = "http://local"
        embedder_model = "local-embed"
        embedder_api_key = "k"
        embedder_timeout_seconds = 5.0

    monkeypatch.setattr(
        "app.embeddings.provider.get_settings", lambda: _FakeSettings()
    )
    embedder = get_embedder()
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    assert embedder._base_url == "http://local"
    assert embedder._model == "local-embed"
    assert embedder._api_key == "k"
    assert embedder._timeout == 5.0
