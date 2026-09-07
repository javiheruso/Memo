"""Embedder wiring through settings.

Mirror of ``app/llm/provider.py``: the pipeline receives its ``Embedder`` from
this provider so swapping in a real embeddings endpoint means registering it in
``EMBEDDERS`` and selecting it via ``MEMO_EMBEDDER``. A provider may expose a
``from_settings`` classmethod to receive its configuration; otherwise it is
instantiated with no arguments.
"""

from app.core.config import get_settings
from app.embeddings.base import Embedder
from app.embeddings.deterministic import DeterministicEmbedder
from app.embeddings.openai_compatible import OpenAICompatibleEmbedder

EMBEDDERS: dict[str, type[Embedder]] = {
    "deterministic": DeterministicEmbedder,
    "openai_compatible": OpenAICompatibleEmbedder,
}


def get_embedder() -> Embedder:
    settings = get_settings()
    try:
        embedder_cls = EMBEDDERS[settings.embedder]
    except KeyError as exc:
        raise ValueError(
            f"Unknown embedder '{settings.embedder}'. "
            f"Known: {', '.join(sorted(EMBEDDERS))}."
        ) from exc
    factory = getattr(embedder_cls, "from_settings", None)
    if factory is not None:
        return factory(settings)
    return embedder_cls()
