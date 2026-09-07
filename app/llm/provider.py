"""Responder wiring through settings.

The chat pipeline receives its ``Responder`` from this provider so neither the
endpoint nor ``ConversationService`` hardcodes a concrete implementation.
Swapping in a real LLM provider means registering it in ``RESPONDERS`` and
selecting it via ``MEMO_RESPONDER``. A responder may expose a
``from_settings`` classmethod to receive its configuration; otherwise it is
instantiated with no arguments.
"""

from app.core.config import get_settings
from app.llm.base import Responder
from app.llm.deterministic import DeterministicResponder
from app.llm.openai_compatible import OpenAICompatibleResponder

RESPONDERS: dict[str, type[Responder]] = {
    "deterministic": DeterministicResponder,
    "openai_compatible": OpenAICompatibleResponder,
}


def get_responder() -> Responder:
    settings = get_settings()
    try:
        responder_cls = RESPONDERS[settings.responder]
    except KeyError as exc:
        raise ValueError(
            f"Unknown responder '{settings.responder}'. "
            f"Known: {', '.join(sorted(RESPONDERS))}."
        ) from exc
    factory = getattr(responder_cls, "from_settings", None)
    if factory is not None:
        return factory(settings)
    return responder_cls()
