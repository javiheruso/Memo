"""Responder wiring through settings.

The chat pipeline receives its ``Responder`` from this provider so neither the
endpoint nor ``ConversationService`` hardcodes a concrete implementation.
Swapping in a real LLM provider means registering it in ``RESPONDERS`` and
selecting it via ``MEMO_RESPONDER``. A responder may expose a
``from_settings`` classmethod to receive its configuration; otherwise it is
instantiated with no arguments.
"""

import logging

from app.core.config import get_settings
from app.llm.base import Responder
from app.llm.deterministic import DeterministicResponder
from app.llm.interpreter import Interpreter, OpenAICompatibleInterpreter
from app.llm.openai_compatible import OpenAICompatibleResponder

logger = logging.getLogger(__name__)

RESPONDERS: dict[str, type[Responder]] = {
    "deterministic": DeterministicResponder,
    "openai_compatible": OpenAICompatibleResponder,
}

INTERPRETERS: dict[str, type[Interpreter]] = {
    "openai_compatible": OpenAICompatibleInterpreter,
}

_DISABLED_INTERPRETERS = {"", "none", "disabled"}


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


def get_interpreter() -> Interpreter | None:
    """Resolve the optional low-confidence interpreter, or ``None``.

    The feature is disabled by default. When enabled it reuses the configured
    LLM endpoint/key unless interpreter-specific values are set. A missing
    endpoint leaves the feature off instead of failing every chat request.
    """
    settings = get_settings()
    provider = (settings.interpreter or "").strip().lower()
    if provider in _DISABLED_INTERPRETERS:
        return None
    try:
        interpreter_cls = INTERPRETERS[provider]
    except KeyError as exc:
        raise ValueError(
            f"Unknown interpreter '{provider}'. "
            f"Known: {', '.join(sorted(INTERPRETERS))}."
        ) from exc

    base_url = (settings.interpreter_base_url or settings.llm_base_url).strip()
    model = (settings.interpreter_model or settings.llm_model).strip()
    if not base_url or not model:
        logger.warning(
            "interpreter is enabled but the endpoint or model is missing; "
            "staying disabled"
        )
        return None

    factory = getattr(interpreter_cls, "from_settings", None)
    if factory is not None:
        return factory(settings)
    return interpreter_cls()
