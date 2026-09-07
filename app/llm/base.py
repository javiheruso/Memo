from dataclasses import dataclass, field
from typing import Protocol

from app.repositories.conversation import ResponseSourceRef
from app.retrieval.context_builder import AnswerContext


@dataclass(frozen=True)
class Response:
    text: str
    sources: list[ResponseSourceRef] = field(default_factory=list)


class Responder(Protocol):
    """Replaceable response generator.

    Today the only implementation is the deterministic text responder. A real
    LLM provider can be swapped in without touching the pipeline: it receives
    the same epistemically-tagged ``AnswerContext`` and reports exactly which
    entities it used so ``response_sources`` stays truthful.
    """

    async def respond(self, context: AnswerContext) -> Response: ...
