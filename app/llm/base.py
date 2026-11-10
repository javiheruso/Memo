from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.domain.conversation_context import ConversationStyle, ConversationTurn
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
    the same epistemically-tagged ``AnswerContext`` plus a bounded, untrusted
    recent-conversation snapshot and the verified profile style. The snapshot
    is context for continuity only: it is never a fact and never a source, and
    the responder reports exactly which context entities it used so
    ``response_sources`` stays truthful.
    """

    async def respond(
        self,
        context: AnswerContext,
        *,
        message: str | None = None,
        history: Sequence[ConversationTurn] = (),
        style: ConversationStyle | None = None,
    ) -> Response: ...
