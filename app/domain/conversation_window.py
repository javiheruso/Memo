"""Read-only, bounded snapshot of one conversation's persisted turns.

The window is a pure contract: it holds no ORM objects, performs no queries,
and never mutates state. The conversation repository loads a window once per
turn, and the service derives its narrow continuity rules (pending ambiguity,
fact subject, resolved person) from that same snapshot.

Turns and their response sources are already ordered by the repository, so the
contract preserves the deterministic ``(created_at, id)`` order the database
returned. Ordering is oldest to newest for turns, and insertion order for the
sources the assistant presented.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from app.domain.enums import Intent

_FACT_SUBJECT_INTENTS = frozenset({Intent.fact_query.value})
_RESOLVED_PERSON_INTENTS = frozenset(
    {Intent.person_query.value, Intent.fact_query.value}
)


@dataclass(frozen=True)
class SourceSnapshot:
    """One response source attached to an assistant turn."""

    entity_type: str
    entity_id: uuid.UUID
    usage_type: str | None = None


@dataclass(frozen=True)
class TurnSnapshot:
    """One persisted message with its ordered response sources."""

    message_id: uuid.UUID
    speaker: str
    text: str
    intent: str | None
    created_at: datetime
    sources: tuple[SourceSnapshot, ...] = ()


@dataclass(frozen=True)
class ConversationWindow:
    """A bounded, ordered slice of the current conversation.

    ``turns`` is oldest to newest. Requesting the window from the repository
    will return at most the configured limit of the most recent messages.
    """

    conversation_id: uuid.UUID
    turns: tuple[TurnSnapshot, ...] = field(default_factory=tuple)

    @property
    def latest_assistant_turn(self) -> TurnSnapshot | None:
        """Return the most recent assistant turn, or ``None``."""
        for turn in reversed(self.turns):
            if turn.speaker == "assistant":
                return turn
        return None

    def pending_person_candidates(self) -> tuple[uuid.UUID, ...]:
        """Return the person candidates offered by the latest assistant turn.

        Candidates keep the order the assistant presented them in. An empty
        result means there is no pending ambiguity to resolve.
        """
        turn = self.latest_assistant_turn
        if turn is None:
            return ()
        return tuple(
            source.entity_id
            for source in turn.sources
            if source.entity_type == "person" and source.usage_type == "candidate"
        )

    def last_fact_subject(self) -> uuid.UUID | None:
        """Return the subject of the immediately previous fact answer.

        Continuity stays deliberately narrow: the latest assistant turn must
        be a resolved ``fact_query`` answer with exactly one ``subject`` person
        source and no ``candidate`` rows.
        """
        return self._subject_person(_FACT_SUBJECT_INTENTS)

    def last_resolved_person(self) -> uuid.UUID | None:
        """Return the person of the immediately previous resolved answer.

        A person or fact answer can establish the subject for a possessive
        follow-up, but an unrelated turn or an unresolved ambiguity never does.
        """
        return self._subject_person(_RESOLVED_PERSON_INTENTS)

    def _subject_person(self, intents: frozenset[str]) -> uuid.UUID | None:
        turn = self.latest_assistant_turn
        if turn is None or turn.intent not in intents:
            return None
        person_sources = [
            source for source in turn.sources if source.entity_type == "person"
        ]
        if any(source.usage_type == "candidate" for source in person_sources):
            return None
        subjects = {
            source.entity_id
            for source in person_sources
            if source.usage_type == "subject"
        }
        if len(subjects) != 1:
            return None
        return next(iter(subjects))
