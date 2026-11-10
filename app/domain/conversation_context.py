"""Bounded, untrusted conversational context handed to a responder.

This module is a pure contract: it carries at most the recent turns of one
conversation plus the small style facts from the active profile. It never
carries entity ids, response sources, embeddings, audio, or any stored record,
so a responder cannot turn a conversation turn into a fact or a source.

The service filters sensitive turns and bounds the snapshot before it reaches
any responder; ``bound_conversation_history`` is the single place that enforces
the turn/character limits, keeping provider prompts small and predictable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_RESPONDER_HISTORY_TURNS = 8
DEFAULT_RESPONDER_MAX_HISTORY_CHARS = 2000


@dataclass(frozen=True)
class ConversationTurn:
    """One already-filtered prior turn. Only a role and text are carried."""

    role: str
    text: str


@dataclass(frozen=True)
class ConversationStyle:
    """Small, verified profile facts used to address the person naturally.

    ``conversational_style`` is the operator-provided style note, not model
    output. It is rendered as guidance and can never introduce facts or sources.
    """

    preferred_name: str | None = None
    conversational_style: str | None = None
    positive_topics: tuple[str, ...] = ()


def bound_conversation_history(
    history: Sequence[ConversationTurn],
    *,
    max_turns: int = DEFAULT_RESPONDER_HISTORY_TURNS,
    max_chars: int = DEFAULT_RESPONDER_MAX_HISTORY_CHARS,
) -> tuple[ConversationTurn, ...]:
    """Return the newest bounded slice of ``history`` (oldest to newest).

    The most recent turns win and the total text length never exceeds
    ``max_chars``. Invalid roles, blank text, and non-positive limits yield an
    empty or shorter snapshot instead of an error.
    """
    if max_turns <= 0 or max_chars <= 0:
        return ()

    selected: list[ConversationTurn] = []
    remaining = max_chars
    for turn in reversed(tuple(history)):
        if turn.role not in ("user", "assistant"):
            continue
        text = turn.text.strip()
        if not text:
            continue
        snippet = text[:remaining]
        if not snippet:
            break
        selected.append(ConversationTurn(role=turn.role, text=snippet))
        remaining -= len(snippet)
        if remaining <= 0 or len(selected) >= max_turns:
            break
    selected.reverse()
    return tuple(selected)


__all__ = [
    "DEFAULT_RESPONDER_HISTORY_TURNS",
    "DEFAULT_RESPONDER_MAX_HISTORY_CHARS",
    "ConversationStyle",
    "ConversationTurn",
    "bound_conversation_history",
]
