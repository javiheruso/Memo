"""Bounded, style-only handling of profile positive topics.

Positive topics are conversational preferences, not stored facts. This module
keeps them from becoming a checklist a responder repeats:

- a suggestion is possible only on a spontaneous conversational turn;
- it is allowed at most once per conversation (any earlier user or assistant
  turn that already surfaced a topic disables later suggestions);
- the person's current message always wins, so a topic they already mentioned
  is never proposed back to them;
- a responder that mentions a topic when it was not allowed is rejected.

Matching reuses the deterministic topic matcher from the sensitive-topic
policy. No topic string is special-cased here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from app.domain.enums import Intent
from app.policies.sensitive_topics import SensitiveTopic, matches_topic
from app.retrieval.context_builder import AnswerContext

_SPONTANEOUS_INTENTS = frozenset({Intent.general_chat, Intent.unknown})

# A topic already present in the current message may be echoed as a statement,
# but never converted into a new suggestion. These markers are language-level
# (an offer or a question), not topic-specific.
_OFFER_OR_QUESTION = re.compile(
    r"\b(?:podemos|puedo|quieres|quiere|te\s+gustar[íi]a|te\s+apetece|"
    r"te\s+interesa|si\s+quieres|te\s+cuento|hablamos\s+de)\b",
    re.IGNORECASE,
)


def mentioned_topics(text: str, topics: Iterable[str]) -> tuple[str, ...]:
    """Return the configured topics explicitly present in ``text``."""
    return tuple(
        topic
        for topic in topics
        if topic.strip() and matches_topic(text, SensitiveTopic(topic=topic))
    )


def context_topic_grounding(context: AnswerContext) -> str:
    """Text of stored items an answer may truthfully paraphrase.

    A positive-topic word that appears here is grounded data, not an
    unsolicited suggestion. It is still never a source by itself; this only
    lets factual answers mention the word without being mistaken for an offer.
    """
    parts: list[str] = []
    if context.person is not None:
        parts.append(context.person.display_name)
    for item in (*context.verified_relations, *context.probable_relations):
        parts.append(item.other_display_name)
    for item in context.facts:
        parts.append(item.text)
    for item in context.memories:
        parts.append(f"{item.title or ''} {item.narrative}")
    for item in context.events:
        parts.append(item.title)
    for item in context.media:
        parts.append(
            f"{item.title or ''} {item.description or ''} {item.place_text or ''}"
        )
    return " ".join(parts)


def may_offer_positive_topics(
    *,
    topics: Sequence[str],
    message: str,
    prior_turn_texts: Iterable[str],
    intent: Intent,
) -> bool:
    """Whether this turn may carry one gentle positive-topic suggestion."""
    if not topics or intent not in _SPONTANEOUS_INTENTS:
        return False
    current = message.strip()
    if not current:
        return False
    if mentioned_topics(current, topics):
        # The person's current topic already wins; never propose it back.
        return False
    return not any(
        mentioned_topics(prior, topics) for prior in prior_turn_texts
    )


def _is_offer_or_question(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith(("?", "¿")) or bool(
        _OFFER_OR_QUESTION.search(stripped)
    )


def positive_topic_violation(
    *,
    answer: str,
    message: str,
    topics: Sequence[str],
    offer_allowed: bool,
    grounded_text: str = "",
) -> bool:
    """True when an answer breaks the bounded positive-topic policy.

    A topic present in the current message wins, and a topic present in stored
    context is a factual mention, not an offer. Otherwise:
    - when suggestions are not allowed, introducing a new topic is a violation;
    - when they are allowed, at most one new topic may be suggested, and never
      alongside a topic the person already chose.
    """
    if not topics:
        return False
    surfaced = set(mentioned_topics(message, topics))
    grounded = set(mentioned_topics(grounded_text, topics))
    proposed = set(mentioned_topics(answer, topics))
    unexpected = proposed - surfaced - grounded
    if not offer_allowed:
        # After an earlier suggestion, a rejection, or a topic change, a
        # preference must not become an offer. A grounded factual mention is
        # still allowed. The person's current topic may be echoed as a plain
        # statement, but a question/offer that puts it back to them is not.
        echoable = set() if _is_offer_or_question(answer) else surfaced
        return bool(proposed - grounded - echoable)
    if surfaced and unexpected:
        return True
    return len(unexpected) > 1
