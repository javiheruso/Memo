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

# A topic already present in the current message may be echoed or discussed,
# but never converted into a new suggestion. These markers are language-level
# offer cues, not topic-specific.
_OFFER_OR_QUESTION = re.compile(
    r"\b(?:podemos|puedo|quieres|quiere|te\s+gustar[íi]a(?:n)?|"
    r"te\s+apetece(?:n)?|te\s+interesa(?:n)?|si\s+quieres|te\s+cuento|"
    r"hablamos\s+de|hablemos\s+de)\b",
    re.IGNORECASE,
)
_OPEN_TOPIC_REQUEST = re.compile(
    r"\b(?:prop[oó]n(?:me)?|dime\s+t[uú]|algo\s+interesante|"
    r"de\s+qu[eé]\s+podemos\s+hablar|de\s+qu[eé]\s+hablamos)\b",
    re.IGNORECASE,
)
# A first-person disclosure is a turn where the person is talking about their
# own life or feelings. Offering a profile topic there is tone-deaf even when
# the topic itself is benign.
_PERSONAL_DISCLOSURE = re.compile(
    r"\b(?:me|mi|mis|estoy|tengo|siento|creo|necesito|prefiero|he|"
    r"preocupad[oa]|preocupa\w*)\b",
    re.IGNORECASE,
)


def _is_personal_disclosure(text: str) -> bool:
    return _PERSONAL_DISCLOSURE.search(text) is not None


def is_open_topic_request(text: str) -> bool:
    """True when the person asks the assistant to choose a topic."""
    return _OPEN_TOPIC_REQUEST.search(text) is not None


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
    if _is_personal_disclosure(current):
        # The person is sharing their own news or feelings; no topic offer.
        return False
    if mentioned_topics(current, topics):
        # The person's current topic already wins; never propose it back.
        return False
    return not any(
        mentioned_topics(prior, topics) for prior in prior_turn_texts
    )


def _is_offer_shape(text: str) -> bool:
    """True only for an explicit offer, not for every question.

    A content question about the topic the person chose ("¿Has estado en la
    playa?") is conversation. "¿Te apetece hablar de la playa?" is an offer
    and must still be rejected when offers are not allowed.
    """
    return _OFFER_OR_QUESTION.search(text) is not None


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
    - when suggestions are not allowed, an offer-shaped answer that proposes a
      new topic is a violation; a plain acknowledgement, statement or content
      question is discussion, not an offer;
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
        # preference must not become an offer. Stored topics remain grounded
        # factual mentions. A content question about the person's chosen topic
        # is discussion; only an explicit offer puts a topic back to them.
        return _is_offer_shape(answer) and bool(proposed - grounded)
    if surfaced and unexpected:
        return True
    return len(unexpected) > 1
