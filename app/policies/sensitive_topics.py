"""Deterministic sensitive-topic policy for the live conversation path.

The conversational profile declares topics that Memo must never initiate on
its own. This module answers three questions in pure code:

1. Did the person explicitly raise a configured topic in this turn?
2. Which context items belong to a blocked topic and must be withheld from a
   spontaneous turn?
3. Did a responder still mention a blocked topic in its output?

Matching is intentionally conservative and deterministic: a topic matches when
its full phrase appears, when all of its significant words appear, or when a
theme trigger (configured or from a small built-in family lexicon) appears.
The lexicon only expands well-known sensitive themes such as death or illness;
it never relaxes the existing trust and sensitivity rules, which run first.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from app.retrieval.context_builder import AnswerContext

# Generic words that should never make a topic match on their own.
_STOPWORDS = {
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "y",
    "o",
    "en",
    "con",
    "por",
    "para",
    "que",
    "al",
    "mi",
    "mis",
    "tu",
    "tus",
    "su",
    "sus",
    "lo",
    "se",
    "es",
    "son",
    "fue",
    "era",
    "como",
    "mas",
    "pero",
    "sin",
    "sobre",
    "entre",
    "cuando",
    "cuanto",
    "quien",
    "the",
    "of",
    "and",
    "to",
    "in",
    "on",
    "with",
    "about",
    "when",
    "who",
    "what",
    "my",
    "is",
    "was",
}

# Conservative trigger expansion for common sensitive themes. The keys are
# folded topic words; values are folded trigger words. Unknown topics still
# match by their full phrase or all significant words.
_TOPIC_FAMILIES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("fallecimiento", "muerte", "murio", "fallecio", "funeral", "entierro"),
        (
            "muerte",
            "murio",
            "fallecimiento",
            "fallecio",
            "funeral",
            "entierro",
            "enterrado",
            "difunto",
        ),
    ),
    (
        ("enfermedad", "enfermo", "enferma", "hospital", "diagnostico"),
        (
            "enfermedad",
            "enfermo",
            "enferma",
            "hospital",
            "hospitalizado",
            "diagnostico",
        ),
    ),
    (
        ("divorcio", "separacion", "separo"),
        ("divorcio", "separacion", "separo"),
    ),
    (
        ("deuda", "deudas", "dinero", "hipoteca"),
        ("deuda", "deudas", "dinero", "hipoteca"),
    ),
)

_TOKEN = re.compile(r"[a-z0-9]+")

SPONTANEOUS_SUPPRESSION_REPLY = (
    "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."
)


def _fold(value: str) -> str:
    """Lowercase and strip accents so matching is spelling-insensitive."""
    normalized = unicodedata.normalize("NFKD", value).lower()
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _words(value: str) -> set[str]:
    return set(_TOKEN.findall(_fold(value)))


def _significant_words(value: str) -> set[str]:
    return {
        word
        for word in _words(value)
        if len(word) > 2 and word not in _STOPWORDS
    }


def _family_triggers(topic: str) -> set[str]:
    folded = _fold(topic)
    triggers: set[str] = set()
    for roots, family_triggers in _TOPIC_FAMILIES:
        if any(root in folded for root in roots):
            triggers.update(family_triggers)
    return triggers


@dataclass(frozen=True)
class SensitiveTopic:
    topic: str
    handling: str | None = None
    triggers: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensitiveTopicEvaluation:
    requested: SensitiveTopic | None
    blocked: tuple[SensitiveTopic, ...]

    @property
    def has_topics(self) -> bool:
        return self.requested is not None or bool(self.blocked)


def matches_topic(text: str, topic: SensitiveTopic) -> bool:
    """True when ``text`` explicitly references ``topic``."""
    folded = _fold(text)
    phrase = _fold(topic.topic).strip()
    if phrase and phrase in folded:
        return True

    topic_words = _significant_words(topic.topic)
    if topic_words and topic_words.issubset(_words(text)):
        return True

    triggers = {_fold(item).strip() for item in topic.triggers if item.strip()}
    triggers.update(_family_triggers(topic.topic))
    words = _words(text)
    for trigger in triggers:
        if not trigger:
            continue
        if " " in trigger:
            if trigger in folded:
                return True
        elif trigger in words:
            return True
    return False


def evaluate(text: str, topics: tuple[SensitiveTopic, ...]) -> SensitiveTopicEvaluation:
    """Classify a user turn against the configured sensitive topics.

    The first explicitly referenced topic is considered requested; every other
    configured topic remains blocked for that turn.
    """
    requested_index: int | None = None
    for index, topic in enumerate(topics):
        if matches_topic(text, topic):
            requested_index = index
            break

    requested = topics[requested_index] if requested_index is not None else None
    blocked = tuple(
        topic for index, topic in enumerate(topics) if index != requested_index
    )
    return SensitiveTopicEvaluation(requested=requested, blocked=blocked)


def _mentions_blocked_topic(text: str, evaluation: SensitiveTopicEvaluation) -> bool:
    return any(matches_topic(text, topic) for topic in evaluation.blocked)


def apply_to_context(
    context: AnswerContext,
    evaluation: SensitiveTopicEvaluation,
    *,
    positive_topics: tuple[str, ...] = (),
) -> AnswerContext:
    """Withhold blocked-topic content and attach the policy directives.

    Existing high-sensitivity filtering and epistemic tags are preserved: this
    function only removes items that belong to a blocked configured topic and
    adds explicit policy lines for the responder.
    """
    if not evaluation.has_topics:
        return context

    memories = [
        item
        for item in context.memories
        if not _mentions_blocked_topic(
            f"{item.title or ''} {item.narrative}", evaluation
        )
    ]
    facts = [
        item
        for item in context.facts
        if not _mentions_blocked_topic(item.text, evaluation)
    ]
    events = [
        item
        for item in context.events
        if not _mentions_blocked_topic(item.title, evaluation)
    ]
    media = [
        item
        for item in context.media
        if not _mentions_blocked_topic(
            f"{item.title or ''} {item.description or ''}", evaluation
        )
    ]

    policies = list(context.policies)
    withheld = (
        (len(context.memories) - len(memories))
        + (len(context.facts) - len(facts))
        + (len(context.events) - len(events))
        + (len(context.media) - len(media))
    )
    if withheld:
        policies.append(
            "sensitive_topic_not_initiated: "
            f"{withheld} item(s) matching configured sensitive topics withheld"
        )
    if evaluation.blocked:
        policies.append(
            "sensitive_topic_not_initiated: "
            "never raise these topics unless the person raises them first: "
            + "; ".join(topic.topic for topic in evaluation.blocked)
        )

    redirect_topic = next(
        (topic for topic in positive_topics if topic.strip()), None
    )
    if evaluation.requested is not None:
        policies.append(
            "sensitive_topic_requested: "
            f"'{evaluation.requested.topic}' was raised explicitly; do not "
            "invent details and gently redirect to a safer subject"
        )

    return replace(
        context,
        memories=memories,
        facts=facts,
        events=events,
        media=media,
        policies=policies,
        sensitive_topic_requested=(
            evaluation.requested.topic if evaluation.requested is not None else None
        ),
        sensitive_redirect_topic=redirect_topic,
        blocked_sensitive_topics=tuple(topic.topic for topic in evaluation.blocked),
    )


def answer_leaks_blocked_topic(
    text: str, evaluation: SensitiveTopicEvaluation
) -> bool:
    """True when responder output mentions a topic that must not be initiated."""
    return _mentions_blocked_topic(text, evaluation)
