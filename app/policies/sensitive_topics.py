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


def _phrase_or_words_match(text: str, topic: SensitiveTopic) -> bool:
    folded = _fold(text)
    phrase = _fold(topic.topic).strip()
    if phrase and phrase in folded:
        return True

    topic_words = _significant_words(topic.topic)
    return bool(topic_words and topic_words.issubset(_words(text)))


def _trigger_matches(text: str, topic: SensitiveTopic) -> bool:
    folded = _fold(text)
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


def _topic_identity_words(topic: SensitiveTopic) -> set[str]:
    """Proper-name tokens that identify a person named in a topic label.

    ``la muerte de Antonio`` and ``Antonio`` carry an identity anchor;
    ``deudas de la familia`` and ``la enfermedad`` do not. Capitalization is
    read from the configured label, and theme vocabulary is excluded so a
    sentence-initial capital ("Muerte de Antonio") is not mistaken for a name.
    """
    raw_tokens = re.findall(r"[^\W\d_]+", topic.topic)
    if not raw_tokens:
        return set()

    non_names = set(_STOPWORDS)
    non_names.update(_fold(item) for item in topic.triggers if item.strip())
    for roots, triggers in _TOPIC_FAMILIES:
        non_names.update(roots)
        non_names.update(triggers)

    anchors: set[str] = set()
    for token in raw_tokens:
        if not token[:1].isupper():
            continue
        folded = _fold(token)
        if folded and folded not in non_names:
            anchors.add(folded)
    return anchors


def _matches_requested_topic(text: str, topic: SensitiveTopic) -> bool:
    """True only when the turn actually references this specific topic.

    A person-specific topic (one whose label carries a proper name) is not
    requested by generic sensitive vocabulary alone. ``Rosa estuvo en el
    hospital`` must not request ``la muerte de Antonio``; ``Antonio estuvo en
    el hospital`` and the explicit label still do.
    """
    if _phrase_or_words_match(text, topic):
        return True

    anchors = _topic_identity_words(topic)
    if anchors and not anchors.issubset(_words(text)):
        return False
    return _trigger_matches(text, topic)


@dataclass(frozen=True)
class SensitiveTopic:
    topic: str
    handling: str | None = None
    triggers: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensitiveTopicEvaluation:
    requested: SensitiveTopic | None
    blocked: tuple[SensitiveTopic, ...]
    # True when only generic sensitive vocabulary matched a person-specific
    # configured topic. In that case ``requested`` is a neutral label, never
    # the configured personal topic.
    generic_reference: bool = False

    @property
    def has_topics(self) -> bool:
        return self.requested is not None or bool(self.blocked)


# Neutral stand-in used when the person raised generic sensitive vocabulary
# without identifying the configured person. It keeps the careful response and
# blocking behavior without exposing the configured personal label.
_GENERIC_SENSITIVE_TOPIC = "un tema sensible"


def matches_topic(text: str, topic: SensitiveTopic) -> bool:
    """True when ``text`` references ``topic`` in any form.

    This broad check is used for leak detection and context withholding. The
    narrower :func:`_matches_requested_topic` decides whether a person-specific
    topic was actually raised by the person.
    """
    return _phrase_or_words_match(text, topic) or _trigger_matches(text, topic)


def evaluate(text: str, topics: tuple[SensitiveTopic, ...]) -> SensitiveTopicEvaluation:
    """Classify a user turn against the configured sensitive topics.

    The first explicitly referenced topic is considered requested; every other
    configured topic remains blocked for that turn. Generic sensitive vocabulary
    that does not identify the person named by a configured topic is treated as
    a neutral sensitive reference: it never becomes ``requested`` under that
    personal label, but the topic stays blocked.
    """
    requested_index: int | None = None
    for index, topic in enumerate(topics):
        if _matches_requested_topic(text, topic):
            requested_index = index
            break

    requested = topics[requested_index] if requested_index is not None else None
    generic_reference = False
    if requested is None and any(matches_topic(text, topic) for topic in topics):
        requested = SensitiveTopic(topic=_GENERIC_SENSITIVE_TOPIC)
        generic_reference = True

    blocked = tuple(
        topic for index, topic in enumerate(topics) if index != requested_index
    )
    return SensitiveTopicEvaluation(
        requested=requested,
        blocked=blocked,
        generic_reference=generic_reference,
    )


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
        sensitive_topic_generic=evaluation.generic_reference,
        sensitive_redirect_topic=redirect_topic,
        blocked_sensitive_topics=tuple(topic.topic for topic in evaluation.blocked),
    )


def answer_leaks_blocked_topic(
    text: str, evaluation: SensitiveTopicEvaluation
) -> bool:
    """True when responder output mentions a topic that must not be initiated."""
    return _mentions_blocked_topic(text, evaluation)
