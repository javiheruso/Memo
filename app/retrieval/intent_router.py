import re
from dataclasses import dataclass

from app.domain.enums import Intent
from app.retrieval.fact_topics import FactTopic, classify_fact_question

_PERSON_QUESTION = re.compile(
    r"(?:quién|quien|who)\s+(?:es|era|fue|is|was)\s+([^?¿]+?)\s*[?¿]*$",
    re.IGNORECASE,
)
# "¿Te acuerdas de cuando íbamos a bailar?" / "Do you remember when we went
# dancing?" -> semantic memory retrieval. Checked before event markers so a
# recollection phrasing about a person or event ("¿Te acuerdas de cuando venía
# María?") stays a memory query.
_MEMORY_MARKERS = re.compile(
    r"\b(?:te\s+acuerdas\b|recuerdas\b|do\s+you\s+remember\b|remember\s+when\b)",
    re.IGNORECASE,
)
_EVENT_MARKERS = re.compile(
    r"\b(viene|vendrá|vienen|vendrán|cuándo|evento|eventos|quedada)\b",
    re.IGNORECASE,
)
_NAME = r"[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]*"
_EVENT_PERSON = re.compile(
    rf"(?:viene|vienen|vendrá|vendrán)\s+(?P<arrival>{_NAME})"
    rf"|(?:evento|eventos|quedada)\b[^?¿]*?\b(?:tiene|hay|habrá|tendrá|es|son)\s+(?P<owner>{_NAME})",
    re.IGNORECASE,
)
# Orientation: "¿Qué día es hoy?" / "¿Qué hora es?" / "What's the date?".
# End-anchored so a generic day/time phrase never swallows an event question
# ("¿Qué día viene María?" is routed as an event by the earlier event check).
# Checked after event markers on purpose: event phrasings take precedence.
_ORIENTATION_DAY = re.compile(
    r"(?:qué día de la semana es|qué día es hoy|en qué día estamos|qué día es|"
    r"what day is it today|what day of the week is it|what day is it)"
    r"\s*[?¿]*$",
    re.IGNORECASE,
)
_ORIENTATION_DATE = re.compile(
    r"(?:qué fecha es hoy|a qué fecha estamos|qué fecha es|"
    r"what's the date|what is the date|whats the date|what date is it)"
    r"\s*[?¿]*$",
    re.IGNORECASE,
)
_ORIENTATION_TIME = re.compile(
    r"(?:qué hora es|qué horas son|what time is it)\s*[?¿]*$",
    re.IGNORECASE,
)
_ORIENTATION_YEAR = re.compile(
    r"(?:en qué año estamos|qué año es|what year is it)\s*[?¿]*$",
    re.IGNORECASE,
)
_ORIENTATION_PATTERNS = (
    _ORIENTATION_DAY,
    _ORIENTATION_DATE,
    _ORIENTATION_TIME,
    _ORIENTATION_YEAR,
)

# Fact/preference questions reuse the existing person resolver, so the router
# only has to recognize a question shape, a topic family, and a proper-name
# subject. A declarative statement ("Me gusta Elena") never matches because a
# question marker is required.
_FACT_QUESTION = re.compile(
    r"^\s*[¿]?\s*(?:qué|que|cuál|cuáles|cual|cuales|dónde|donde|cómo|como|"
    r"what|which|where|how)\b|[?¿]",
    re.IGNORECASE,
)
_NAME_PHRASE = rf"{_NAME}(?:\s+{_NAME})*"
_FACT_SUBJECT_PREPOSITION = re.compile(
    rf"\b(?i:a|de|para|sobre)\s+(?P<name>{_NAME_PHRASE})"
)
_FACT_SUBJECT_VERB = re.compile(
    rf"\b(?i:tiene|tienen|prefiere|prefieren|naci[oó]|vivi[oó]|vive|"
    rf"trabaj[oó]|trabaja|dedicaba|dedica|opina|cree|piensa)\s+"
    rf"(?P<name>{_NAME_PHRASE})"
)
_FACT_SUBJECT_POSSESSIVE = re.compile(rf"(?P<name>{_NAME_PHRASE})'s\b")
_FACT_SUBJECT_ENGLISH_AUX = re.compile(
    rf"\b(?i:does|did|was|is|are)\s+(?P<name>{_NAME_PHRASE})"
)
_FACT_SUBJECT_PATTERNS = (
    _FACT_SUBJECT_PREPOSITION,
    _FACT_SUBJECT_VERB,
    _FACT_SUBJECT_POSSESSIVE,
    _FACT_SUBJECT_ENGLISH_AUX,
)


def _extract_fact_subject(text: str) -> str | None:
    for pattern in _FACT_SUBJECT_PATTERNS:
        match = pattern.search(text)
        if match:
            term = match.group("name").strip()
            if term:
                return term
    return None


@dataclass(frozen=True)
class RouteResult:
    intent: Intent
    term: str | None = None
    fact_topic: FactTopic | None = None


class IntentRouter:
    """Minimal deterministic intent classifier.

    Kept to a small dossier-aligned vocabulary: ``person_query``,
    ``memory_query``, ``fact_query``, ``event_query``, ``orientation_query``,
    ``general_chat`` and ``unknown``. ``general_chat`` is the catch-all for
    normal conversational text so everyday topics never touch personal memory.

    Precedence: ``person_query`` > ``memory_query`` > ``fact_query`` >
    ``event_query`` > ``orientation_query`` > ``general_chat``. Orientation
    patterns are end-anchored so "¿Qué día viene María?" stays an event query.
    Fact questions are checked before event markers so "¿Cuándo nació Elena?"
    asks for a stored fact instead of an event.
    """

    def route(self, text: str) -> RouteResult:
        normalized = text.strip()
        if not normalized:
            return RouteResult(intent=Intent.unknown)

        match = _PERSON_QUESTION.search(normalized)
        if match:
            term = match.group(1).strip()
            if term:
                return RouteResult(intent=Intent.person_query, term=term)

        if _MEMORY_MARKERS.search(normalized):
            return RouteResult(intent=Intent.memory_query)

        fact_topic = classify_fact_question(normalized)
        if fact_topic is not None and _FACT_QUESTION.search(normalized):
            term = _extract_fact_subject(normalized)
            if term:
                return RouteResult(
                    intent=Intent.fact_query, term=term, fact_topic=fact_topic
                )

        if _EVENT_MARKERS.search(normalized):
            term = None
            event_match = _EVENT_PERSON.search(normalized)
            if event_match:
                term = event_match.group("arrival") or event_match.group("owner")
            if term:
                return RouteResult(intent=Intent.event_query, term=term.strip())
            return RouteResult(intent=Intent.event_query)

        for pattern in _ORIENTATION_PATTERNS:
            if pattern.search(normalized):
                return RouteResult(intent=Intent.orientation_query)

        return RouteResult(intent=Intent.general_chat)
