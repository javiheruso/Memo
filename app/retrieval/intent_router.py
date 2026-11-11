import re
from dataclasses import dataclass

from app.domain.enums import Intent
from app.retrieval.fact_topics import FactTopic, classify_fact_question
from app.retrieval.relationship_roles import (
    RelationshipRoleQuery,
    parse_relationship_role_query,
)

_NAME = r"[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ-]*"
_NAME_PHRASE = rf"{_NAME}(?:\s+{_NAME})*"

_PERSON_QUESTION = re.compile(
    r"(?:quién|quien|who)\s+(?:es|era|fue|is|was)\s+([^?¿]+?)\s*[?¿]*$",
    re.IGNORECASE,
)
# Natural Spanish ways to ask about a person: "Háblame de Elena",
# "Cuéntame sobre Miguel", "Dime algo de Elena". A proper-name subject is
# required (capitalized, like `_NAME_PHRASE`) and the phrase must end after the
# name, so generic requests ("Cuéntame un chiste", "Háblame de la vida") stay
# general chat. The trailing courtesy clause is optional: "Háblame de Elena,
# por favor".
_PERSON_REQUEST_SUFFIX = (
    rf"(?P<name>(?-i:{_NAME_PHRASE}))\s*(?:,\s*por\s+favor)?\s*[?.!¿¡]*$"
)
_PERSON_ABOUT_REQUEST = re.compile(
    rf"^\s*[¿¡]?\s*(?:h[áa]blame|cu[ée]ntame|dime)\s+(?:algo\s+)?"
    rf"(?:de|sobre|acerca\s+de)\s+{_PERSON_REQUEST_SUFFIX}",
    re.IGNORECASE,
)
_PERSON_DESCRIBE_REQUEST = re.compile(
    rf"^\s*[¿¡]?\s*descr[íi]beme\s+(?:a\s+)?{_PERSON_REQUEST_SUFFIX}",
    re.IGNORECASE,
)
_PERSON_POLITE_REQUEST = re.compile(
    rf"^\s*[¿¡]?\s*(?:me\s+)?(?:puedes|podr[íi]as|puede|podr[íi]a)\s+"
    rf"(?:hablar(?:me)?|contar(?:me)?|decir(?:me)?)\s+(?:algo\s+)?"
    rf"(?:de|sobre|acerca\s+de)\s+{_PERSON_REQUEST_SUFFIX}",
    re.IGNORECASE,
)
_PERSON_REQUEST_PATTERNS = (
    _PERSON_ABOUT_REQUEST,
    _PERSON_DESCRIBE_REQUEST,
    _PERSON_POLITE_REQUEST,
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
# An elliptical follow-up carries the topic but not the subject: "¿Y de
# comer?", "¿Y dónde vivió?". The continuation marker is required so a bare
# question about the assistant ("¿Qué música te gusta?") is never attached to
# the previous fact subject.
_FACT_FOLLOW_UP = re.compile(r"^\s*[¿]?\s*(?:y|and)\b", re.IGNORECASE)
# A possessive follow-up keeps the previous person but names a topic through
# "su"/"sus" ("Háblame sobre su infancia"). It must be a request/question shape
# and mention a recognizable fact topic, so unrelated possessive phrases stay
# general chat. The subject is never extracted here; the caller supplies the
# immediately previous resolved person.
_POSSESSIVE_FOLLOW_UP = re.compile(
    r"^\s*[¿¡]?\s*(?:"
    r"(?:h[áa]blame|cu[ée]ntame|dime)\s+(?:algo\s+)?(?:de|sobre|acerca\s+de)"
    r"|(?:qu[ée]|que)\s+(?:sabes|conoces|tienes|hay)\s+(?:de|sobre|acerca\s+de)"
    r"|(?:c[óo]mo|como)\s+(?:era|fue)"
    r")\s+(?:sus|su)\s+(?P<topic>[^?.!¿¡]+?)\s*[?.!¿¡]*$",
    re.IGNORECASE,
)
# Natural Spanish labels used when a carried-topic follow-up has no matching
# data. They are noun phrases so "No sé nada de <label> de <person> todavía"
# reads naturally without exposing storage or system internals.
FOLLOW_UP_TOPIC_LABELS: dict[FactTopic, str] = {
    FactTopic.preference: "los gustos",
    FactTopic.dislike: "los disgustos",
    FactTopic.hobby: "las aficiones",
    FactTopic.food: "la comida",
    FactTopic.drink: "la bebida",
    FactTopic.music: "la música",
    FactTopic.childhood: "la infancia",
    FactTopic.place: "la residencia",
    FactTopic.occupation: "el trabajo",
    FactTopic.habit: "las costumbres",
    FactTopic.belief: "las opiniones",
    FactTopic.value: "los valores",
    FactTopic.conversation: "la forma de hablar",
    FactTopic.family: "la familia",
    FactTopic.general: "eso",
}
# Pronoun follow-ups keep the previous resolved person without naming it:
# "¿Y ella?", "¿Qué sabes de él?", "Háblame de ese". The subject is never
# extracted here; the caller supplies the immediately previous resolved person
# and refuses when there is none (for example after an ambiguous answer).
_PRONOUN_REFERENCE = r"(?:él|ella|ellos|ellas|ese|esa)"
# A pronoun-only question ("¿Quién es él?", "¿Quién era ella?") is not a new
# person name. It must reach the verified pronoun carry-over path instead of
# being resolved as a person literally named "él"/"ella". Explicit names such as
# "¿Quién es Antonio?" never match this guarded shape.
_EXPLICIT_PRONOUN_QUESTION = re.compile(
    rf"^\s*[¿¡]?\s*(?:y\s+)?(?:qui[ée]n|quien)\s+"
    rf"(?:es|era|fue)\s+{_PRONOUN_REFERENCE}\s*[?.!¿¡]*$",
    re.IGNORECASE,
)
_PRONOUN_FOLLOW_UP_PATTERNS = (
    re.compile(
        rf"^\s*[¿¡]?\s*(?:y\s+)?{_PRONOUN_REFERENCE}\s*[?.!¿¡]*$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*[¿¡]?\s*(?:qu[ée]|que)\s+(?:sabes|conoces|tienes)\s+"
        rf"(?:de|sobre|acerca\s+de)\s+{_PRONOUN_REFERENCE}\s*[?.!¿¡]*$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*[¿¡]?\s*(?:h[áa]blame|cu[ée]ntame|dime)\s+(?:algo\s+)?"
        rf"(?:de|sobre|acerca\s+de)\s+{_PRONOUN_REFERENCE}\s*[?.!¿¡]*$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*[¿¡]?\s*(?:y\s+)?(?:qui[ée]n|quien)\s+(?:es|era|fue)\s+"
        rf"{_PRONOUN_REFERENCE}\s*[?.!¿¡]*$",
        re.IGNORECASE,
    ),
)
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


def is_explicit_pronoun_question(text: str) -> bool:
    """True for a Spanish pronoun-only person question.

    These questions name no person: the pronoun must be supplied from the
    immediately previous verified resolved subject, never from the literal
    pronoun text and never from the database as if it were a name.
    """
    return _EXPLICIT_PRONOUN_QUESTION.match(text.strip()) is not None


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
    follow_up_topic: str | None = None
    relationship_role: RelationshipRoleQuery | None = None


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
    asks for a stored fact instead of an event. Natural request phrasings
    ("Háblame de Elena", "Cuéntame sobre Miguel") are person queries, but they
    require a proper-name subject so generic chat is not widened.
    """

    def route(self, text: str) -> RouteResult:
        normalized = text.strip()
        if not normalized:
            return RouteResult(intent=Intent.unknown)

        # A pronoun-only question is not a proper name. Leave it as general
        # chat so the service's pronoun follow-up block can reuse the verified
        # previous subject, or stay safely conversational when none exists.
        if is_explicit_pronoun_question(normalized):
            return RouteResult(intent=Intent.general_chat)

        # An explicit relationship role ("mi marido", "¿Y mi pareja?") is an
        # explicit person query, but the subject comes from the conversation's
        # active resolved person rather than from the role words themselves.
        relationship_role = parse_relationship_role_query(normalized)
        if relationship_role is not None:
            return RouteResult(
                intent=Intent.person_query,
                relationship_role=relationship_role,
            )

        match = _PERSON_QUESTION.search(normalized)
        if match:
            term = match.group(1).strip()
            if term:
                return RouteResult(intent=Intent.person_query, term=term)

        # Natural request phrasings join the same top-priority person query
        # bucket. They require a capitalized proper name, so they never steal
        # generic chat ("Cuéntame un chiste") or memory recall ("¿Te acuerdas
        # de cuando venía María?").
        for pattern in _PERSON_REQUEST_PATTERNS:
            match = pattern.search(normalized)
            if match:
                term = match.group("name").strip()
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

    def route_fact_follow_up(self, text: str) -> RouteResult | None:
        """Classify an elliptical fact follow-up, without choosing a subject.

        A follow-up must start with an explicit continuation marker ("¿Y ...?")
        and name a fact topic while omitting the subject. The caller is
        responsible for supplying the immediately previous resolved subject;
        this method never guesses one. An explicit subject in the text is
        handled by :meth:`route` and wins over any carried context.
        """
        normalized = text.strip()
        if not normalized:
            return None
        if not _FACT_FOLLOW_UP.search(normalized):
            return None
        if not _FACT_QUESTION.search(normalized):
            return None
        if _extract_fact_subject(normalized) is not None:
            return None
        topic = classify_fact_question(normalized)
        if topic is None:
            return None
        return RouteResult(intent=Intent.fact_query, term=None, fact_topic=topic)

    def route_possessive_follow_up(self, text: str) -> RouteResult | None:
        """Classify a follow-up that names a topic with "su"/"sus".

        Examples: "Háblame sobre su infancia", "cuentame de su ninez",
        "¿Qué sabes de su familia?". The method extracts only the topic; the
        caller supplies the immediately previous resolved person. An explicit
        name in the same phrase is left to :meth:`route`, so this method never
        overrides a new subject. Returns ``None`` when the text is not a
        recognizable possessive request or no fact family matches.
        """
        normalized = text.strip()
        if not normalized:
            return None
        match = _POSSESSIVE_FOLLOW_UP.search(normalized)
        if match is None:
            return None
        if _extract_fact_subject(normalized) is not None:
            return None
        topic_text = match.group("topic").strip()
        topic = classify_fact_question(topic_text)
        if topic is None:
            return None
        return RouteResult(
            intent=Intent.fact_query,
            term=None,
            fact_topic=topic,
            follow_up_topic=FOLLOW_UP_TOPIC_LABELS.get(topic),
        )

    def route_pronoun_follow_up(self, text: str) -> RouteResult | None:
        """Classify a pronoun-only request for the previous resolved person.

        Examples: "¿Y ella?", "¿Qué sabes de él?", "Háblame de ese". The method
        returns a ``person_query`` with no term; the caller supplies the
        immediately previous resolved person or leaves the turn unchanged. An
        explicit name is never overridden.
        """
        normalized = text.strip()
        if not normalized:
            return None
        if _extract_fact_subject(normalized) is not None:
            return None
        if any(pattern.match(normalized) for pattern in _PRONOUN_FOLLOW_UP_PATTERNS):
            return RouteResult(intent=Intent.person_query)
        return None
