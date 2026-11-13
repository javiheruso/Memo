import re
import unicodedata
from dataclasses import dataclass

from app.domain.enums import Intent
from app.retrieval.fact_topics import FactTopic, classify_fact_question
from app.retrieval.relationship_roles import (
    RelationshipRoleQuery,
    RelativeFollowUpQuery,
    parse_relationship_role_query,
)

# A name candidate is a short run of letter-only words. It intentionally does
# not require capitalization: STT transcripts arrive lowercase and without
# punctuation. Ordinary function words and generic nouns are rejected so
# "háblame de la vida" never becomes a person term; an unresolved lowercase
# candidate is also treated as ordinary conversation by the service.
_NAME_TOKEN = r"[^\W\d_]+(?:[-'][^\W\d_]+)*"
_NON_NAME_TOKENS = frozenset(
    {
        # Spanish function words and generic request vocabulary.
        "la", "el", "los", "las", "lo", "un", "una", "unos", "unas",
        "de", "del", "al", "a", "en", "con", "por", "para", "sobre",
        "mi", "mis", "tu", "tus", "su", "sus", "él", "ella", "ellos",
        "ellas", "ese", "esa", "eso", "esos", "esas", "este", "esta",
        "ti", "mí", "me", "nos", "os", "vos", "yo", "tú", "usted",
        "ustedes", "nosotros", "nosotras", "vosotros", "vosotras",
        "estos", "estas", "aquel", "aquella", "que", "qué", "quien",
        "quién", "cual", "cuál", "cuando", "cuándo", "donde", "dónde",
        "como", "cómo", "cuanto", "cuánto", "y", "o", "u", "pero", "si",
        "sí", "no", "muy", "más", "mas", "menos", "todo", "toda", "todos",
        "todas", "algo", "nada", "ayer", "hoy", "mañana", "tarde",
        "noche", "semana", "mes", "año", "vida", "casa", "playa",
        "viaje", "historia", "tema", "cosa", "cosas", "familia",
        "trabajo", "música", "musica", "comida", "salud", "tiempo",
        "día", "dia", "pueblo", "ciudad", "mundo", "país", "pais",
        "recuerdo", "recuerdos", "proyecto", "proyectos", "foto",
        "fotos", "documento", "documentos", "dato", "datos", "asunto",
        "parte", "lugar", "momento", "vez", "veces", "era", "eran", "es",
        "fue", "son",
        # English function words and generic request vocabulary.
        "the", "of", "about", "and", "with", "for", "what", "which",
        "where", "who", "how", "when", "like", "lives", "live", "lived",
        "work", "works", "born", "does", "did", "is", "was", "are",
    }
)


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


_FOLDED_NON_NAME_TOKENS = frozenset(_fold(token) for token in _NON_NAME_TOKENS)
_STOP_NAMES = "|".join(
    re.escape(token)
    for token in sorted(_NON_NAME_TOKENS, key=len, reverse=True)
)
_NAME_CANDIDATE = (
    rf"{_NAME_TOKEN}(?:\s+(?!(?:{_STOP_NAMES})\b){_NAME_TOKEN}){{0,2}}"
)


def _clean_name_candidate(raw: str) -> str | None:
    """Return the leading name-like words of ``raw``, or ``None``.

    The regex only bounds the candidate length; this helper stops at the first
    function word or generic noun. It never invents a name, and a candidate made
    only of non-name words is rejected.
    """
    selected: list[str] = []
    for token in raw.split():
        if _fold(token) in _FOLDED_NON_NAME_TOKENS:
            break
        selected.append(token)
        if len(selected) == 3:
            break
    if not selected:
        return None
    return " ".join(selected)


_PERSON_QUESTION = re.compile(
    r"(?:quién|quien|who)\s+(?:es|era|fue|is|was)\s+([^?¿]+?)\s*[?¿]*$",
    re.IGNORECASE,
)
# Natural ways to ask about a person: "Háblame de Elena", "Cuéntame sobre
# Miguel", "Me gustaría saber de Elena", "Tengo curiosidad por Miguel". Names
# may be lowercase (STT) but must survive `_clean_name_candidate`, so generic
# requests ("Cuéntame un chiste", "Háblame de la vida") stay general chat. The
# trailing courtesy clause is optional: "Háblame de Elena, por favor".
_PERSON_REQUEST_SUFFIX = (
    rf"(?P<name>{_NAME_CANDIDATE})\s*(?:,\s*por\s+favor)?\s*[?.!¿¡]*$"
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
    rf"(?:hablar(?:me)?|contar(?:me)?|decir(?:me)?|describir(?:me)?)\s+"
    rf"(?:algo\s+)?(?:de|sobre|acerca\s+de)\s+{_PERSON_REQUEST_SUFFIX}",
    re.IGNORECASE,
)
_PERSON_INFO_SEEKING_REQUEST = re.compile(
    rf"^\s*[¿¡]?\s*(?:me\s+gustar[íi]a|me\s+gustaria|quisiera|quiero|necesito)\s+"
    rf"(?:saber|conocer|preguntar(?:te)?)\s+(?:algo\s+)?"
    rf"(?:de|sobre|acerca\s+de|por)\s+{_PERSON_REQUEST_SUFFIX}",
    re.IGNORECASE,
)
_PERSON_CURIOSITY_REQUEST = re.compile(
    rf"^\s*[¿¡]?\s*(?:tengo\s+(?:mucha\s+)?curiosidad|me\s+interesa)\s+"
    rf"(?:por|sobre|de)\s+{_PERSON_REQUEST_SUFFIX}",
    re.IGNORECASE,
)
_PERSON_ENGLISH_REQUEST = re.compile(
    rf"^\s*(?:tell\s+me\s+about|what\s+do\s+you\s+know\s+about|"
    rf"can\s+you\s+tell\s+me\s+about|could\s+you\s+tell\s+me\s+about|"
    rf"i\s+want\s+to\s+know\s+about|i'?d\s+like\s+to\s+know\s+about|"
    rf"i\s+would\s+like\s+to\s+know\s+about)\s+{_PERSON_REQUEST_SUFFIX}",
    re.IGNORECASE,
)
_PERSON_REQUEST_PATTERNS = (
    _PERSON_ABOUT_REQUEST,
    _PERSON_DESCRIBE_REQUEST,
    _PERSON_POLITE_REQUEST,
    _PERSON_INFO_SEEKING_REQUEST,
    _PERSON_CURIOSITY_REQUEST,
    _PERSON_ENGLISH_REQUEST,
)
# "¿Te acuerdas de cuando íbamos a bailar?" / "Do you remember when we went
# dancing?" -> semantic memory retrieval. Checked before event markers so a
# recollection phrasing about a person or event ("¿Te acuerdas de cuando venía
# María?") stays a memory query.
_MEMORY_MARKERS = re.compile(
    r"\b(?:te\s+acuerdas\b|recuerdas\b|do\s+you\s+remember\b|remember\s+when\b)",
    re.IGNORECASE,
)
# Questions about the person's own stored knowledge ("¿Tengo algún proyecto?",
# "¿Qué sabes de mis recuerdos?") are memory questions, not generic chat. The
# vocabulary names stored-knowledge nouns, never a specific subject, so the
# same rule works for any topic the person asks about.
_PERSONAL_KNOWLEDGE_MARKERS = re.compile(
    r"\b(?:"
    r"(?:qu[ée]|que|cu[áa]les|cuales)\s+(?:sabes|recuerdas|tienes|conoces|hay)"
    r"\s+(?:de|sobre|acerca\s+de)\s+(?:mi|mis|m[íi]|lo\s+que)\b"
    r"|(?:tengo|tenemos|hay)\s+(?:alg[uú]n[oa]?s?|algo)\s+(?:de\s+)?"
    r"(?:proyect\w*|recuerd\w*|memori\w*|histori\w*|fotograf\w*|foto\w*|"
    r"document\w*|registr\w*|dato\w*)\b"
    r"|(?:qu[ée]|que|cu[áa]les|cuales)\s+"
    r"(?:proyect\w*|recuerd\w*|memori\w*|histori\w*|fotograf\w*|foto\w*|"
    r"document\w*|registr\w*|dato\w*)\s+(?:tengo|tenemos|hay)\b"
    r")",
    re.IGNORECASE,
)
# Short replies that accept a previous offer. They are language-level markers,
# not topic rules: the topic always comes from the immediately previous
# assistant question, never from this list.
_ACCEPTANCE = re.compile(
    r"^\s*[¡¿]?\s*(?:"
    r"(?:s[íi]|claro(?:\s+que\s+s[íi])?|vale|de\s+acuerdo|perfecto|bueno|"
    r"ok(?:ey)?|aj[áa]|por\s+supuesto|desde\s+luego|entendido)"
    r"(?:\s*,?\s*(?:por\s+favor|gracias|claro|"
    r"cu[ée]nta(?:me|melo|mela|nos)?|dime|adelante|sigue|contin[úu]a))?|"
    r"cu[ée]nta(?:me|melo|mela|nos)?"
    r"(?:\s+(?:algo\s+m[áa]s|m[áa]s|eso|esto|lo\s+mismo|"
    r"eso\s+mismo|esto\s+mismo))?|"
    r"dime(?:\s+(?:eso|esto|lo\s+mismo|eso\s+mismo|esto\s+mismo|m[áa]s))?|"
    r"sigue|contin[úu]a|adelante|exacto|correcto|as[íi]\s+es|"
    r"me\s+interesa|me\s+gustar[íi]a|quiero|"
    r"(?:eso|esto|esa|ese)\s+mism[oa]|el\s+mismo|la\s+misma|eso|esto|esa|ese"
    r")\s*[.!]*\s*$",
    re.IGNORECASE,
)
# "Sobre costura", "de música", "hablando de la playa": a topic selection in
# answer to the previous offer. It is only used when the previous assistant
# turn asked a question, so it never hijacks an unrelated topic switch.
_TOPIC_SELECTION = re.compile(
    r"^\s*[¿¡]?\s*(?:sobre|de|hablando\s+de|acerca\s+de)\s+\S.*$",
    re.IGNORECASE,
)
_EVENT_MARKERS = re.compile(
    r"\b(viene|vienen|vendr[áa]|vendr[áa]n|cu[áa]ndo|evento|eventos|quedada)\b",
    re.IGNORECASE,
)
_EVENT_PERSON = re.compile(
    rf"(?:viene|vienen|vendr[áa]|vendr[áa]n)\s+(?P<arrival>{_NAME_CANDIDATE})"
    rf"|(?:evento|eventos|quedada)\b[^?¿]*?\b"
    rf"(?:tiene|tienen|hay|habr[áa]|tendr[áa]|es|son)\s+(?P<owner>{_NAME_CANDIDATE})",
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
# A relative follow-up keeps the previous resolved person and asks about one
# close family link: "¿Y su hija?", "¿Y su padre?". It is deliberately narrow:
# only these four nouns and no extra content, so an unrelated possessive
# sentence stays general chat. The caller supplies the active person and
# refuses the turn when there is no verified carry-over.
_RELATIVE_FOLLOW_UP = re.compile(
    r"^\s*[¿¡]?\s*(?:y\s+)?su\s+(?P<role>hija|hijo|madre|padre)\s*[?.!¿¡]*$",
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
    rf"\b(?i:a|de|para|sobre)\s+(?P<name>{_NAME_CANDIDATE})"
)
_FACT_SUBJECT_VERB = re.compile(
    rf"\b(?i:tiene|tienen|prefiere|prefieren|naci[oó]|vivi[oó]|vive|"
    rf"trabaj[oó]|trabaja|dedicaba|dedica|opina|cree|piensa)\s+"
    rf"(?P<name>{_NAME_CANDIDATE})"
)
_FACT_SUBJECT_POSSESSIVE = re.compile(rf"(?P<name>{_NAME_CANDIDATE})'s\b")
_FACT_SUBJECT_ENGLISH_AUX = re.compile(
    rf"\b(?i:does|did|was|is|are)\s+(?P<name>{_NAME_CANDIDATE})"
)
_FACT_SUBJECT_PATTERNS = (
    _FACT_SUBJECT_PREPOSITION,
    _FACT_SUBJECT_VERB,
    _FACT_SUBJECT_POSSESSIVE,
    _FACT_SUBJECT_ENGLISH_AUX,
)


def _is_question_like(text: str) -> bool:
    normalized = text.strip()
    return (
        "?" in normalized
        or "¿" in normalized
        or re.match(
            r"^(?:qu[ée]|que|cu[áa]l|cu[áa]les|cual|cuales|dime|"
            r"cu[ée]ntame|h[áa]blame)\b",
            normalized,
            re.IGNORECASE,
        )
        is not None
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
        for match in pattern.finditer(text):
            term = _clean_name_candidate(match.group("name"))
            if term is None:
                continue
            # A question-topic word ("de comer", "su infancia") is not an
            # explicit person subject, so elliptical follow-ups stay elliptical.
            if classify_fact_question(term) is not None:
                continue
            return term
    return None


# Request cues are the boundary between a personal lookup and a narrative.
# They are deliberately language-level: interrogative words, imperative or
# info-seeking verbs, recollection markers and question punctuation. A bare
# third-person narrative ("María viene mañana") or a self-disclosure
# ("Ayer estuve pensando en Carmen") has no cue and stays conversational.
_REQUEST_CUE_PREFIX = re.compile(
    r"^\s*[¿¡]?\s*(?:"
    r"dime|d[íi]me|cu[ée]ntame|cuentame|h[áa]blame|hablame|descr[íi]beme|"
    r"describeme|me\s+gustar[íi]a|me\s+gustaria|quisiera|quiero|necesito|"
    r"puedes|podr[íi]as|puede|podr[íi]a|sabes|conoces|sabr[íi]as|"
    r"conocer[íi]as|tengo\s+curiosidad|me\s+interesa|tell\s+me|"
    r"what\s+do\s+you\s+know|can\s+you\s+tell|could\s+you\s+tell|"
    r"i\s+want\s+to\s+know|i'?d\s+like\s+to\s+know|i\s+would\s+like\s+to\s+know"
    r")\b",
    re.IGNORECASE,
)
_INTERROGATIVE_START = re.compile(
    r"^\s*[¿¡]?\s*(?:y\s+)?(?:qu[ée]|que|cu[áa]l|cuales|cual|"
    r"cu[áa]ndo|cuando|d[óo]nde|donde|c[óo]mo|como|qui[ée]n|quien|"
    r"qui[ée]nes|cu[áa]nto|cuanto|what|which|where|when|who|how)\b",
    re.IGNORECASE,
)
_INTERROGATIVE_ANYWHERE = re.compile(
    r"\b(?:qu[ée]|que|cu[áa]l|cuales|cual|cu[áa]ndo|cuando|d[óo]nde|"
    r"donde|c[óo]mo|como|qui[ée]n|quien|qui[ée]nes|cu[áa]nto|cuanto|"
    r"what|which|where|when|who|how)\b",
    re.IGNORECASE,
)
# "¿Sabes que ...?" / "¿A que no sabes que ...?" states knowledge instead of
# asking for stored data, so it must not promote a personal lookup.
_KNOWLEDGE_CHECK = re.compile(
    r"^\s*[¿¡]?\s*(?:a\s+que\s+no\s+)?(?:sabes|sab[íi]as|no\s+sabes|"
    r"te\s+cuento)\s+que\b",
    re.IGNORECASE,
)
# "¡Qué bien que ...!" / "¡Qué alegría!" are exclamations, not information
# requests, even though they start with an interrogative word.
_EXCLAMATION_START = re.compile(
    r"^\s*[¿¡]?\s*(?:qu[ée]|que)\s+(?:bien|bueno|buena|bonito|bonita|"
    r"grande|alegr[íi]a|gusto|pena|l[áa]stima|suerte|orgullo|rabia|miedo)\b",
    re.IGNORECASE,
)
# A narrative preface ("Ayer vi a María, ¿qué crees?") does not turn a
# recollection into a personal lookup just because a later clause has a
# question word. "Cuando" is intentionally absent: "cuando viene María" is a
# valid STT event request.
_NARRATIVE_START = re.compile(
    r"^\s*[¿]?\s*(?:ayer|anoche|hoy|antes|después|despues|estuve|estaba|"
    r"estuvimos|vi|vimos|fui|fuimos|recuerdo|me\s+acuerdo)\b",
    re.IGNORECASE,
)


# Candidate selection is a short answer to the previous ambiguity question,
# not a narrative that happens to contain a candidate's name. Narrative markers
# keep "Ayer vi a Paco García" from silently selecting a candidate.
_NARRATIVE_SELECTION_MARKERS = re.compile(
    r"\b(?:ayer|anoche|antes|después|despues|cuando|entonces|estuve|estaba|"
    r"estuvimos|estuvo|vi|vimos|vio|fui|fuimos|ten[ií]a|viv[ií]a|"
    r"vivi[oó]|recuerdo|acuerdo|viene|vinieron|vino|vendr[áa]|mañana|hoy|"
    r"siempre|nunca|muri[oó]|falleci[oó])\b",
    re.IGNORECASE,
)
_CANDIDATE_ORDINAL = re.compile(
    r"^\s*(?:primer[oa]|segund[oa]|tercer[oa]|[uú]ltim[oa]|\d+)"
    r"\s*[.!?]*$",
    re.IGNORECASE,
)
_CANDIDATE_ARTICLES = re.compile(
    r"^\s*(?:el|la|los|las|un|una)\s+(?:(?:de|del)\s+)?",
    re.IGNORECASE,
)
# "él era el panadero" is a clarification of the previous candidate, not a
# narrative sentence. The pronoun/copula lead-in is stripped before the name
# check; narrative markers still reject a real recollection.
_CANDIDATE_LEAD_IN = re.compile(
    r"^\s*(?:él|ella|ese|esa)\s+(?:era|es|fue)\s+",
    re.IGNORECASE,
)


def _has_explicit_request_cue(text: str) -> bool:
    normalized = text.strip()
    if (
        _KNOWLEDGE_CHECK.match(normalized)
        or _EXCLAMATION_START.match(normalized)
        or _NARRATIVE_START.match(normalized)
    ):
        return False
    if _REQUEST_CUE_PREFIX.match(normalized):
        return True
    if _INTERROGATIVE_START.match(normalized):
        return True
    if "?" in normalized or "¿" in normalized:
        # A question mark is enough for an interrogative-word question or a
        # yes/no event question ("¿Viene María hoy?"). A narrative question
        # such as "¿Ayer vi a María?" has no cue and stays conversational.
        if _INTERROGATIVE_ANYWHERE.search(normalized):
            return True
        if _EVENT_MARKERS.search(normalized):
            return True
    return False


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
    require a plausible name subject so generic chat is not widened.

    Fact and event intents additionally require :meth:`is_personal_lookup_request`.
    A narrative event mention ("María viene mañana") or a personal comment
    ("Ayer estuve pensando en Carmen") therefore stays ``general_chat`` and
    never reaches entity resolution, in the deterministic path or through an
    optional interpreter suggestion.
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

        lookup_requested = self.is_personal_lookup_request(normalized)

        match = _PERSON_QUESTION.search(normalized)
        if match:
            term = _clean_name_candidate(match.group(1))
            if term:
                return RouteResult(intent=Intent.person_query, term=term)

        # Natural request phrasings join the same top-priority person query
        # bucket. They never steal generic chat ("Cuéntame un chiste") or
        # memory recall ("¿Te acuerdas de cuando venía María?").
        for pattern in _PERSON_REQUEST_PATTERNS:
            match = pattern.search(normalized)
            if match:
                term = _clean_name_candidate(match.group("name"))
                if term:
                    return RouteResult(intent=Intent.person_query, term=term)

        if _MEMORY_MARKERS.search(normalized):
            return RouteResult(intent=Intent.memory_query)

        if _PERSONAL_KNOWLEDGE_MARKERS.search(normalized) and _is_question_like(
            normalized
        ):
            return RouteResult(intent=Intent.memory_query)

        fact_topic = classify_fact_question(normalized)
        if (
            lookup_requested
            and fact_topic is not None
            and _FACT_QUESTION.search(normalized)
        ):
            term = _extract_fact_subject(normalized)
            if term:
                return RouteResult(
                    intent=Intent.fact_query, term=term, fact_topic=fact_topic
                )

        if lookup_requested and _EVENT_MARKERS.search(normalized):
            term = None
            event_match = _EVENT_PERSON.search(normalized)
            if event_match:
                term = _clean_name_candidate(
                    event_match.group("arrival") or event_match.group("owner")
                )
            if term:
                return RouteResult(intent=Intent.event_query, term=term)
            return RouteResult(intent=Intent.event_query)

        for pattern in _ORIENTATION_PATTERNS:
            if pattern.search(normalized):
                return RouteResult(intent=Intent.orientation_query)

        return RouteResult(intent=Intent.general_chat)

    def is_personal_lookup_request(self, text: str) -> bool:
        """True only for a message that asks the assistant for personal data.

        This is the explicit personal-lookup boundary. It recognizes direct and
        natural request phrasings, questions with or without punctuation,
        recollection and stored-knowledge requests, and role questions. It does
        not recognize a bare name, a third-person narrative, a self-disclosure
        or a general conversation turn, so those never reach entity resolution.
        The caller still owns verified carry-over and pending-candidate
        selection; this method only classifies the current text.
        """
        normalized = text.strip()
        if not normalized:
            return False
        if parse_relationship_role_query(normalized) is not None:
            return True
        if _MEMORY_MARKERS.search(normalized):
            return True
        if _PERSONAL_KNOWLEDGE_MARKERS.search(normalized) and _is_question_like(
            normalized
        ):
            return True

        match = _PERSON_QUESTION.search(normalized)
        if match and _clean_name_candidate(match.group(1)) is not None:
            return True
        for pattern in _PERSON_REQUEST_PATTERNS:
            match = pattern.search(normalized)
            if match and _clean_name_candidate(match.group("name")) is not None:
                return True

        if not _has_explicit_request_cue(normalized):
            return False
        return (
            classify_fact_question(normalized) is not None
            or _EVENT_MARKERS.search(normalized) is not None
            or _extract_fact_subject(normalized) is not None
        )

    def is_candidate_selection(self, text: str) -> bool:
        """True for a short reply that picks one pending ambiguity candidate.

        Examples: "el panadero", "el segundo", "2", "Paco García". A narrative
        sentence that merely mentions a candidate ("Ayer vi a Paco García") is
        not a selection, so pending ambiguity is not silently collapsed.
        """
        normalized = text.strip().strip(" ¿¡")
        if not normalized or _NARRATIVE_SELECTION_MARKERS.search(normalized):
            return False
        if _CANDIDATE_ORDINAL.match(normalized):
            return True
        candidate = _CANDIDATE_LEAD_IN.sub("", normalized, count=1)
        candidate = _CANDIDATE_ARTICLES.sub("", candidate, count=1)
        return _clean_name_candidate(candidate.strip(" .!?¿¡")) is not None

    def is_acceptance(self, text: str) -> bool:
        """True for a short acceptance/continuation reply.

        The caller must still verify that the immediately previous assistant
        turn was an offer or question; this method only classifies the reply's
        language shape.
        """
        return _ACCEPTANCE.match(text.strip()) is not None

    def is_topic_selection(self, text: str) -> bool:
        """True for "sobre X" / "de X" after an offer.

        Like :meth:`is_acceptance`, the caller supplies the previous-turn
        requirement. The topic itself is extracted by retrieval, never by the
        router.
        """
        return _TOPIC_SELECTION.match(text.strip()) is not None

    def is_continuation_reply(self, text: str) -> bool:
        """True when ``text`` accepts or selects from a previous offer."""
        return self.is_acceptance(text) or self.is_topic_selection(text)

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

    def route_relative_follow_up(self, text: str) -> RelativeFollowUpQuery | None:
        """Classify "¿Y su hija/hijo/madre/padre?" without choosing a subject.

        The method only recognizes the four unambiguous close-family nouns. It
        never infers gender from names or stored data: the caller matches the
        directed parent/child relation and the responder renders a neutral
        label. The immediately previous resolved person must be supplied by the
        caller; without one the turn stays conversational.
        """
        normalized = text.strip()
        if not normalized:
            return None
        match = _RELATIVE_FOLLOW_UP.match(normalized)
        if match is None:
            return None
        role = _fold(match.group("role"))
        return RelativeFollowUpQuery(
            role=role,
            active_is_parent=role in ("hija", "hijo"),
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
