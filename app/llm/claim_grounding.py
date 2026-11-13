"""Deterministic support checks for responder answers.

An allowed source id only proves that an entity was visible to the model; it
does not prove that the sentence the model wrote is supported by that entity.
These helpers add the missing, purely lexical check:

- a claim segment is supported when every content token in it belongs to one
  single stored context item (subject, relation, fact, memory, event or media),
  so a sentence cannot splice a fact from one item onto a person from another;
- an honest no-data answer is safe when the only content it keeps, beyond the
  person/topic the user asked about, is generic no-data wording;
- answer classes whose wording cannot be attributed deterministically
  (unstructured memory prose, multi-subject relations, sensitive topics) are
  routed to the deterministic responder instead of being trusted.

The checks are intentionally conservative. They do not attempt natural-language
understanding and they are not a model verifier: a false negative is preferred
over accepting an invented personal detail.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.domain.enums import Intent, invert_relationship_type
from app.retrieval.context_builder import AnswerContext, FactItem, MemoryItem
from app.retrieval.entity_resolver import ResolutionKind

_TOKEN = re.compile(r"[a-z0-9]+")

# Generic function words and conversational framing. They carry no factual
# detail and are ignored when matching a sentence against stored text.
_GROUNDING_STOPWORDS = frozenset(
    {
        "a",
        "al",
        "algo",
        "como",
        "con",
        "cuando",
        "de",
        "aunque",
        "quien",
        "pues",
        "porque",
        "mientras",
        "donde",
        "del",
        "el",
        "ella",
        "el",
        "ellos",
        "en",
        "era",
        "es",
        "esa",
        "esta",
        "ese",
        "eso",
        "esos",
        "estas",
        "este",
        "estos",
        "fue",
        "ha",
        "la",
        "las",
        "lo",
        "los",
        "mas",
        "me",
        "mi",
        "mis",
        "no",
        "que",
        "se",
        "sin",
        "sobre",
        "son",
        "su",
        "sus",
        "te",
        "tu",
        "tus",
        "un",
        "una",
        "y",
        "ya",
        # Frame words a natural answer uses around stored text. They carry no
        # factual detail: "tengo un recuerdo guardado que encaja con eso"
        # repeats the stored content, it does not add to it.
        "acuerdo",
        "aqui",
        "asi",
        "contar",
        "contarte",
        "creo",
        "encaja",
        "encajan",
        "entonces",
        "guardada",
        "guardadas",
        "guardado",
        "guardados",
        "has",
        "hemos",
        "han",
        "he",
        "parece",
        "puedo",
        "puedes",
        "recuerdo",
        "recuerdos",
        "recordar",
        "segun",
        "si",
        "solo",
        "tambien",
        "tiene",
        "tienes",
        "tengo",
    }
)

# Generic no-data wording. This is discourse vocabulary, never a name, date or
# topic list: it can only excuse words like "no tengo información", never an
# invented person, place or year.
_NO_DATA_DISCOURSE = frozenset(
    {
        "informacion",
        "dato",
        "datos",
        "detalle",
        "detalles",
        "nada",
        "todavia",
        "aun",
        "saber",
        "sabe",
        "sabria",
        "conozco",
        "conoces",
        "explicar",
        "explicarte",
        "decir",
        "decirte",
        "tema",
    }
)

HONEST_NO_DATA = re.compile(
    r"\bno\s+(?:tengo|s[ée]|recuerdo|conozco|hay|encuentro|dispongo)\b",
    re.IGNORECASE,
)

_CLAIM_SPLIT = re.compile(r"[.!?;\n]+")
_CITATION = re.compile(r"^\s*\(?\s*(?:fuente|source)\s*[:=]", re.IGNORECASE)

# Relationship labels used only to recognise a stored relation in a generated
# sentence. Multi-subject relations are still routed to the deterministic
# responder; this vocabulary is for the evaluator's support audit.
_RELATION_LABELS = {
    "spouse_of": "pareja de",
    "partner_of": "pareja de",
    "sibling_of": "hermano de",
    "parent_of": "progenitor de",
    "child_of": "hijo de",
    "grandparent_of": "abuelo de",
    "grandchild_of": "nieto de",
    "friend_of": "amigo de",
    "other": "conocido de",
}


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).lower()
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _stem(token: str) -> str:
    """Fold a Spanish gender ending so "hija"/"hijo" compare equal.

    This is a morphological normalization, not a vocabulary: it is applied to
    every token on both sides of the comparison.
    """
    if len(token) >= 5 and token.endswith("s"):
        token = token[:-1]
    if len(token) >= 4 and token[-1] in "ao":
        return token[:-1]
    return token


_FOLDED_STOPWORDS = frozenset(_fold(word) for word in _GROUNDING_STOPWORDS)
_FOLDED_NO_DATA = frozenset(_stem(_fold(word)) for word in _NO_DATA_DISCOURSE)


def content_tokens(text: str) -> set[str]:
    """Return folded, gender-folded content tokens with stopwords removed."""
    return {
        _stem(token)
        for token in _TOKEN.findall(_fold(text))
        if len(token) >= 2 and token not in _FOLDED_STOPWORDS
    }


@dataclass(frozen=True)
class _SupportUnit:
    kind: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class ClaimSupportAudit:
    """Safe, deterministic classification of an answer's textual claims."""

    supported_claims: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    source_absent_personal_claims: tuple[str, ...] = ()

    @property
    def has_unsupported(self) -> bool:
        return bool(self.unsupported_claims)


def _segments(answer: str) -> list[str]:
    return [segment.strip() for segment in _CLAIM_SPLIT.split(answer) if segment.strip()]


def _is_question(segment: str) -> bool:
    # Segments are split on the closing mark, so an opening Spanish "¿" is the
    # only reliable signal. A statement that merely ends with ", ¿verdad?" is
    # still audited as a claim.
    return segment.lstrip().startswith("¿")


def _unit(kind: str, *parts: str | None) -> _SupportUnit:
    tokens: set[str] = set()
    for part in parts:
        if part:
            tokens |= content_tokens(part)
    return _SupportUnit(kind=kind, tokens=frozenset(tokens))


def _relation_label(relationship_type: object) -> str:
    return _RELATION_LABELS.get(str(getattr(relationship_type, "value", relationship_type)), "")


def support_units(context: AnswerContext) -> list[_SupportUnit]:
    """Every stored item rendered as its own support unit."""
    subject = context.person.display_name if context.person is not None else None
    units: list[_SupportUnit] = []
    if subject:
        units.append(_unit("subject", subject))
    for item in (*context.verified_relations, *context.probable_relations):
        label = _relation_label(item.relationship_type)
        inverse = _relation_label(invert_relationship_type(item.relationship_type))
        units.append(_unit("relationship", subject, label, item.other_display_name))
        units.append(_unit("relationship", item.other_display_name, inverse, subject))
    for fact in context.facts:
        units.append(_fact_unit(subject, fact))
    for memory in context.memories:
        units.append(_memory_unit(subject, memory))
    for event in context.events:
        units.append(_unit("event", subject, event.title))
    for media in context.media:
        units.append(
            _unit(
                "media",
                subject,
                media.title,
                media.description,
                media.place_text,
            )
        )
    return units


def _fact_unit(subject: str | None, fact: FactItem) -> _SupportUnit:
    return _unit("fact", subject, fact.text)


def _memory_unit(subject: str | None, memory: MemoryItem) -> _SupportUnit:
    return _unit("memory", subject, memory.title, memory.narrative)


def has_stored_detail(context: AnswerContext) -> bool:
    """True when the context carries any stored entity beyond a bare subject."""
    return bool(
        context.verified_relations
        or context.probable_relations
        or context.facts
        or context.memories
        or context.events
        or context.media
    )


def _allowed_no_data_tokens(context: AnswerContext, message: str | None) -> set[str]:
    allowed: set[str] = set(_FOLDED_NO_DATA)
    if context.person is not None:
        allowed |= content_tokens(context.person.display_name)
    if context.term:
        allowed |= content_tokens(context.term)
    if context.follow_up_topic:
        allowed |= content_tokens(context.follow_up_topic)
    if message:
        allowed |= content_tokens(message)
    return allowed


def answer_is_claim_free(
    context: AnswerContext, answer: str, message: str | None = None
) -> bool:
    """True when the answer only repeats the subject/question or no-data copy.

    A bare name (``"Francisco."``) or an honest no-data sentence passes; any
    invented person, place, year, event or attribute fails.
    """
    allowed = _allowed_no_data_tokens(context, message)
    for segment in _segments(answer):
        if _is_question(segment):
            continue
        tokens = content_tokens(segment)
        if tokens and not tokens <= allowed:
            return False
    return True


def no_data_answer_is_safe(
    context: AnswerContext, answer: str, message: str | None = None
) -> bool:
    """True when a "no stored data" answer is explicit and adds no claim."""
    return bool(HONEST_NO_DATA.search(answer)) and answer_is_claim_free(
        context, answer, message
    )


def answer_claims_are_supported(
    context: AnswerContext,
    answer: str,
    *,
    kinds: frozenset[str],
    message: str | None = None,
) -> bool:
    """True when every claim segment is supported by one single context item."""
    units = [
        unit
        for unit in support_units(context)
        if unit.kind == "subject" or unit.kind in kinds
    ]
    for segment in _segments(answer):
        if _is_question(segment) or _CITATION.match(segment):
            continue
        if no_data_answer_is_safe(context, segment, message):
            continue
        tokens = content_tokens(segment)
        if not tokens:
            continue
        if not any(tokens <= unit.tokens for unit in units):
            return False
    return True


def answer_needs_deterministic_responder(
    context: AnswerContext, answer: str, message: str | None = None
) -> bool:
    """True when generated wording cannot be deterministically supported.

    - an explicitly raised sensitive topic is answered by the careful
      deterministic copy, which never adds details;
    - memory queries quote unstructured prose, where attribution cannot be
      checked from the stored text alone;
    - a person answer with any stored detail can splice facts, relations or
      memories across people, so it is rendered deterministically;
    - fact and event answers are accepted only when every claim is covered by
      the corresponding stored item; a no-data answer must be honestly
      source-free.
    """
    if context.sensitive_topic_requested is not None:
        return True
    if context.intent == Intent.memory_query:
        return True
    if context.intent == Intent.person_query:
        if context.resolution == ResolutionKind.ambiguous:
            return False
        if has_stored_detail(context):
            return True
        return not answer_is_claim_free(context, answer, message)
    if context.intent == Intent.fact_query:
        if not context.facts:
            return not no_data_answer_is_safe(context, answer, message)
        return not answer_claims_are_supported(
            context, answer, kinds=frozenset({"fact"}), message=message
        )
    if context.intent == Intent.event_query:
        if not context.events:
            return not no_data_answer_is_safe(context, answer, message)
        return not answer_claims_are_supported(
            context, answer, kinds=frozenset({"event"}), message=message
        )
    return False


def audit_answer_claims(
    context: AnswerContext, answer: str, message: str | None = None
) -> ClaimSupportAudit:
    """Classify an already-produced answer for the evaluation harness.

    Deterministic fallbacks are not audited here: the harness records them as a
    separate outcome. This function describes model-accepted text only.
    """
    units = support_units(context)
    detail = has_stored_detail(context)
    supported: list[str] = []
    unsupported: list[str] = []
    source_absent: list[str] = []
    for segment in _segments(answer):
        if _is_question(segment) or _CITATION.match(segment):
            continue
        if no_data_answer_is_safe(context, segment, message):
            continue
        tokens = content_tokens(segment)
        if not tokens:
            continue
        if any(tokens <= unit.tokens for unit in units):
            supported.append(segment)
        else:
            unsupported.append(segment)
            if not detail:
                source_absent.append(segment)
    return ClaimSupportAudit(
        supported_claims=tuple(supported),
        unsupported_claims=tuple(unsupported),
        source_absent_personal_claims=tuple(source_absent),
    )
