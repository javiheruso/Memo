"""Pure projection of a provisional conversational referent.

A provisional referent answers a narrow question for one turn: *who is the
person talking about in this conversation?* It is derived only from the
user-authored turns already present in the bounded conversation window (plus
the current user message), and it never touches storage.

Design constraints:

* No ORM object, query, cache, table, column, migration or dependency.
* Assistant text can never establish or update a referent.
* The projection carries a surface name/expression, a conversational
  description/relationship, and the unverified user-authored origin.
* A verified registered identity is kept in separate optional fields and is
  only ever supplied by a caller that already ran the normal resolution
  procedure. The projection itself never infers identity from a name.
* A referent is not an entity: it has no database id of its own and must never
  become a fact, relationship, memory, claim, source or knowledge gap.

The service renders the result as an explicit unverified, conversation-only
prompt block. ``AnswerContext`` items, allowed source ids and response sources
stay exactly as they were; a provisional referent is not citable.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace

from app.domain.conversation_window import ConversationWindow


@dataclass(frozen=True)
class ProvisionalReferent:
    """A user-introduced referent that has not been verified against storage.

    ``surface`` is the name or expression the person used (``María``,
    ``mi vecina``). ``description`` is the conversational description or
    relationship the person attached to it, if any. ``origin_text`` is the
    user-authored sentence that introduced or corrected it.

    The ``verified_*`` fields are intentionally separate. They stay ``None``
    unless an existing resolution procedure establishes them; the projection
    never derives an identity from a matching surface name.
    """

    surface: str
    description: str | None
    origin_text: str
    verified_person_id: uuid.UUID | None = None
    verified_display_name: str | None = None

    def with_verified_identity(
        self, *, person_id: uuid.UUID, display_name: str
    ) -> ProvisionalReferent:
        """Attach an identity already established by normal resolution."""
        return replace(
            self,
            verified_person_id=person_id,
            verified_display_name=display_name,
        )


_NAME_TOKEN = r"[^\W\d_]+(?:[-'][^\W\d_]+)*"
_NAME_BODY = rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN})?"

# Relationship and description heads accepted after "mi/una/la". The list is
# deliberately linguistic, not domain data: it keeps "mi vecina" from becoming
# a person named "Mi" while still covering everyday conversational Spanish.
_DESCRIPTION_HEADS = (
    "compañera",
    "compañero",
    "cuidadora",
    "cuidador",
    "fisioterapeuta",
    "profesora",
    "profesor",
    "enfermera",
    "enfermero",
    "vecina",
    "vecino",
    "hermana",
    "hermano",
    "sobrina",
    "sobrino",
    "abuela",
    "abuelo",
    "madre",
    "padre",
    "hija",
    "hijo",
    "nieta",
    "nieto",
    "prima",
    "primo",
    "nuera",
    "yerno",
    "mujer",
    "marido",
    "esposa",
    "esposo",
    "pareja",
    "amiga",
    "amigo",
    "señora",
    "señor",
    "muchacha",
    "muchacho",
    "chica",
    "chico",
)
_DESCRIPTION_ALTERNATION = "|".join(
    re.escape(head) for head in sorted(_DESCRIPTION_HEADS, key=len, reverse=True)
)
_DESCRIPTION_BODY = (
    rf"(?:{_DESCRIPTION_ALTERNATION})"
    rf"(?:\s+(?:de|del|de\s+la)\s+{_NAME_TOKEN}){{0,3}}"
)

# Corrections come first so "María no es mi hija, es mi vecina" supersedes the
# earlier interpretation instead of being read as two unrelated statements.
# Leading correction fillers are stripped before extraction so natural
# corrections such as "No, perdón, María no es mi hija, es mi vecina" are read
# as one statement. Stripping is only used for parsing; the stored origin text
# remains verbatim.
_LEADING_CORRECTION_FILLER = re.compile(
    r"^\s*(?:no|perd[oó]n|perdona|ay|uy|espera|un\s+momento|bueno|oye|"
    r"quiero\s+decir|mejor\s+dicho|en\s+realidad|me\s+refer[ií]a\s+a|que)"
    r"\b[\s,.;:!¿¡-]*",
    re.IGNORECASE,
)


def _without_correction_fillers(text: str) -> str:
    current = text.strip()
    while True:
        stripped = _LEADING_CORRECTION_FILLER.sub("", current, count=1)
        if stripped == current:
            return current
        current = stripped


_P_CORRECTION_NOT = re.compile(
    rf"^\s*(?:no,?\s*)?(?P<name>{_NAME_BODY})\s+no\s+(?:es|era)\s+"
    rf"(?:mi|una|la)\s+(?P<old>{_DESCRIPTION_BODY})\s*[,;.]?\s*"
    rf"(?:sino|es|ahora\s+es)\s+(?:mi|una|la)\s+(?P<desc>{_DESCRIPTION_BODY})",
    re.IGNORECASE,
)
_P_CORRECTION_ES_NOT = re.compile(
    rf"^\s*(?:no,?\s*)?(?P<name>{_NAME_BODY})\s+(?:es|era)\s+"
    rf"(?:mi|una|la)\s+(?P<desc>{_DESCRIPTION_BODY})\s*[,;]\s*"
    rf"no\s+(?:mi|una|la)\s+(?P<old>{_DESCRIPTION_BODY})",
    re.IGNORECASE,
)
_P_CORRECTION_REFERIA = re.compile(
    rf"^\s*(?:perd[oó]n|perdona|quiero\s+decir|me\s+refer[ií]a\s+a|"
    rf"mejor\s+dicho|en\s+realidad)\b[^.!?]*?\b"
    rf"(?P<name>{_NAME_BODY})\s*,\s*(?:mi|una|la)\s+"
    rf"(?P<desc>{_DESCRIPTION_BODY})",
    re.IGNORECASE,
)
_P_CORRECTION_PREFACE = re.compile(
    rf"^\s*(?:perd[oó]n|perdona|quiero\s+decir|me\s+refer[ií]a\s+a|"
    rf"mejor\s+dicho|en\s+realidad|no)\b[^.!?]*?\b"
    rf"(?P<name>{_NAME_BODY})\s*,?\s*(?:es|era)\s+(?:mi|una|la)\s+"
    rf"(?P<desc>{_DESCRIPTION_BODY})",
    re.IGNORECASE,
)
_P_NAME_ES_DESC = re.compile(
    rf"^\s*(?P<name>{_NAME_BODY})\s+(?:es|era)\s+(?:mi|una|la)\s+"
    rf"(?P<desc>{_DESCRIPTION_BODY})",
    re.IGNORECASE,
)
_P_NAME_COMMA_DESC = re.compile(
    rf"^\s*(?P<name>{_NAME_BODY})\s*,\s*(?:mi|una|la)\s+"
    rf"(?P<desc>{_DESCRIPTION_BODY})",
    re.IGNORECASE,
)
_P_DESC_SE_LLAMA = re.compile(
    rf"^\s*(?:mi|una|la)\s+(?P<desc>{_DESCRIPTION_BODY})\s+se\s+llama\s+"
    rf"(?P<name>{_NAME_BODY})",
    re.IGNORECASE,
)
_P_TENGO_DESC_LLAMA = re.compile(
    rf"^\s*(?:tengo|tenemos|conozco)\s+(?:una|un|a\s+una|a\s+un)\s+"
    rf"(?P<desc>{_DESCRIPTION_BODY})\s+"
    rf"(?:que\s+se\s+llama|llamada|de\s+nombre)\s+(?P<name>{_NAME_BODY})",
    re.IGNORECASE,
)
_P_DESC_COMMA_NAME = re.compile(
    rf"^\s*(?:mi|una|la)\s+(?P<desc>{_DESCRIPTION_BODY})\s*,\s*"
    rf"(?P<name>{_NAME_BODY})",
    re.IGNORECASE,
)
_P_DESC_NAME = re.compile(
    rf"^\s*(?:mi|una|la)\s+(?P<desc>{_DESCRIPTION_BODY})\s+"
    rf"(?P<name>{_NAME_BODY})",
    re.IGNORECASE,
)
_P_DESC_ONLY = re.compile(
    rf"^\s*(?:(?:mi|la)\s+(?P<desc_mi>{_DESCRIPTION_BODY})|"
    rf"(?:tengo|tenemos|conozco)\s+(?:una|un)\s+(?P<desc_una>{_DESCRIPTION_BODY}))",
    re.IGNORECASE,
)

# Explicit topic changes win over any carry-over. They are language-level
# markers, not topic knowledge.
_TOPIC_SHIFT = re.compile(
    r"\b(?:cambiando\s+de\s+tema|cambio\s+de\s+tema|cambiemos\s+de\s+tema|"
    r"otra\s+cosa|otro\s+tema|dejemos\s+eso|dejando\s+eso|olvida\s+eso|"
    r"no\s+importa|da\s+igual)\b",
    re.IGNORECASE,
)
_NEW_SUBJECT_REQUEST = re.compile(
    rf"^\s*(?:quiero|prefiero|me\s+gustar[íi]a|vamos\s+a|podemos)\s+"
    rf"hablar\s+(?:de|sobre)\s+(?P<name>{_NAME_BODY})",
    re.IGNORECASE,
)

# Ordinary function words that can never start a person name.
_NAME_STOPWORDS = frozenset(
    {
        "mi",
        "mis",
        "tu",
        "tus",
        "su",
        "sus",
        "una",
        "un",
        "la",
        "el",
        "los",
        "las",
        "tengo",
        "tenemos",
        "conozco",
        "no",
        "perdon",
        "perdona",
        "quiero",
        "decir",
        "me",
        "referia",
        "mejor",
        "dicho",
        "en",
        "realidad",
        "es",
        "era",
        "se",
        "llama",
        "llamada",
        "y",
        "pero",
        "tambien",
        "ella",
        "esa",
        "ese",
        "esta",
        "este",
        "que",
        "quien",
        "como",
        "cuando",
        "donde",
    }
)
_FOLDED_DESCRIPTION_HEADS = frozenset(_DESCRIPTION_HEADS)


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W\d_]+", _fold(value)))


def _clean_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    # The name body may greedily take one trailing token from the next clause
    # ("María y", "María que"). That token is a function word, not part of the
    # name, so trim there instead of discarding the whole capture.
    tokens: list[str] = []
    for token in raw.split():
        if _fold(token) in _NAME_STOPWORDS:
            break
        tokens.append(token)
        if len(tokens) == 2:
            break
    if not tokens:
        return None
    folded = [_fold(token) for token in tokens]
    if len(tokens) == 1 and folded[0] in _FOLDED_DESCRIPTION_HEADS:
        return None
    return " ".join(tokens)


def _clean_description(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = " ".join(raw.split()).strip(" ,.;!?¿¡")
    return text or None


@dataclass(frozen=True)
class _Candidate:
    surface: str | None
    description: str | None


def _extract_candidate(text: str) -> _Candidate | None:
    """Extract an introduction or correction from one user-authored turn."""
    normalized = _without_correction_fillers(text)
    for pattern in (
        _P_CORRECTION_NOT,
        _P_CORRECTION_ES_NOT,
        _P_CORRECTION_REFERIA,
        _P_CORRECTION_PREFACE,
        _P_NAME_ES_DESC,
        _P_NAME_COMMA_DESC,
        _P_DESC_SE_LLAMA,
        _P_TENGO_DESC_LLAMA,
        _P_DESC_COMMA_NAME,
        _P_DESC_NAME,
    ):
        match = pattern.match(normalized)
        if match is None:
            continue
        name = _clean_name(match.groupdict().get("name"))
        description = _clean_description(match.groupdict().get("desc"))
        if name is None and description is None:
            continue
        return _Candidate(surface=name, description=description)

    match = _P_DESC_ONLY.match(normalized)
    if match is not None:
        description = _clean_description(
            match.groupdict().get("desc_mi") or match.groupdict().get("desc_una")
        )
        if description is not None:
            prefix = "mi" if match.groupdict().get("desc_mi") else "una"
            return _Candidate(surface=f"{prefix} {description}", description=description)
    return None


def surfaces_match(first: str, second: str) -> bool:
    """True when two surface expressions name the same token run.

    ``María`` matches ``María Gómez``; ``María`` does not match ``Mariana``.
    """
    first_tokens = set(_tokens(first))
    second_tokens = set(_tokens(second))
    if not first_tokens or not second_tokens:
        return False
    return (
        first_tokens == second_tokens
        or first_tokens <= second_tokens
        or second_tokens <= first_tokens
    )


def _user_turns(
    window: ConversationWindow | None, current_message: str | None
) -> list[str]:
    texts: list[str] = []
    if window is not None:
        texts.extend(turn.text for turn in window.turns if turn.speaker == "user")
    if current_message is not None and current_message.strip():
        texts.append(current_message)
    return texts


def project_provisional_referent(
    window: ConversationWindow | None,
    *,
    current_message: str | None = None,
    is_sensitive: Callable[[str], bool] | None = None,
    ambiguity_pending: bool = False,
    lookup_term_for: Callable[[str], str | None] | None = None,
) -> ProvisionalReferent | None:
    """Project at most one unverified referent from user-authored turns.

    The scan is chronological and keeps the latest introduction or correction.
    A later clear signal (sensitive turn, explicit topic shift, an explicit
    lookup of a different person) drops it. Because the scan is bounded by the
    existing conversation window, a referent whose origin has left the window
    simply disappears.
    """
    referent: ProvisionalReferent | None = None
    for text in _user_turns(window, current_message):
        if is_sensitive is not None and is_sensitive(text):
            referent = None
            continue

        candidate = _extract_candidate(text)
        if candidate is not None:
            # An indirect description mention ("mi vecina me ayuda") is a
            # reference, not a new referent, when it matches the current
            # description.
            if (
                candidate.surface is None
                and referent is not None
                and referent.description is not None
                and candidate.description is not None
                and _fold(referent.description) == _fold(candidate.description)
            ):
                continue
            referent = ProvisionalReferent(
                surface=candidate.surface or candidate.description or "",
                description=candidate.description,
                origin_text=text,
            )
            continue

        if referent is None:
            continue

        if _TOPIC_SHIFT.search(text):
            referent = None
            continue

        lookup_term = lookup_term_for(text) if lookup_term_for is not None else None
        if lookup_term and not surfaces_match(lookup_term, referent.surface):
            referent = None
            continue

        new_subject = _NEW_SUBJECT_REQUEST.match(text)
        if new_subject is not None:
            name = _clean_name(new_subject.group("name"))
            if name is not None and not surfaces_match(name, referent.surface):
                referent = None
                continue

    if ambiguity_pending:
        return None
    return referent


__all__ = [
    "ProvisionalReferent",
    "project_provisional_referent",
    "surfaces_match",
]
