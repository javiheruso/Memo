"""Deterministic parsing of explicit relationship-role questions.

A person can ask about a stored relationship instead of naming a person:
"¿Quién es mi marido?", "¿Y mi esposo?", "Háblame de mi pareja". The parser
only recognizes an explicit role phrase whose noun maps to a relationship type
the domain already stores (``spouse_of`` and ``partner_of``); it never infers a
relationship from a bare pronoun or a generic question.
"""

import re
import unicodedata
from dataclasses import dataclass

from app.domain.enums import RelationshipType


@dataclass(frozen=True)
class RelationshipRoleQuery:
    """A recognized role phrase and the stored relationship types it covers."""

    role: str
    relationship_types: tuple[RelationshipType, ...]


_ROLE_SPECS: dict[str, RelationshipRoleQuery] = {
    "marido": RelationshipRoleQuery("marido", (RelationshipType.spouse_of,)),
    "esposo": RelationshipRoleQuery("esposo", (RelationshipType.spouse_of,)),
    "esposa": RelationshipRoleQuery("esposa", (RelationshipType.spouse_of,)),
    "mujer": RelationshipRoleQuery("mujer", (RelationshipType.spouse_of,)),
    "cónyuge": RelationshipRoleQuery("cónyuge", (RelationshipType.spouse_of,)),
    "conyuge": RelationshipRoleQuery("cónyuge", (RelationshipType.spouse_of,)),
    "pareja": RelationshipRoleQuery(
        "pareja", (RelationshipType.spouse_of, RelationshipType.partner_of)
    ),
}


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).lower()
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


_FOLDED_ROLES = {_fold(role): spec for role, spec in _ROLE_SPECS.items()}
_ROLE_ALTERNATION = "|".join(
    re.escape(role) for role in sorted(_ROLE_SPECS, key=len, reverse=True)
)

_PATTERNS = (
    re.compile(
        rf"^\s*[¿¡]?\s*(?:qui[ée]n|quien)\s+(?:es|era|fue)\s+mi\s+"
        rf"(?P<role>{_ROLE_ALTERNATION})\s*[?¿]*$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*[¿¡]?\s*(?:y\s+)?mi\s+(?P<role>{_ROLE_ALTERNATION})"
        rf"\s*[?.!¿¡]*$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*[¿¡]?\s*(?:h[áa]blame|cu[ée]ntame|dime)\s+(?:algo\s+)?"
        rf"(?:de|sobre|acerca\s+de)\s+mi\s+(?P<role>{_ROLE_ALTERNATION})"
        rf"\s*[?.!¿¡]*$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*[¿¡]?\s*(?:qu[ée]|que)\s+(?:sabes|conoces|tienes)\s+"
        rf"(?:de|sobre|acerca\s+de)\s+mi\s+(?P<role>{_ROLE_ALTERNATION})"
        rf"\s*[?¿]*$",
        re.IGNORECASE,
    ),
)


def parse_relationship_role_query(text: str) -> RelationshipRoleQuery | None:
    """Return the explicit role query, or ``None`` when the text is not one.

    The whole phrase must match, so a declarative sentence that happens to
    contain "mi marido" ("Mi marido se llamaba Miguel") is never treated as a
    lookup request.
    """
    normalized = text.strip()
    if not normalized:
        return None
    for pattern in _PATTERNS:
        match = pattern.match(normalized)
        if match is None:
            continue
        role = match.group("role").strip()
        spec = _FOLDED_ROLES.get(_fold(role))
        if spec is not None:
            return spec
    return None
