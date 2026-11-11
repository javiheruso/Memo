"""Verify an interpreter suggestion against Memo's own records.

A suggestion is never trusted as a route. The person term is resolved through
the same ``EntityResolver``/``PersonRepository`` used by the deterministic path,
so ambiguity is preserved and an unknown name is rejected instead of producing
an answer or a knowledge gap. Topics were already constrained to ``FactTopic``
by the suggestion schema; this module only builds the verified route.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

from app.domain.enums import Intent
from app.retrieval.entity_resolver import EntityResolver, Resolution, ResolutionKind
from app.retrieval.intent_router import RouteResult
from app.schemas.interpreter import InterpreterSuggestion

logger = logging.getLogger(__name__)


def fold(value: str) -> str:
    """Lowercase and strip accents for spelling-insensitive comparisons."""
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return "".join(char for char in normalized if not unicodedata.combining(char))


def term_is_grounded(message: str, term: str | None) -> bool:
    """True only when ``term`` is a whole token run of the current message.

    The interpreter may only reference a person the user actually mentioned in
    the current turn. History, earlier turns and database names are not
    evidence, and a substring such as ``Antonio`` inside ``Antoniola`` does not
    count. Accents and case are ignored; punctuation is a token boundary.
    """
    if not term:
        return False
    folded_term = fold(term).strip()
    if not folded_term:
        return False
    folded_message = fold(message)
    pattern = rf"(?<![a-z0-9]){re.escape(folded_term)}(?![a-z0-9])"
    return re.search(pattern, folded_message) is not None


@dataclass(frozen=True)
class VerifiedInterpreterRoute:
    """A suggestion whose person term exists in the database."""

    route: RouteResult
    resolution: Resolution


class InterpreterVerifier:
    def __init__(self, resolver: EntityResolver) -> None:
        self._resolver = resolver

    async def verify(
        self, suggestion: InterpreterSuggestion, *, message: str
    ) -> VerifiedInterpreterRoute | None:
        """Return a usable route, or ``None`` to keep deterministic behavior.

        The suggested term must be grounded as a whole token run in the current
        user message before any database lookup. An ``ambiguous`` resolution is
        returned as-is: the existing pipeline surfaces the candidates and never
        collapses them into one person. A ``not_found`` resolution is discarded
        so the interpreter can never open a knowledge gap.
        """
        if suggestion.intent is None or not suggestion.term:
            return None
        if not term_is_grounded(message, suggestion.term):
            logger.warning(
                "interpreter suggestion rejected: term not grounded in the "
                "current message"
            )
            return None

        resolution = await self._resolver.resolve(suggestion.term)
        if resolution.kind == ResolutionKind.not_found:
            return None

        if suggestion.intent == "person_query":
            return VerifiedInterpreterRoute(
                route=RouteResult(
                    intent=Intent.person_query,
                    term=suggestion.term,
                ),
                resolution=resolution,
            )

        if suggestion.fact_topic is None:
            return None
        return VerifiedInterpreterRoute(
            route=RouteResult(
                intent=Intent.fact_query,
                term=suggestion.term,
                fact_topic=suggestion.fact_topic,
                follow_up_topic=suggestion.follow_up_topic,
            ),
            resolution=resolution,
        )


__all__ = [
    "InterpreterVerifier",
    "VerifiedInterpreterRoute",
    "fold",
    "term_is_grounded",
]
