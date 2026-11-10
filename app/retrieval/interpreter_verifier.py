"""Verify an interpreter suggestion against Memo's own records.

A suggestion is never trusted as a route. The person term is resolved through
the same ``EntityResolver``/``PersonRepository`` used by the deterministic path,
so ambiguity is preserved and an unknown name is rejected instead of producing
an answer or a knowledge gap. Topics were already constrained to ``FactTopic``
by the suggestion schema; this module only builds the verified route.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.enums import Intent
from app.retrieval.entity_resolver import EntityResolver, Resolution, ResolutionKind
from app.retrieval.intent_router import RouteResult
from app.schemas.interpreter import InterpreterSuggestion


@dataclass(frozen=True)
class VerifiedInterpreterRoute:
    """A suggestion whose person term exists in the database."""

    route: RouteResult
    resolution: Resolution


class InterpreterVerifier:
    def __init__(self, resolver: EntityResolver) -> None:
        self._resolver = resolver

    async def verify(
        self, suggestion: InterpreterSuggestion
    ) -> VerifiedInterpreterRoute | None:
        """Return a usable route, or ``None`` to keep deterministic behavior.

        An ``ambiguous`` resolution is returned as-is: the existing pipeline
        surfaces the candidates and never collapses them into one person. A
        ``not_found`` resolution is discarded so the interpreter can never open
        a knowledge gap.
        """
        if suggestion.intent is None or not suggestion.term:
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


__all__ = ["InterpreterVerifier", "VerifiedInterpreterRoute"]
