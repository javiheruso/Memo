"""Deterministic text responder for the foundation batch.

User-facing conversation copy follows the dossier examples, so it is in
Spanish. It is deliberately conservative: it only renders data present in the
``AnswerContext``, respects the epistemic kind of each item, surfaces
ambiguity instead of choosing, and stops at the first progressive hint
(dossier doc 01: name + relation, family context/facts, photograph, shared
memory). Media is handled as metadata only: the responder may cite that a
photograph is stored and its approximate date/place, but it never describes
what an image shows. The ``sources`` it reports are exactly the entities its
text actually used, so ``response_sources`` stays truthful for traceability.
Events are phrased by their own status: a planned or confirmed event is never
presented as already completed.
"""

from datetime import UTC, datetime

from app.domain.enums import EpistemicKind, EventStatus, Intent, RelationshipType
from app.llm.base import Responder, Response
from app.repositories.conversation import ResponseSourceRef
from app.retrieval.context_builder import AnswerContext
from app.retrieval.entity_resolver import ResolutionKind

_PHOTO_MEDIA_TYPE = "photo"

_RELATION_LABELS: dict[RelationshipType, str] = {
    RelationshipType.spouse_of: "pareja de",
    RelationshipType.partner_of: "pareja de",
    RelationshipType.sibling_of: "hermano de",
    RelationshipType.parent_of: "progenitor de",
    RelationshipType.child_of: "hijo de",
    RelationshipType.grandparent_of: "abuelo de",
    RelationshipType.grandchild_of: "nieto de",
    RelationshipType.friend_of: "amigo de",
    RelationshipType.other: "conocido de",
}

_GENERAL_REPLY = "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."
_UNKNOWN_REPLY = "No he entendido bien lo que has dicho. ¿Puedes repetirlo?"
_EVENT_ASK_WHOSE = "¿De quién quieres saber sus eventos?"

_WEEKDAYS = (
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
)
_MONTHS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)


class DeterministicResponder(Responder):
    async def respond(self, context: AnswerContext) -> Response:
        if context.intent == Intent.general_chat:
            return Response(text=_GENERAL_REPLY)
        if context.intent == Intent.orientation_query:
            return self._respond_orientation_query(context)
        if context.intent == Intent.event_query:
            return self._respond_event_query(context)
        if context.intent == Intent.memory_query:
            return self._respond_memory_query(context)
        if context.intent == Intent.unknown:
            return Response(text=_UNKNOWN_REPLY)
        if context.intent != Intent.person_query:
            return Response(text=_UNKNOWN_REPLY)
        return self._respond_person_query(context)

    def _respond_person_query(self, context: AnswerContext) -> Response:
        if context.resolution == ResolutionKind.not_found:
            return Response(text=f"No tengo guardado quién era {context.term}.")

        if context.resolution == ResolutionKind.ambiguous:
            return self._ambiguous_response(context)

        if context.person is None:
            return Response(text=f"No tengo guardado quién era {context.term}.")

        display = context.person.display_name

        if context.verified_relations:
            hints = ", ".join(
                self._relation_phrase(item) for item in context.verified_relations
            )
            return Response(
                text=f"{display} {hints}.",
                sources=[
                    ResponseSourceRef("person", context.person.id, "subject"),
                    *[
                        ResponseSourceRef(
                            "relationship", item.entity_id, item.epistemic.value
                        )
                        for item in context.verified_relations
                    ],
                ],
            )

        if context.probable_relations:
            hints = " ".join(
                self._probable_relation_phrase(item)
                for item in context.probable_relations
            )
            return Response(
                text=f"No tengo claro exactamente quién era {display}. {hints}",
                sources=[
                    ResponseSourceRef("person", context.person.id, "subject"),
                    *[
                        ResponseSourceRef(
                            "relationship", item.entity_id, item.epistemic.value
                        )
                        for item in context.probable_relations
                    ],
                ],
            )

        verified_facts = [
            item
            for item in context.facts
            if item.epistemic == EpistemicKind.verified_fact
        ]
        probable_facts = [
            item
            for item in context.facts
            if item.epistemic == EpistemicKind.probable_fact
        ]
        if verified_facts or probable_facts:
            return self._facts_response(
                context, display, verified_facts, probable_facts
            )

        photos = [
            item for item in context.media if item.media_type == _PHOTO_MEDIA_TYPE
        ]
        if photos:
            return self._photo_response(context, display, photos[0])

        if context.memories:
            lines = " ".join(
                f"Recuerdo guardado: '{item.narrative}'" for item in context.memories
            )
            return Response(
                text=(
                    f"No tengo claro exactamente quién era {display}, "
                    f"pero aparece en algún recuerdo. {lines}"
                ),
                sources=[
                    ResponseSourceRef("person", context.person.id, "subject"),
                    *[
                        ResponseSourceRef(
                            "memory", item.entity_id, item.epistemic.value
                        )
                        for item in context.memories
                    ],
                ],
            )

        return Response(
            text=f"Tengo guardado a {display}, pero todavía no tengo detalles.",
            sources=[ResponseSourceRef("person", context.person.id, "subject")],
        )

    @staticmethod
    def _facts_response(context, display, verified_facts, probable_facts) -> Response:
        chunks = [
            f"Dato: '{item.text}'"
            for item in verified_facts
        ]
        chunks += [
            f"Dato probable: '{item.text}'"
            for item in probable_facts
        ]
        used = [*verified_facts, *probable_facts]
        return Response(
            text=(
                f"No tengo claro exactamente quién era {display}, "
                f"pero tengo algún dato guardado. {' '.join(chunks)}"
            ),
            sources=[
                ResponseSourceRef("person", context.person.id, "subject"),
                *[
                    ResponseSourceRef("fact", item.entity_id, item.epistemic.value)
                    for item in used
                ],
            ],
        )

    @staticmethod
    def _photo_response(context, display, photo) -> Response:
        clause_parts = []
        if photo.approx_date is not None:
            clause_parts.append(f"de {photo.approx_date.year}")
        if photo.place_text:
            clause_parts.append(f"en {photo.place_text}")
        clause = f" {', '.join(clause_parts)}" if clause_parts else ""
        return Response(
            text=(
                f"No tengo claro exactamente quién era {display}, "
                f"pero tengo guardada una fotografía{clause}."
            ),
            sources=[
                ResponseSourceRef("person", context.person.id, "subject"),
                ResponseSourceRef("media", photo.entity_id, "media"),
            ],
        )

    @staticmethod
    def _relation_phrase(item) -> str:
        label = _RELATION_LABELS.get(item.relationship_type, "relacionado con")
        return f"era {label} {item.other_display_name}"

    @staticmethod
    def _probable_relation_phrase(item) -> str:
        label = _RELATION_LABELS.get(item.relationship_type, "relacionado con")
        return f"Posiblemente era {label} {item.other_display_name}, pero no está confirmado."

    def _respond_event_query(self, context: AnswerContext) -> Response:
        if context.resolution == ResolutionKind.not_found:
            return Response(text=f"No tengo guardado quién es {context.term}.")

        if context.resolution == ResolutionKind.ambiguous:
            return self._ambiguous_response(context)

        if context.term is None or context.person is None:
            return Response(text=_EVENT_ASK_WHOSE)

        display = context.person.display_name
        now = datetime.now(UTC)
        upcoming = [
            event
            for event in context.events
            if event.status in (EventStatus.planned, EventStatus.confirmed)
            and _as_utc(event.start_at) >= now
        ]
        if not upcoming:
            return Response(text=f"No tengo eventos próximos guardados para {display}.")

        phrases = [self._event_phrase(event) for event in upcoming]
        return Response(
            text=f"Para {display}: {' '.join(phrases)}.",
            sources=[
                ResponseSourceRef("person", context.person.id, "subject"),
                *[
                    ResponseSourceRef(
                        "event", event.entity_id, event.status.value
                    )
                    for event in upcoming
                ],
            ],
        )

    def _respond_memory_query(self, context: AnswerContext) -> Response:
        if not context.memories:
            return Response(text="No tengo guardado ese recuerdo todavía.")
        lines = " ".join(
            f"Recuerdo guardado: '{item.narrative}'" for item in context.memories
        )
        return Response(
            text=f"Tengo un recuerdo guardado que encaja con eso: {lines}",
            sources=[
                ResponseSourceRef(
                    "memory", item.entity_id, item.epistemic.value
                )
                for item in context.memories
            ],
        )

    @staticmethod
    def _respond_orientation_query(context: AnswerContext) -> Response:
        now = _as_utc(context.now)
        weekday = _WEEKDAYS[now.weekday()]
        month = _MONTHS[now.month - 1]
        return Response(
            text=(
                f"Hoy es {weekday}, {now.day} de {month} de {now.year} "
                f"y son las {now:%H:%M}."
            )
        )

    @staticmethod
    def _ambiguous_response(context: AnswerContext) -> Response:
        names = ", ".join(candidate.display_name for candidate in context.candidates)
        return Response(
            text=(
                f"Hay varias personas llamadas {context.term}: {names}. "
                "¿A cuál te refieres?"
            ),
            sources=[
                ResponseSourceRef("person", candidate.id, "candidate")
                for candidate in context.candidates
            ],
        )

    @staticmethod
    def _event_phrase(event) -> str:
        when = _as_utc(event.start_at).strftime("%d/%m/%Y")
        if event.status == EventStatus.planned:
            return f"está planificado '{event.title}' para el {when}"
        return f"está confirmado '{event.title}' para el {when}"


def _as_utc(value: datetime) -> datetime:
    """Normalize a datetime to an aware UTC value.

    SQLite stores ``DateTime(timezone=True)`` values without a timezone
    marker and reads them back naive; PostgreSQL returns them aware. Both are
    stored in UTC, so an aware comparison needs the missing marker restored.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
