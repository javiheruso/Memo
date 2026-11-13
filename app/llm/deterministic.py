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

import difflib
import re
from collections.abc import Sequence
from datetime import UTC, datetime

from app.domain.conversation_context import ConversationStyle, ConversationTurn
from app.domain.enums import (
    EpistemicKind,
    EventStatus,
    Intent,
    RelationshipType,
    SourceType,
)
from app.llm.base import Responder, Response
from app.policies.positive_topics import (
    is_open_topic_request,
    mentioned_topics,
)
from app.policies.trust import (
    is_answerable_fact_status,
    is_answerable_relationship_status,
)
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
_FACT_ASK_WHOSE = "¿De quién quieres saber ese dato?"

# Short deterministic acknowledgements for conversational turns. They never
# mention storage, system internals, or invented data, and the continued-chat
# reply deliberately avoids greeting or restarting the conversation on every
# turn when the responder has recent history.
_AFFIRMATION_REPLY = "Vale."
_NEGATION_REPLY = "Entiendo."
_THANKS_REPLY = "De nada."
_FAREWELL_REPLY = "Hasta luego."
_PERSONAL_COMMENT_REPLY = "Gracias por contármelo."
_CONTINUE_REPLY = "Te escucho."

_AFFIRMATION = re.compile(
    r"^\s*[¡¿]?\s*(?:s[íi]|claro|vale|de acuerdo|perfecto|bueno|ok|okey|ajá|aja)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"^\s*[¡¿]?\s*(?:no|nada|para nada|ni hablar)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_THANKS = re.compile(r"\b(?:gracias|te lo agradezco)\b", re.IGNORECASE)
_FAREWELL = re.compile(
    r"\b(?:adi[oó]s|hasta luego|hasta pronto|nos vemos)\b", re.IGNORECASE
)
# A personal comment is a first-person statement the deterministic responder
# can acknowledge without inventing anything. It is intentionally narrow: a
# third-person recollection ("Vivíamos en Madrid") stays general chat instead of
# being acknowledged as a present feeling.
_PERSONAL_COMMENT = re.compile(
    r"\b(?:me|mi|mis|estoy|tengo|quiero|siento|creo|necesito|prefiero|he)\b",
    re.IGNORECASE,
)
_TOPIC_REJECTION = re.compile(
    r"\b(?:no\s+me\s+(?:apetece|gusta|interesa)|prefiero\s+no|"
    r"no\s+quiero|dejemos|cambiemos\s+de\s+tema|otro\s+tema)\b",
    re.IGNORECASE,
)
_INVENT_MEMORY_REQUEST = re.compile(
    r"\b(?:inv[eé]ntate|inventa(?:te|r|s|e)?|invento)\b[^.!?]*\brecuerd",
    re.IGNORECASE,
)
_DO_YOU_KNOW = re.compile(
    r"^\s*[¿¡]?\s*(?:t[uú]\s+)?(?:sabes|sabe)\s+"
    r"(?:si|qu[eé]|qui[eé]n|cu[aá]ndo|d[oó]nde|c[oó]mo)\b",
    re.IGNORECASE,
)
_TOPIC_SUGGESTIONS = (
    "Podemos hablar del tiempo, de la comida o de lo que tú prefieras.",
    "¿Y si hablamos de música o de alguna película que te guste?",
    "Podemos hablar de recetas, de viajes o de lo que te apetezca.",
)


def is_farewell_message(text: str) -> bool:
    """True for a short spoken farewell."""
    return _FAREWELL.search(text) is not None


def is_invent_memory_request(text: str) -> bool:
    """True when the person explicitly asks for an invented memory."""
    return _INVENT_MEMORY_REQUEST.search(text) is not None


def is_do_you_know_question(text: str) -> bool:
    """True when the person asks whether the assistant knows a personal fact."""
    return _DO_YOU_KNOW.search(text) is not None


def _topic_suggestion(history: Sequence[ConversationTurn]) -> str:
    """Pick a neutral topic suggestion the assistant has not already used."""
    previous = [turn.text for turn in history if turn.role == "assistant"]
    for candidate in _TOPIC_SUGGESTIONS:
        if not any(
            difflib.SequenceMatcher(
                None, candidate.lower(), text.lower()
            ).ratio()
            >= 0.9
            for text in previous
        ):
            return candidate
    return _TOPIC_SUGGESTIONS[0]


_SOURCE_LABELS: dict[str, str] = {
    SourceType.user.value: "lo que me contó la propia persona",
    SourceType.family_member.value: "un familiar",
    SourceType.conversation.value: "una conversación registrada",
    SourceType.document.value: "un documento",
    SourceType.photo.value: "una fotografía",
    SourceType.calendar.value: "un calendario",
    SourceType.system.value: "datos de ejemplo",
    SourceType.other.value: "otra fuente",
}

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
    async def respond(
        self,
        context: AnswerContext,
        *,
        message: str | None = None,
        history: Sequence[ConversationTurn] = (),
        style: ConversationStyle | None = None,
    ) -> Response:
        if context.sensitive_topic_requested is not None:
            return self._sensitive_topic_response(context, message=message)
        if context.intent == Intent.general_chat:
            return self._respond_general_chat(
                message=message, history=history, style=style
            )
        if context.intent == Intent.orientation_query:
            return self._respond_orientation_query(context)
        if context.intent == Intent.event_query:
            return self._respond_event_query(context)
        if context.intent == Intent.memory_query:
            return self._respond_memory_query(context)
        if context.intent == Intent.fact_query:
            return self._respond_fact_query(context)
        if context.intent == Intent.unknown:
            return Response(text=_UNKNOWN_REPLY)
        if context.intent != Intent.person_query:
            return Response(text=_UNKNOWN_REPLY)
        return self._respond_person_query(context)

    @staticmethod
    def _respond_general_chat(
        *,
        message: str | None,
        history: Sequence[ConversationTurn],
        style: ConversationStyle | None,
    ) -> Response:
        """Answer a conversational turn briefly and naturally.

        The reply depends only on the current message and whether the
        conversation already has history. It never greets or restarts the
        conversation on every turn, never invents data, and never mentions
        storage or system internals. ``style`` is accepted for parity with the
        provider responder; the deterministic copy stays short and does not
        require a configured name.
        """
        text = (message or "").strip()
        if not text:
            return Response(text=_GENERAL_REPLY)
        if _THANKS.search(text):
            return Response(text=_THANKS_REPLY)
        if _FAREWELL.search(text):
            return Response(text=_FAREWELL_REPLY)
        if _INVENT_MEMORY_REQUEST.search(text):
            return Response(
                text=(
                    "No puedo inventar recuerdos tuyos. "
                    "Prefiero no inventar detalles."
                )
            )
        if style is not None and style.positive_topics and not _TOPIC_REJECTION.search(
            text
        ):
            chosen = mentioned_topics(text, style.positive_topics)
            if chosen:
                return Response(
                    text=f"¿Qué es lo que más te gusta de {chosen[0]}?"
                )
        if is_open_topic_request(text):
            return Response(text=_topic_suggestion(history))
        if _DO_YOU_KNOW.search(text):
            return Response(text="No, no tengo información sobre eso.")
        if _TOPIC_REJECTION.search(text):
            return Response(text="Entiendo. Hablamos de otra cosa.")
        if _AFFIRMATION.search(text):
            return Response(text=_AFFIRMATION_REPLY)
        if _NEGATION.search(text):
            return Response(text=_NEGATION_REPLY)
        if _PERSONAL_COMMENT.search(text):
            return Response(text=_PERSONAL_COMMENT_REPLY)
        if history:
            return Response(text=_CONTINUE_REPLY)
        return Response(text=_GENERAL_REPLY)

    @staticmethod
    def _sensitive_topic_response(
        context: AnswerContext, *, message: str | None = None
    ) -> Response:
        """Careful acknowledgement that never invents details.

        The reply names the topic only because the person raised it, states
        that Memo will not invent details, and offers a safer configured
        subject when one exists. It never cites a memory or a fact, so an
        explicit question cannot pull high-sensitivity details into the answer.
        A generic sensitive reference is acknowledged with empathy instead of
        the configured personal label or a profile-topic redirect.
        """
        if context.sensitive_topic_generic:
            if message and ("?" in message or "¿" in message):
                return Response(
                    text=(
                        "No quiero inventar detalles sobre eso. Si te apetece, "
                        "cuéntame cómo te sientes."
                    )
                )
            return Response(
                text=(
                    "Lo siento. No quiero inventar detalles sobre eso; si te "
                    "apetece, cuéntame cómo te sientes."
                )
            )
        topic = context.sensitive_topic_requested
        redirect = context.sensitive_redirect_topic
        if redirect:
            text = (
                f"Entiendo que quieres hablar de {topic}. No quiero inventar "
                "detalles, así que prefiero no extenderme. Si te parece, "
                f"podemos hablar de {redirect}."
            )
        else:
            text = (
                f"Entiendo que quieres hablar de {topic}. No quiero inventar "
                "detalles, así que prefiero no extenderme. ¿Te apetece hablar "
                "de otra cosa?"
            )
        return Response(text=text)

    def _respond_person_query(self, context: AnswerContext) -> Response:
        if context.relationship_role is not None:
            return self._respond_relationship_query(context)

        if context.resolution == ResolutionKind.not_found:
            return Response(text=f"No tengo guardado quién era {context.term}.")

        if context.resolution == ResolutionKind.ambiguous:
            return self._ambiguous_response(context)

        if context.person is None:
            return Response(text=f"No tengo guardado quién era {context.term}.")

        display = context.person.display_name

        verified_relations = [
            item
            for item in context.verified_relations
            if is_answerable_relationship_status(item.status)
        ]
        probable_relations = [
            item
            for item in context.probable_relations
            if is_answerable_relationship_status(item.status)
        ]

        if verified_relations:
            hints = ", ".join(
                self._relation_phrase(item) for item in verified_relations
            )
            return Response(
                text=f"{display} {hints}.",
                sources=[
                    ResponseSourceRef("person", context.person.id, "subject"),
                    *[
                        ResponseSourceRef(
                            "relationship", item.entity_id, item.epistemic.value
                        )
                        for item in verified_relations
                    ],
                ],
            )

        if probable_relations:
            hints = " ".join(
                self._probable_relation_phrase(item)
                for item in probable_relations
            )
            return Response(
                text=f"No tengo claro exactamente quién era {display}. {hints}",
                sources=[
                    ResponseSourceRef("person", context.person.id, "subject"),
                    *[
                        ResponseSourceRef(
                            "relationship", item.entity_id, item.epistemic.value
                        )
                        for item in probable_relations
                    ],
                ],
            )

        verified_facts = [
            item
            for item in context.facts
            if is_answerable_fact_status(item.status)
            and item.epistemic == EpistemicKind.verified_fact
        ]
        probable_facts = [
            item
            for item in context.facts
            if is_answerable_fact_status(item.status)
            and item.epistemic == EpistemicKind.probable_fact
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

    def _respond_relationship_query(self, context: AnswerContext) -> Response:
        """Answer an explicit role question about the active person.

        The related person is only named from stored relationships. A probable
        link is always phrased as uncertain; several distinct related people are
        reported as ambiguous instead of choosing one. Pending and rejected
        rows never reach this method.
        """
        role = context.relationship_role
        if context.person is None:
            return Response(text="Antes de responder, dime de quién estamos hablando.")

        verified = [
            item
            for item in context.verified_relations
            if is_answerable_relationship_status(item.status)
        ]
        probable = [
            item
            for item in context.probable_relations
            if is_answerable_relationship_status(item.status)
        ]
        if not verified and not probable:
            return Response(
                text=f"No tengo guardado quién es tu {role}.",
                sources=[ResponseSourceRef("person", context.person.id, "subject")],
            )

        if verified:
            return self._relationship_answer(context, role, verified, confirmed=True)
        return self._relationship_answer(context, role, probable, confirmed=False)

    @classmethod
    def _relationship_answer(
        cls,
        context: AnswerContext,
        role: str,
        items,
        *,
        confirmed: bool,
    ) -> Response:
        names: list[str] = []
        for item in items:
            if item.other_display_name not in names:
                names.append(item.other_display_name)

        sources = cls._relationship_query_sources(context, items)
        if len(names) > 1:
            listed = ", ".join(names)
            if confirmed:
                text = (
                    f"Tengo guardadas varias personas como tu {role}: {listed}. "
                    "No sé cuál es."
                )
            else:
                text = (
                    f"Tengo anotadas varias personas posibles como tu {role}: "
                    f"{listed}, pero ninguna está confirmada."
                )
            return Response(text=text, sources=sources)

        item = items[0]
        suffix = _source_suffix(item.source_type)
        if confirmed:
            text = f"Tu {role} es {item.other_display_name}{suffix}."
        else:
            text = (
                f"Creo que tu {role} es {item.other_display_name}, "
                f"pero no está confirmado{suffix}."
            )
        return Response(text=text, sources=sources)

    @staticmethod
    def _relationship_query_sources(
        context: AnswerContext, items
    ) -> list[ResponseSourceRef]:
        sources = [ResponseSourceRef("person", context.person.id, "subject")]
        seen: set[tuple[str, object, str | None]] = set()
        for item in items:
            candidates = [
                (
                    "person",
                    item.related_person_id,
                    "related",
                ),
                (
                    "relationship",
                    item.entity_id,
                    item.epistemic.value,
                ),
            ]
            if item.source_id is not None:
                candidates.append(("source", item.source_id, "provenance"))
            for entity_type, entity_id, usage_type in candidates:
                if entity_id is None:
                    continue
                key = (entity_type, entity_id, usage_type)
                if key in seen:
                    continue
                seen.add(key)
                sources.append(ResponseSourceRef(entity_type, entity_id, usage_type))
        return sources

    def _respond_fact_query(self, context: AnswerContext) -> Response:
        """Answer a scoped fact/preference question from stored facts only.

        The retriever already narrowed the facts to the question family. This
        method keeps status and provenance explicit: verified items may be
        stated as confirmed, probable ones are always labelled as uncertain,
        and each item names its stored source when one exists. No relation,
        memory or other tier is consulted, so a fact question cannot drift into
        a different hint tier.
        """
        if context.resolution == ResolutionKind.not_found:
            return Response(text=f"No tengo guardado quién era {context.term}.")

        if context.resolution == ResolutionKind.ambiguous:
            return self._ambiguous_response(context)

        if context.person is None or context.term is None:
            return Response(text=_FACT_ASK_WHOSE)

        display = context.person.display_name
        verified_facts = [
            item
            for item in context.facts
            if is_answerable_fact_status(item.status)
            and item.epistemic == EpistemicKind.verified_fact
        ]
        probable_facts = [
            item
            for item in context.facts
            if is_answerable_fact_status(item.status)
            and item.epistemic == EpistemicKind.probable_fact
        ]
        if not verified_facts and not probable_facts:
            if context.follow_up_topic is not None:
                return Response(
                    text=(
                        f"No sé nada de {context.follow_up_topic} de {display} todavía."
                    ),
                    sources=[ResponseSourceRef("person", context.person.id, "subject")],
                )
            return Response(
                text=f"No tengo guardado ningún dato sobre eso de {display}.",
                sources=[
                    ResponseSourceRef("person", context.person.id, "subject")
                ],
            )

        segments: list[str] = []
        if verified_facts:
            segments.append(
                "tengo como confirmado "
                + self._render_fact_list(verified_facts)
            )
        if probable_facts:
            segments.append(
                "tengo como probable, sin confirmar, "
                + self._render_fact_list(probable_facts)
            )

        return Response(
            text=f"Sobre {display}: {' '.join(segments)}",
            sources=self._fact_query_sources(
                context, [*verified_facts, *probable_facts]
            ),
        )

    @staticmethod
    def _render_fact_list(facts) -> str:
        return "; ".join(
            f"'{item.text}'{_source_suffix(item.source_type)}" for item in facts
        )

    @staticmethod
    def _fact_query_sources(context: AnswerContext, facts) -> list[ResponseSourceRef]:
        sources = [ResponseSourceRef("person", context.person.id, "subject")]
        seen_sources: set = set()
        for item in facts:
            sources.append(
                ResponseSourceRef("fact", item.entity_id, item.epistemic.value)
            )
            if item.source_id is not None and item.source_id not in seen_sources:
                seen_sources.add(item.source_id)
                sources.append(
                    ResponseSourceRef("source", item.source_id, "provenance")
                )
        return sources

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
            f"'{item.narrative}'" for item in context.memories
        )
        return Response(
            text=(
                "No sé si te refieres a esto, pero tengo guardado este "
                f"recuerdo: {lines}"
            ),
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


def _source_suffix(source_type: str | None) -> str:
    label = _SOURCE_LABELS.get(source_type or "")
    return f" (según {label})" if label else ""


def _as_utc(value: datetime) -> datetime:
    """Normalize a datetime to an aware UTC value.

    SQLite stores ``DateTime(timezone=True)`` values without a timezone
    marker and reads them back naive; PostgreSQL returns them aware. Both are
    stored in UTC, so an aware comparison needs the missing marker restored.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
