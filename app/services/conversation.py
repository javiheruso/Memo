import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace

from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.base import ClaimExtractor
from app.claims.contradiction import (
    ContradictionDetector,
    DeterministicContradictionDetector,
)
from app.claims.deterministic import DeterministicClaimExtractor
from app.claims.grouping import ClaimGrouper, DeterministicClaimGrouper
from app.domain.conversation_context import (
    DEFAULT_RESPONDER_HISTORY_TURNS,
    DEFAULT_RESPONDER_MAX_HISTORY_CHARS,
    ConversationStyle,
    ConversationTurn,
    bound_conversation_history,
)
from app.domain.conversation_window import ConversationWindow
from app.domain.enums import Intent
from app.embeddings.base import Embedder
from app.llm.base import Responder, Response
from app.llm.interpreter import (
    DEFAULT_INTERPRETER_HISTORY_TURNS,
    DEFAULT_INTERPRETER_MAX_HISTORY_CHARS,
    DEFAULT_INTERPRETER_MAX_INPUT_CHARS,
    Interpreter,
    InterpreterHistoryTurn,
    bound_interpreter_history,
)
from app.models import ConversationProfile, KnowledgeGap, Person
from app.policies.sensitive_topics import (
    SPONTANEOUS_SUPPRESSION_REPLY,
    SensitiveTopic,
    SensitiveTopicEvaluation,
    answer_leaks_blocked_topic,
    apply_to_context,
    evaluate,
    matches_topic,
)
from app.policies.trust import (
    filter_sensitive_memories,
    is_answerable_fact_status,
)
from app.repositories.conversation import (
    ConversationAccessDenied,
    ConversationRepository,
    ResponseSourceRef,
)
from app.repositories.event import EventRepository
from app.repositories.fact import FactRepository
from app.repositories.knowledge_gap import KnowledgeGapRepository
from app.repositories.media import MediaRepository
from app.repositories.memory import MemoryRepository
from app.repositories.person import PersonRepository
from app.repositories.profile import ConversationProfileRepository
from app.repositories.relationship import RelationshipRepository
from app.retrieval.context_builder import AnswerContext, ContextBuilder
from app.retrieval.disambiguation import DisambiguationResolver
from app.retrieval.entity_resolver import EntityResolver, Resolution, ResolutionKind
from app.retrieval.fact_topics import FactTopic
from app.retrieval.intent_router import IntentRouter, RouteResult
from app.retrieval.interpreter_verifier import (
    InterpreterVerifier,
    VerifiedInterpreterRoute,
)
from app.retrieval.relationship_roles import RelationshipRoleQuery
from app.retrieval.semantic_retriever import SemanticRetriever
from app.retrieval.structured_retriever import PersonData, StructuredRetriever
from app.services.claim_ingestion import ClaimIngestionService

logger = logging.getLogger(__name__)


def _log_interpreter_failure(exc: Exception) -> None:
    """Log only the failure category; never conversation or model text."""
    logger.warning(
        "interpreter verification failed; keeping deterministic routing (category=%s)",
        type(exc).__name__,
    )


_OFFER_MARKERS = re.compile(
    r"\b(?:podemos|puedo|quieres|quiere|te\s+gustar[íi]a|te\s+apetece|"
    r"te\s+interesa|si\s+quieres|te\s+cuento|hablamos\s+de)\b",
    re.IGNORECASE,
)


def _with_grounded_subject(
    context: AnswerContext, sources: list[ResponseSourceRef]
) -> list[ResponseSourceRef]:
    """Keep the verified answer subject citable for pronoun continuity.

    A generative responder may cite the relation, fact or memory it paraphrased
    and forget the person the answer is about. A later "¿Y él?"/"¿Y ella?"
    resolves the person from the previous assistant turn's ``subject`` source,
    so a person/fact answer must always carry its verified subject even when the
    model omitted it. The subject is part of the answer, so this stays truthful;
    it is never invented, because it comes from the resolved ``AnswerContext``.
    """
    person = context.person
    if person is None or not sources or context.intent not in (
        Intent.person_query,
        Intent.fact_query,
    ):
        return sources
    if any(
        source.entity_type == "person" and source.entity_id == person.id
        for source in sources
    ):
        return sources
    return [
        *sources,
        ResponseSourceRef(
            entity_type="person",
            entity_id=person.id,
            usage_type="subject",
        ),
    ]


def _turn_is_offer(text: str) -> bool:
    """True when an assistant turn asked or offered something.

    A question mark or a generic offer phrase is the signal; the exact topic
    words are never interpreted here.
    """
    return (
        "?" in text
        or "¿" in text
        or _OFFER_MARKERS.search(text) is not None
    )


def _recent_turns_touch_sensitive(
    turns, topics: tuple[SensitiveTopic, ...]
) -> bool:
    if not topics:
        return False
    return any(
        matches_topic(turn.text, topic)
        for turn in turns[-2:]
        for topic in topics
    )


def _without_source_ids(data: PersonData, excluded: set[uuid.UUID]) -> PersonData:
    """Drop entities already presented by the previous assistant turn."""
    return replace(
        data,
        relationships=[
            item for item in data.relationships if item.id not in excluded
        ],
        facts=[item for item in data.facts if item.id not in excluded],
        memories=[item for item in data.memories if item.id not in excluded],
        events=[item for item in data.events if item.id not in excluded],
        media=[item for item in data.media if item.id not in excluded],
    )


def _topics_from_profile(
    profile: ConversationProfile | None,
) -> tuple[SensitiveTopic, ...]:
    if profile is None:
        return ()
    return tuple(
        SensitiveTopic(
            topic=topic.topic,
            handling=topic.handling,
            triggers=tuple(topic.triggers or ()),
        )
        for topic in profile.sensitive_topics
    )


def _style_from_profile(
    profile: ConversationProfile | None,
) -> ConversationStyle | None:
    """Return only the verified profile style sent to a responder."""
    if profile is None:
        return None
    return ConversationStyle(
        preferred_name=profile.preferred_name,
        conversational_style=profile.conversational_style,
        positive_topics=tuple(profile.positive_topics or ()),
    )


# Only conversational turns can contribute candidate claims. Retrieval intents
# (person/memory/event/orientation queries) ask the system for information; the
# user's declarative statements live in general chat.
_INGEST_INTENTS = {Intent.general_chat, Intent.unknown}

# A user can accept an offer the assistant made without any stored entity
# behind it. The reply is a stable, honest no-data statement: it never repeats
# the offer and never implies that projects or records exist.
_NO_STORED_CONTINUATION_REPLY = (
    "No tengo nada guardado sobre eso todavía. No quiero inventar detalles."
)


@dataclass(frozen=True)
class _Continuation:
    """Verified continuation of an accepted offer.

    ``no_stored_data`` is a fail-closed path: the service returns a fixed
    honest reply without calling the responder, so a generative model cannot
    restart the offer/acceptance loop.
    """

    route: RouteResult
    context: AnswerContext | None = None
    resolution: Resolution | None = None
    follow_up_topic: str | None = None
    no_stored_data: bool = False


@dataclass(frozen=True)
class ChatResult:
    conversation_id: uuid.UUID
    intent: Intent
    answer: str
    sources: list[ResponseSourceRef] = field(default_factory=list)
    ambiguity: Resolution | None = None
    gap: KnowledgeGap | None = None
    gap_created: bool = False


class ConversationService:
    """Orchestrates the baseline query pipeline end to end.

    Flow: intent routing -> entity resolution -> structured retrieval /
    semantic memory retrieval -> context building (policies) -> response ->
    one atomic persistence unit for the user message, assistant reply, response
    sources and any turn-derived knowledge gap. On conversational turns,
    candidate claims are then extracted from the stored user message and
    persisted append-only outside that unit, without changing the response.
    """

    def __init__(
        self,
        session: AsyncSession,
        responder: Responder,
        embedder: Embedder,
        claim_extractor: ClaimExtractor | None = None,
        claim_grouper: ClaimGrouper | None = None,
        claim_contradiction_detector: ContradictionDetector | None = None,
        interpreter: Interpreter | None = None,
    ) -> None:
        self._session = session
        self._router = IntentRouter()
        self._people = PersonRepository(session)
        self._resolver = EntityResolver(self._people)
        self._memory_repo = MemoryRepository(session)
        self._facts = FactRepository(session)
        self._relationships = RelationshipRepository(session)
        self._structured_retriever = StructuredRetriever(
            person_repo=self._people,
            relationship_repo=self._relationships,
            fact_repo=self._facts,
            memory_repo=self._memory_repo,
            event_repo=EventRepository(session),
            media_repo=MediaRepository(session),
        )
        self._semantic_retriever = SemanticRetriever(
            embedder=embedder,
            memory_repo=self._memory_repo,
        )
        self._context_builder = ContextBuilder()
        self._responder = responder
        self._conversations = ConversationRepository(session)
        self._disambiguation = DisambiguationResolver(PersonRepository(session))
        self._gaps = KnowledgeGapRepository(session)
        self._profiles = ConversationProfileRepository(session)
        self._interpreter = interpreter
        self._interpreter_verifier = InterpreterVerifier(self._resolver)
        self._claim_extractor = claim_extractor or DeterministicClaimExtractor()
        self._claim_ingestion = ClaimIngestionService(
            session,
            grouper=claim_grouper or DeterministicClaimGrouper(),
            contradiction_detector=(
                claim_contradiction_detector or DeterministicContradictionDetector()
            ),
        )

    @asynccontextmanager
    async def _turn_transaction(self) -> AsyncIterator[None]:
        """Make the turn's persistence atomic.

        The user message, assistant response, response sources and any gap
        derived from the turn are written without intermediate commits and
        committed together. If any step raises, the whole turn is rolled back,
        so a stored user message can never exist without its reply (or an
        assistant reply without its sources). Claim ingestion is intentionally
        outside this boundary: it is fail-safe and independently persisted
        after the turn, as the domain documents.
        """
        try:
            yield
            await self._conversations.commit_turn()
        except Exception:
            await self._conversations.rollback_turn()
            raise

    async def handle_user_message(
        self,
        text: str,
        conversation_id: uuid.UUID | None = None,
        *,
        owner_session_id: str | None = None,
    ) -> ChatResult:
        route = self._router.route(text)
        profile = await self._profiles.get_active()
        sensitive_topics = _topics_from_profile(profile)
        sensitive_evaluation = evaluate(text, sensitive_topics)
        positive_topics = (
            tuple(profile.positive_topics or ()) if profile is not None else ()
        )

        # Load the bounded continuity snapshot before the write unit. This keeps
        # the optional interpreter network call outside the turn transaction,
        # and every later continuity decision derives from this same snapshot.
        window = await self._load_turn_window(conversation_id, owner_session_id)
        responder_history = self._responder_history(window, sensitive_topics)
        style = _style_from_profile(profile)

        interpreter_route: VerifiedInterpreterRoute | None = None
        if self._should_interpret(text, route, window, sensitive_evaluation):
            interpreter_route = await self._interpret(text, window, sensitive_topics)

        async with self._turn_transaction():
            conversation = await self._conversations.get_or_create(
                conversation_id,
                owner_session_id=owner_session_id,
                commit=False,
            )
            if window is None:
                window = ConversationWindow(conversation_id=conversation.id)

            pending = window.pending_person_candidates()
            if pending:
                person = await self._disambiguation.resolve(text, pending)
                if person is not None:
                    return await self._disambiguation_turn(
                        conversation.id,
                        text,
                        person,
                        sensitive_evaluation,
                        positive_topics,
                        history=responder_history,
                        style=style,
                    )

            # A short acceptance ("sí", "claro", "cuéntame", "ese mismo") or a
            # topic selection ("sobre costura") only means something when the
            # immediately previous assistant turn offered or asked something.
            # The continuation is always built from stored sources cited by
            # that turn, never from the offer text itself.
            continuation: _Continuation | None = None
            if route.intent in (Intent.general_chat, Intent.unknown):
                continuation = await self._accepted_offer_continuation(
                    text, window, sensitive_topics
                )

            # Elliptical follow-ups ("¿Y de comer?", "¿Y dónde vivió?") keep the
            # subject of the immediately previous resolved fact answer. The
            # repository refuses to carry a subject across any other intent or
            # an ambiguous answer, so this never guesses after unrelated turns
            # or multiple candidate persons.
            follow_up_resolution: Resolution | None = None
            follow_up_topic: str | None = None
            interpreter_applied = False
            if continuation is None and route.intent in (
                Intent.general_chat,
                Intent.event_query,
            ):
                follow_up_route = self._router.route_fact_follow_up(text)
                if follow_up_route is not None:
                    subject_id = window.last_fact_subject()
                    if subject_id is not None:
                        subject = await self._people.get(subject_id)
                        if subject is not None:
                            follow_up_resolution = Resolution.from_people(
                                subject.display_name, [subject]
                            )
                            route = follow_up_route
                else:
                    # Possessive follow-ups ("Háblame sobre su infancia") keep
                    # the immediately previous resolved person, whether that
                    # answer was a person or fact query. The repository returns
                    # None after unrelated or ambiguous turns, so no subject
                    # leaks into a new topic.
                    possessive_route = self._router.route_possessive_follow_up(text)
                    if possessive_route is not None:
                        subject_id = window.last_resolved_person()
                        if subject_id is not None:
                            subject = await self._people.get(subject_id)
                            if subject is not None:
                                follow_up_resolution = Resolution.from_people(
                                    subject.display_name, [subject]
                                )
                                follow_up_topic = possessive_route.follow_up_topic
                                route = possessive_route
                    else:
                        # Pronoun follow-ups ("¿Y ella?", "¿Qué sabes de él?")
                        # also need an immediately previous resolved person.
                        # After an ambiguous or unrelated turn there is none, so
                        # the turn stays conversational and nothing leaks.
                        pronoun_route = self._router.route_pronoun_follow_up(text)
                        if pronoun_route is not None:
                            subject_id = window.last_resolved_person()
                            if subject_id is not None:
                                subject = await self._people.get(subject_id)
                                if subject is not None:
                                    follow_up_resolution = Resolution.from_people(
                                        subject.display_name, [subject]
                                    )
                                    route = pronoun_route

            # Only a turn the deterministic router and carry-over rules left
            # unresolved may use the verified interpreter route. The explicit
            # follow-up above always wins.
            if (
                continuation is None
                and interpreter_route is not None
                and follow_up_resolution is None
                and route.intent in (Intent.general_chat, Intent.unknown)
            ):
                route = interpreter_route.route
                follow_up_resolution = interpreter_route.resolution
                follow_up_topic = interpreter_route.route.follow_up_topic
                interpreter_applied = True

            resolution: Resolution | None = None
            context: AnswerContext
            if continuation is not None:
                route = continuation.route
                context = continuation.context or AnswerContext(intent=route.intent)
                if continuation.context is not None:
                    context = replace(context, accepted_offer=True)
                resolution = continuation.resolution
                follow_up_topic = continuation.follow_up_topic
            elif route.relationship_role is not None:
                context = await self._relationship_query_context(
                    route.relationship_role, window
                )
            elif route.intent == Intent.memory_query:
                semantic = await self._semantic_retriever.search(text)
                context = self._context_builder.build_from_semantic_memories(
                    semantic, term=route.term
                )
            elif route.intent in (
                Intent.person_query,
                Intent.event_query,
                Intent.fact_query,
            ) and (route.term or follow_up_resolution):
                resolution = follow_up_resolution or await self._resolver.resolve(
                    route.term
                )
                if resolution.kind == ResolutionKind.resolved:
                    if route.intent == Intent.fact_query:
                        data = await self._structured_retriever.retrieve_facts(
                            resolution.person.id,
                            topic=route.fact_topic or FactTopic.general,
                        )
                    else:
                        data = await self._structured_retriever.retrieve(
                            resolution.person.id
                        )
                    context = self._context_builder.build_from_person_data(
                        data, resolution, intent=route.intent
                    )
                else:
                    context = self._context_builder.build_from_resolution(
                        resolution, route.intent
                    )
            else:
                context = AnswerContext(intent=route.intent, term=route.term)

            if follow_up_topic is not None:
                context = replace(context, follow_up_topic=follow_up_topic)

            context = apply_to_context(
                context, sensitive_evaluation, positive_topics=positive_topics
            )
            if continuation is not None and continuation.no_stored_data:
                # Fail closed: a generative responder must never be able to
                # turn a baseless offer into a repeated offer/acceptance loop.
                answer = Response(text=_NO_STORED_CONTINUATION_REPLY)
            else:
                answer = await self._respond(
                    context,
                    sensitive_evaluation,
                    conversation.id,
                    message=text,
                    history=responder_history,
                    style=style,
                )
            answer = replace(
                answer, sources=_with_grounded_subject(context, answer.sources)
            )

            user_message = await self._conversations.add_message(
                conversation.id,
                speaker="user",
                text=text,
                intent=route.intent.value,
                commit=False,
            )
            user_message_id = user_message.id
            assistant_message = await self._conversations.add_message(
                conversation.id,
                speaker="assistant",
                text=answer.text,
                intent=route.intent.value,
                commit=False,
            )
            if answer.sources:
                await self._conversations.add_response_sources(
                    assistant_message.id, answer.sources, commit=False
                )

            gap: KnowledgeGap | None = None
            gap_created = False
            # A verified interpreter route never reaches ``not_found`` (unknown
            # names are discarded), and this guard makes that invariant explicit:
            # an interpreter suggestion can never open a knowledge gap.
            if (
                route.relationship_role is None
                and not interpreter_applied
                and context.resolution == ResolutionKind.not_found
                and route.term
            ):
                gap, gap_created = await self._gaps.upsert(
                    gap_type="person_unknown",
                    canonical_reference=route.term.lower(),
                    conversation_id=conversation.id,
                    description=f"User asked about unknown person '{route.term}'.",
                    commit=False,
                )

        # Claim ingestion is fail-safe and intentionally independent from the
        # turn transaction: failures are caught and can never undo the reply.
        if route.intent in _INGEST_INTENTS:
            await self._ingest_claims(text, conversation.id, user_message_id)

        return ChatResult(
            conversation_id=conversation.id,
            intent=route.intent,
            answer=answer.text,
            sources=answer.sources,
            ambiguity=(
                resolution
                if resolution is not None
                and resolution.kind == ResolutionKind.ambiguous
                else None
            ),
            gap=gap,
            gap_created=gap_created,
        )

    async def _accepted_offer_continuation(
        self,
        text: str,
        window: ConversationWindow | None,
        sensitive_topics: tuple[SensitiveTopic, ...],
    ) -> _Continuation | None:
        """Continue a previous offer from stored sources, or fail closed.

        Continuation facts, memories and people come from the sources the
        previous answer actually cited. The offer text is not authority. When
        no cited source yields verified context, the caller returns an honest
        no-data reply without invoking a generative responder.
        """
        if window is None or not self._router.is_continuation_reply(text):
            return None
        offer_turn = window.latest_assistant_turn
        if offer_turn is None:
            return None
        memory_continuation = (
            offer_turn.intent == Intent.memory_query.value
            and bool(offer_turn.sources)
        )
        if not (_turn_is_offer(offer_turn.text) or memory_continuation):
            return None
        # An ambiguity is unresolved until the person names a candidate; never
        # let an acceptance pick one of them.
        if window.pending_person_candidates():
            return None
        # Never carry an acceptance across a sensitive-topic boundary.
        if _recent_turns_touch_sensitive(window.turns, sensitive_topics):
            return None

        sources = offer_turn.sources
        excluded = {source.entity_id for source in sources}

        memory_ids = [
            source.entity_id
            for source in sources
            if source.entity_type == "memory"
        ]
        if memory_ids:
            memories = []
            filtered = 0
            for memory_id in memory_ids:
                memory = await self._memory_repo.get(memory_id)
                if memory is None:
                    continue
                safe, blocked = filter_sensitive_memories([memory])
                filtered += len(blocked)
                memories.extend(safe)
            if memories:
                return _Continuation(
                    route=RouteResult(intent=Intent.memory_query),
                    context=self._context_builder.build_from_memories(
                        memories, filtered_count=filtered
                    ),
                )

        fact_ids = [
            source.entity_id
            for source in sources
            if source.entity_type == "fact"
        ]
        if fact_ids:
            facts = [
                fact
                for fact_id in fact_ids
                if (fact := await self._facts.get(fact_id)) is not None
                and is_answerable_fact_status(fact.status)
            ]
            if facts:
                subject = await self._people.get(facts[0].subject_person_id)
                if subject is not None:
                    resolution = Resolution.from_people(
                        subject.display_name, [subject]
                    )
                    context = self._context_builder.build_from_person_data(
                        PersonData(
                            person=subject,
                            relationships=[],
                            facts=facts,
                            memories=[],
                            events=[],
                            media=[],
                        ),
                        resolution,
                        intent=Intent.fact_query,
                    )
                    return _Continuation(
                        route=RouteResult(
                            intent=Intent.fact_query,
                            term=subject.display_name,
                            fact_topic=FactTopic.general,
                        ),
                        context=context,
                        resolution=resolution,
                    )

        person_ids = [
            source.entity_id
            for source in sources
            if source.entity_type == "person"
            and source.usage_type == "subject"
        ] or [
            source.entity_id
            for source in sources
            if source.entity_type == "person"
        ]
        if person_ids:
            person = await self._people.get(person_ids[0])
            if person is not None:
                data = await self._structured_retriever.retrieve(person.id)
                data = _without_source_ids(data, excluded)
                safe_memories, _ = filter_sensitive_memories(data.memories)
                data = replace(data, memories=safe_memories)
                if (
                    data.relationships
                    or data.facts
                    or data.memories
                    or data.events
                    or data.media
                ):
                    resolution = Resolution.from_people(
                        person.display_name, [person]
                    )
                    return _Continuation(
                        route=RouteResult(
                            intent=Intent.person_query,
                            term=person.display_name,
                        ),
                        context=self._context_builder.build_from_person_data(
                            data, resolution, intent=Intent.person_query
                        ),
                        resolution=resolution,
                    )

        # The offer itself is untrusted text, not a stored source. Without a
        # cited source there is no verified context to continue from.
        return _Continuation(
            route=RouteResult(intent=Intent.general_chat),
            no_stored_data=True,
        )

    async def _load_turn_window(
        self,
        conversation_id: uuid.UUID | None,
        owner_session_id: str | None,
    ) -> ConversationWindow | None:
        """Load the bounded continuity snapshot before the write unit.

        ``None`` means there is no existing conversation to read: either a new
        conversation was requested, or a non-owned/missing id will be handled by
        ``get_or_create`` exactly as before. A foreign conversation raises the
        same access error as the existing ownership checks.
        """
        if conversation_id is None:
            return None
        window = await self._conversations.load_window(
            conversation_id, owner_session_id=owner_session_id
        )
        if window is None and owner_session_id is not None:
            raise ConversationAccessDenied
        return window

    async def _relationship_query_context(
        self,
        role_query: RelationshipRoleQuery,
        window: ConversationWindow | None,
    ) -> AnswerContext:
        """Answer an explicit role question from the current active person.

        The active person is the subject of the immediately previous resolved
        person/fact turn. When there is none, the context has no subject and
        the responder explicitly avoids guessing. Relationships come from one
        person's stored rows only; rejected rows are filtered by the repository
        and pending rows are withheld by the context builder.
        """
        active_person_id = window.last_resolved_person() if window is not None else None
        if active_person_id is None:
            return self._context_builder.build_from_relationship_query(
                None, role_query.role, []
            )

        active_person = await self._people.get(active_person_id)
        if active_person is None:
            return self._context_builder.build_from_relationship_query(
                None, role_query.role, []
            )

        relationships = await self._relationships.for_person(active_person.id)
        matching = [
            relationship
            for relationship in relationships
            if relationship.relationship_type in role_query.relationship_types
        ]
        return self._context_builder.build_from_relationship_query(
            active_person, role_query.role, matching
        )

    def _should_interpret(
        self,
        text: str,
        route: RouteResult,
        window: ConversationWindow | None,
        sensitive_evaluation: SensitiveTopicEvaluation,
    ) -> bool:
        """True only for an unresolved, safe, bounded conversational turn.

        Already-resolved deterministic routes, pending ambiguity, resolved
        carry-over, pronoun-only follow-ups, configured sensitive-topic
        requests, and empty or oversized text never reach the optional network
        call.
        """
        if self._interpreter is None:
            return False
        if route.intent not in (Intent.general_chat, Intent.unknown):
            return False
        max_input_chars = getattr(
            self._interpreter,
            "max_input_chars",
            DEFAULT_INTERPRETER_MAX_INPUT_CHARS,
        )
        normalized = text.strip()
        if not normalized or len(normalized) > max_input_chars:
            return False
        if sensitive_evaluation.requested is not None:
            return False

        # A pronoun-only follow-up is resolved from verified carry-over or
        # intentionally left conversational. It contains no name for the
        # interpreter to ground, so an interpreter call cannot help; skipping
        # it also prevents "él"/"ella" from being re-read as a person name.
        if self._router.route_pronoun_follow_up(normalized) is not None:
            return False

        if window is None:
            return True
        if window.pending_person_candidates():
            return False

        follow_up = self._router.route_fact_follow_up(normalized)
        if follow_up is not None:
            return window.last_fact_subject() is None
        possessive = self._router.route_possessive_follow_up(normalized)
        if possessive is not None:
            return window.last_resolved_person() is None
        return True

    def _responder_history(
        self,
        window: ConversationWindow | None,
        sensitive_topics: tuple[SensitiveTopic, ...],
    ) -> tuple[ConversationTurn, ...]:
        """Build a bounded, current-conversation-only, re-filtered history.

        Only role and text are carried. Turns matching any configured sensitive
        topic are dropped so a blocked topic never reaches a responder through
        conversational context, even though the transcript was already shown to
        the person. The snapshot never includes ids, sources or stored records,
        so no history item can become a fact or a source.
        """
        if window is None:
            return ()
        turns = [
            ConversationTurn(role=turn.speaker, text=turn.text)
            for turn in window.turns
            if turn.speaker in ("user", "assistant")
            and not any(matches_topic(turn.text, topic) for topic in sensitive_topics)
        ]
        return bound_conversation_history(
            turns,
            max_turns=DEFAULT_RESPONDER_HISTORY_TURNS,
            max_chars=DEFAULT_RESPONDER_MAX_HISTORY_CHARS,
        )

    def _interpreter_history(
        self,
        window: ConversationWindow | None,
        sensitive_topics: tuple[SensitiveTopic, ...],
    ) -> tuple[InterpreterHistoryTurn, ...]:
        """Build a bounded, current-conversation-only, re-filtered history.

        Only role and text are carried. Turns matching any configured sensitive
        topic are dropped again here, even though the transcript was already
        shown to the user.
        """
        if window is None or self._interpreter is None:
            return ()
        turns = [
            InterpreterHistoryTurn(role=turn.speaker, text=turn.text)
            for turn in window.turns
            if turn.speaker in ("user", "assistant")
            and not any(matches_topic(turn.text, topic) for topic in sensitive_topics)
        ]
        return bound_interpreter_history(
            turns,
            max_turns=getattr(
                self._interpreter,
                "history_turns",
                DEFAULT_INTERPRETER_HISTORY_TURNS,
            ),
            max_chars=getattr(
                self._interpreter,
                "max_history_chars",
                DEFAULT_INTERPRETER_MAX_HISTORY_CHARS,
            ),
        )

    async def _interpret(
        self,
        text: str,
        window: ConversationWindow | None,
        sensitive_topics: tuple[SensitiveTopic, ...],
    ) -> VerifiedInterpreterRoute | None:
        """Ask the interpreter once and verify its suggestion.

        Any provider, schema or verification failure returns ``None`` so the
        deterministic route handles the turn unchanged.
        """
        if self._interpreter is None:
            return None
        history = self._interpreter_history(window, sensitive_topics)
        try:
            suggestion = await self._interpreter.suggest(text.strip(), history=history)
        except Exception as exc:  # noqa: BLE001 - optional feature must fail closed
            _log_interpreter_failure(exc)
            return None
        if suggestion is None:
            return None
        try:
            return await self._interpreter_verifier.verify(suggestion, message=text)
        except Exception as exc:  # noqa: BLE001 - verification must fail closed
            _log_interpreter_failure(exc)
            return None

    async def _disambiguation_turn(
        self,
        conversation_id: uuid.UUID,
        text: str,
        person: Person,
        sensitive_evaluation: SensitiveTopicEvaluation,
        positive_topics: tuple[str, ...],
        *,
        history: tuple[ConversationTurn, ...] = (),
        style: ConversationStyle | None = None,
    ) -> ChatResult:
        """Answer a follow-up that selected one pending ambiguity candidate.

        The user is picking a person already listed by the previous assistant
        turn, not making a new query or a declarative statement, so neither
        claim ingestion nor a knowledge-gap upsert runs for this turn. The
        caller owns the turn transaction; this method never commits.
        """
        await self._conversations.add_message(
            conversation_id,
            speaker="user",
            text=text,
            intent=Intent.person_query.value,
            commit=False,
        )
        resolution = Resolution.from_people(person.display_name, [person])
        data = await self._structured_retriever.retrieve(person.id)
        context = self._context_builder.build_from_person_data(
            data, resolution, intent=Intent.person_query
        )
        context = apply_to_context(
            context, sensitive_evaluation, positive_topics=positive_topics
        )
        answer = await self._respond(
            context,
            sensitive_evaluation,
            conversation_id,
            message=text,
            history=history,
            style=style,
        )
        answer = replace(
            answer, sources=_with_grounded_subject(context, answer.sources)
        )
        assistant_message = await self._conversations.add_message(
            conversation_id,
            speaker="assistant",
            text=answer.text,
            intent=Intent.person_query.value,
            commit=False,
        )
        if answer.sources:
            await self._conversations.add_response_sources(
                assistant_message.id, answer.sources, commit=False
            )
        return ChatResult(
            conversation_id=conversation_id,
            intent=Intent.person_query,
            answer=answer.text,
            sources=answer.sources,
        )

    async def _respond(
        self,
        context: AnswerContext,
        sensitive_evaluation: SensitiveTopicEvaluation,
        conversation_id: uuid.UUID,
        *,
        message: str | None = None,
        history: tuple[ConversationTurn, ...] = (),
        style: ConversationStyle | None = None,
    ) -> Response:
        """Run the responder and enforce non-initiation on its output.

        Context filtering already withholds blocked-topic items from a
        spontaneous turn. This final guard handles a responder that ignores the
        context or the prompt: if its text mentions a blocked topic anyway, the
        answer is replaced with a safe generic reply and its sources are
        dropped, so no trace of the topic reaches the person. Conversation
        history is passed separately from the facts/sources in ``context``.
        """
        answer = await self._responder.respond(
            context, message=message, history=history, style=style
        )
        if answer_leaks_blocked_topic(answer.text, sensitive_evaluation):
            logger.warning(
                "responder output in conversation %s mentioned a blocked "
                "sensitive topic; replacing it with a safe reply",
                conversation_id,
            )
            return Response(text=SPONTANEOUS_SUPPRESSION_REPLY)
        return answer

    async def _ingest_claims(
        self, text: str, conversation_id: uuid.UUID, message_id: uuid.UUID
    ) -> None:
        """Persist and group candidate claims for a conversational user message.

        Fail-safe by design: any extraction, persistence or grouping error is
        caught and logged so it can never change the response behavior or break
        the turn. Claims are stored append-only as ``candidate``; nothing here
        promotes them to facts or memories, and a grouping failure leaves the
        claims persisted but ungrouped.
        """
        try:
            candidates = self._claim_extractor.extract(text)
            if candidates:
                await self._claim_ingestion.ingest(
                    candidates,
                    conversation_id=conversation_id,
                    message_id=message_id,
                )
        except Exception as exc:
            # Keep the session usable for later work even when the independent
            # claim unit fails; rows committed before the failure stay durable.
            try:
                await self._session.rollback()
            except Exception:
                logger.exception(
                    "claim ingestion rollback failed after message %s", message_id
                )
            logger.warning(
                "claim ingestion (extraction/persistence/grouping) failed for "
                "message %s (%s); skipping ingestion",
                message_id,
                type(exc).__name__,
            )
