import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.base import ClaimExtractor
from app.claims.contradiction import ContradictionDetector, DeterministicContradictionDetector
from app.claims.deterministic import DeterministicClaimExtractor
from app.claims.grouping import ClaimGrouper, DeterministicClaimGrouper
from app.domain.enums import Intent
from app.embeddings.base import Embedder
from app.llm.base import Responder, Response
from app.models import ConversationProfile, KnowledgeGap, Person
from app.repositories.conversation import (
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
from app.retrieval.intent_router import IntentRouter
from app.retrieval.semantic_retriever import SemanticRetriever
from app.retrieval.structured_retriever import StructuredRetriever
from app.policies.sensitive_topics import (
    SPONTANEOUS_SUPPRESSION_REPLY,
    SensitiveTopic,
    SensitiveTopicEvaluation,
    answer_leaks_blocked_topic,
    apply_to_context,
    evaluate,
)
from app.services.claim_ingestion import ClaimIngestionService

logger = logging.getLogger(__name__)


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

# Only conversational turns can contribute candidate claims. Retrieval intents
# (person/memory/event/orientation queries) ask the system for information; the
# user's declarative statements live in general chat.
_INGEST_INTENTS = {Intent.general_chat, Intent.unknown}


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
    persistence of the conversation, messages, response sources and knowledge
    gaps. On conversational turns, candidate claims are extracted from the user
    message, persisted append-only and routed into claim groups, all without
    changing the response.
    """

    def __init__(
        self,
        session: AsyncSession,
        responder: Responder,
        embedder: Embedder,
        claim_extractor: ClaimExtractor | None = None,
        claim_grouper: ClaimGrouper | None = None,
        claim_contradiction_detector: ContradictionDetector | None = None,
    ) -> None:
        self._router = IntentRouter()
        self._resolver = EntityResolver(PersonRepository(session))
        self._memory_repo = MemoryRepository(session)
        self._structured_retriever = StructuredRetriever(
            person_repo=PersonRepository(session),
            relationship_repo=RelationshipRepository(session),
            fact_repo=FactRepository(session),
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
        self._claim_extractor = claim_extractor or DeterministicClaimExtractor()
        self._claim_ingestion = ClaimIngestionService(
            session,
            grouper=claim_grouper or DeterministicClaimGrouper(),
            contradiction_detector=(
                claim_contradiction_detector or DeterministicContradictionDetector()
            ),
        )

    async def handle_user_message(
        self, text: str, conversation_id: uuid.UUID | None = None
    ) -> ChatResult:
        route = self._router.route(text)
        profile = await self._profiles.get_active()
        sensitive_topics = _topics_from_profile(profile)
        sensitive_evaluation = evaluate(text, sensitive_topics)
        positive_topics = (
            tuple(profile.positive_topics or ()) if profile is not None else ()
        )
        conversation = await self._conversations.get_or_create(conversation_id)

        pending = await self._conversations.last_assistant_candidates(
            conversation.id
        )
        if pending:
            person = await self._disambiguation.resolve(text, pending)
            if person is not None:
                return await self._disambiguation_turn(
                    conversation.id,
                    text,
                    person,
                    sensitive_evaluation,
                    positive_topics,
                )

        user_message = await self._conversations.add_message(
            conversation.id, speaker="user", text=text, intent=route.intent.value
        )
        if route.intent in _INGEST_INTENTS:
            await self._ingest_claims(text, conversation.id, user_message.id)

        resolution: Resolution | None = None
        context: AnswerContext
        if route.intent == Intent.memory_query:
            semantic = await self._semantic_retriever.search(text)
            context = self._context_builder.build_from_semantic_memories(
                semantic, term=route.term
            )
        elif route.intent in (Intent.person_query, Intent.event_query) and route.term:
            resolution = await self._resolver.resolve(route.term)
            if resolution.kind == ResolutionKind.resolved:
                data = await self._structured_retriever.retrieve(resolution.person.id)
                context = self._context_builder.build_from_person_data(
                    data, resolution, intent=route.intent
                )
            else:
                context = self._context_builder.build_from_resolution(
                    resolution, route.intent
                )
        else:
            context = AnswerContext(intent=route.intent, term=route.term)

        context = apply_to_context(
            context, sensitive_evaluation, positive_topics=positive_topics
        )
        answer = await self._respond(context, sensitive_evaluation, conversation.id)
        assistant_message = await self._conversations.add_message(
            conversation.id,
            speaker="assistant",
            text=answer.text,
            intent=route.intent.value,
        )
        if answer.sources:
            await self._conversations.add_response_sources(
                assistant_message.id, answer.sources
            )

        gap: KnowledgeGap | None = None
        gap_created = False
        if context.resolution == ResolutionKind.not_found and route.term:
            gap, gap_created = await self._gaps.upsert(
                gap_type="person_unknown",
                canonical_reference=route.term.lower(),
                conversation_id=conversation.id,
                description=f"User asked about unknown person '{route.term}'.",
            )

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

    async def _disambiguation_turn(
        self,
        conversation_id: uuid.UUID,
        text: str,
        person: Person,
        sensitive_evaluation: SensitiveTopicEvaluation,
        positive_topics: tuple[str, ...],
    ) -> ChatResult:
        """Answer a follow-up that selected one pending ambiguity candidate.

        The user is picking a person already listed by the previous assistant
        turn, not making a new query or a declarative statement, so neither
        claim ingestion nor a knowledge-gap upsert runs for this turn.
        """
        await self._conversations.add_message(
            conversation_id,
            speaker="user",
            text=text,
            intent=Intent.person_query.value,
        )
        resolution = Resolution.from_people(person.display_name, [person])
        data = await self._structured_retriever.retrieve(person.id)
        context = self._context_builder.build_from_person_data(
            data, resolution, intent=Intent.person_query
        )
        context = apply_to_context(
            context, sensitive_evaluation, positive_topics=positive_topics
        )
        answer = await self._respond(context, sensitive_evaluation, conversation_id)
        assistant_message = await self._conversations.add_message(
            conversation_id,
            speaker="assistant",
            text=answer.text,
            intent=Intent.person_query.value,
        )
        if answer.sources:
            await self._conversations.add_response_sources(
                assistant_message.id, answer.sources
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
    ) -> Response:
        """Run the responder and enforce non-initiation on its output.

        Context filtering already withholds blocked-topic items from a
        spontaneous turn. This final guard handles a responder that ignores the
        context or the prompt: if its text mentions a blocked topic anyway, the
        answer is replaced with a safe generic reply and its sources are
        dropped, so no trace of the topic reaches the person.
        """
        answer = await self._responder.respond(context)
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
        except Exception:
            logger.warning(
                "claim ingestion (extraction/persistence/grouping) failed for "
                "message %s; skipping ingestion",
                message_id,
                exc_info=True,
            )
