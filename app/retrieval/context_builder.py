import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from app.domain.enums import (
    EpistemicKind,
    EventStatus,
    FactStatus,
    Intent,
    MemoryStatus,
    RelationshipStatus,
    RelationshipType,
)
from app.models import Event, Fact, Person, Relationship
from app.policies.trust import (
    epistemic_kind_for_fact,
    epistemic_kind_for_relationship,
    filter_sensitive_memories,
    is_answerable_fact_status,
    is_answerable_relationship_status,
)
from app.retrieval.entity_resolver import PersonCandidate, Resolution, ResolutionKind
from app.retrieval.semantic_retriever import SemanticSearchResult
from app.retrieval.structured_retriever import PersonData


@dataclass(frozen=True)
class RelationshipItem:
    entity_id: uuid.UUID
    other_display_name: str
    relationship_type: RelationshipType
    status: RelationshipStatus
    epistemic: EpistemicKind
    related_person_id: uuid.UUID | None = None
    source_id: uuid.UUID | None = None
    source_type: str | None = None


@dataclass(frozen=True)
class FactItem:
    entity_id: uuid.UUID
    text: str
    status: str
    epistemic: EpistemicKind
    fact_type: str | None = None
    source_id: uuid.UUID | None = None
    source_type: str | None = None


@dataclass(frozen=True)
class MemoryItem:
    entity_id: uuid.UUID
    title: str | None
    narrative: str
    memory_status: MemoryStatus
    epistemic: EpistemicKind


@dataclass(frozen=True)
class EventItem:
    entity_id: uuid.UUID
    title: str
    start_at: datetime
    status: EventStatus


@dataclass(frozen=True)
class MediaItem:
    """Metadata-only rendering of a stored media row.

    Carries exactly the columns a caregiver may have filled in (type, caption,
    approximate date, place) and nothing else. There is never an interpretation
    of the image content here, so no responder can describe what a photo
    "shows" from this item.
    """

    entity_id: uuid.UUID
    media_type: str
    title: str | None
    description: str | None
    approx_date: date | None
    place_text: str | None


@dataclass(frozen=True)
class AnswerContext:
    """The minimal, epistemically-tagged context handed to a responder.

    The responder only ever sees this context. Verified relations and facts
    live in one list, memories in another, and each item carries an explicit
    epistemic kind; nothing is merged across categories, so a memory can never
    be presented as a verified fact and the responder cannot invent data.
    Events keep their own status so a planned or confirmed event is never
    phrased as a completed one. ``now`` is a code-provided clock (UTC) so any
    day/date/time statement is grounded in the application's time, never the
    model's or the responder's own knowledge. ``follow_up_topic`` is set only
    when a possessive follow-up carried the previous person forward; it holds
    a natural Spanish noun phrase for an honest "no data about that yet" reply.
    """

    intent: Intent
    term: str | None = None
    resolution: ResolutionKind | None = None
    follow_up_topic: str | None = None
    relationship_role: str | None = None
    candidates: list[PersonCandidate] = field(default_factory=list)
    person: PersonCandidate | None = None
    verified_relations: list[RelationshipItem] = field(default_factory=list)
    probable_relations: list[RelationshipItem] = field(default_factory=list)
    facts: list[FactItem] = field(default_factory=list)
    memories: list[MemoryItem] = field(default_factory=list)
    events: list[EventItem] = field(default_factory=list)
    media: list[MediaItem] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)
    sensitive_topic_requested: str | None = None
    sensitive_redirect_topic: str | None = None
    blocked_sensitive_topics: tuple[str, ...] = ()
    now: datetime = field(default_factory=lambda: datetime.now(UTC))


class ContextBuilder:
    """Build an ``AnswerContext`` applying the trust-model policies in code."""

    def build_from_resolution(
        self, resolution: Resolution, intent: Intent
    ) -> AnswerContext:
        if resolution.kind == ResolutionKind.ambiguous:
            return AnswerContext(
                intent=intent,
                term=resolution.term,
                resolution=resolution.kind,
                candidates=resolution.candidates,
                policies=[
                    (
                        "do_not_collapse_ambiguity: multiple candidates surfaced, "
                        "no arbitrary pick"
                    )
                ],
            )
        return AnswerContext(
            intent=intent,
            term=resolution.term,
            resolution=resolution.kind,
            policies=["do_not_invent: no stored data, acknowledge unknown"],
        )

    def build_from_person_data(
        self,
        data: PersonData,
        resolution: Resolution,
        intent: Intent = Intent.person_query,
    ) -> AnswerContext:
        answerable_relationships = [
            relationship
            for relationship in data.relationships
            if is_answerable_relationship_status(relationship.status)
        ]
        withheld_relationships = [
            relationship
            for relationship in data.relationships
            if not is_answerable_relationship_status(relationship.status)
        ]
        verified_relations: list[RelationshipItem] = []
        probable_relations: list[RelationshipItem] = []
        for relationship in answerable_relationships:
            item = self._relationship_item(relationship, data.person.id)
            if item.epistemic == EpistemicKind.verified_fact:
                verified_relations.append(item)
            else:
                probable_relations.append(item)

        answerable_facts = [
            fact for fact in data.facts if is_answerable_fact_status(fact.status)
        ]
        withheld_facts = [
            fact for fact in data.facts if not is_answerable_fact_status(fact.status)
        ]
        facts = [self._fact_item(fact) for fact in answerable_facts]

        safe_memories, filtered_memories = filter_sensitive_memories(data.memories)
        policies: list[str] = []
        policies.extend(
            self._withheld_status_policies(
                facts=withheld_facts,
                relationships=withheld_relationships,
            )
        )
        if filtered_memories:
            policies.append(
                "sensitive_content_filtered: "
                f"{len(filtered_memories)} high-sensitivity memories excluded"
            )
        memories = [
            MemoryItem(
                entity_id=memory.id,
                title=memory.title,
                narrative=memory.narrative,
                memory_status=memory.memory_status,
                epistemic=EpistemicKind.memory,
            )
            for memory in safe_memories
        ]

        events = [self._event_item(event) for event in data.events]

        media = [
            MediaItem(
                entity_id=item.id,
                media_type=item.media_type,
                title=item.title,
                description=item.description,
                approx_date=item.approx_date,
                place_text=item.place_text,
            )
            for item in data.media
        ]

        return AnswerContext(
            intent=intent,
            term=resolution.term,
            resolution=ResolutionKind.resolved,
            person=resolution.person,
            verified_relations=verified_relations,
            probable_relations=probable_relations,
            facts=facts,
            memories=memories,
            events=events,
            media=media,
            policies=policies,
        )

    def build_from_relationship_query(
        self,
        person: Person | None,
        role: str,
        relationships: list[Relationship],
    ) -> AnswerContext:
        """Build a context for an explicit role question about an active person.

        Only verified and probable relationships are answerable; pending or
        rejected rows are withheld exactly like any other relationship answer.
        ``person`` is the resolved subject from the current conversation, and
        ``None`` means no safe subject exists, so the responder must not guess.
        """
        verified_relations: list[RelationshipItem] = []
        probable_relations: list[RelationshipItem] = []
        withheld: list[Relationship] = []
        if person is not None:
            for relationship in relationships:
                if not is_answerable_relationship_status(relationship.status):
                    withheld.append(relationship)
                    continue
                item = self._relationship_item(relationship, person.id)
                if item.epistemic == EpistemicKind.verified_fact:
                    verified_relations.append(item)
                else:
                    probable_relations.append(item)

        policies: list[str] = []
        if withheld:
            policies.extend(
                self._withheld_status_policies(
                    facts=[],
                    relationships=withheld,
                )
            )
        return AnswerContext(
            intent=Intent.person_query,
            term=role,
            resolution=(ResolutionKind.resolved if person is not None else None),
            person=(
                PersonCandidate(id=person.id, display_name=person.display_name)
                if person is not None
                else None
            ),
            verified_relations=verified_relations,
            probable_relations=probable_relations,
            policies=policies,
            relationship_role=role,
        )

    def build_from_semantic_memories(
        self,
        result: SemanticSearchResult,
        *,
        term: str | None = None,
        intent: Intent = Intent.memory_query,
    ) -> AnswerContext:
        """Build a context from semantic memory hits for a ``memory_query``.

        Sensitivity filtering already happened inside the retriever; this only
        wraps the safe hits as ``memory`` epistemic items so no semantic hit is
        ever presented as a verified fact. A note is added to ``policies`` when
        hits were excluded.
        """
        memories = [
            MemoryItem(
                entity_id=hit.memory.id,
                title=hit.memory.title,
                narrative=hit.memory.narrative,
                memory_status=hit.memory.memory_status,
                epistemic=EpistemicKind.memory,
            )
            for hit in result.hits
        ]
        policies: list[str] = []
        if result.filtered_count:
            policies.append(
                "sensitive_content_filtered: "
                f"{result.filtered_count} semantic memories excluded"
            )
        return AnswerContext(
            intent=intent,
            term=term,
            memories=memories,
            policies=policies,
        )

    @staticmethod
    def _withheld_status_policies(
        *,
        facts: list[Fact],
        relationships: list[Relationship],
    ) -> list[str]:
        """Name every withheld row so pending is never silently re-labelled."""
        policies: list[str] = []
        pending_facts = sum(fact.status == FactStatus.pending for fact in facts)
        rejected_facts = sum(fact.status == FactStatus.rejected for fact in facts)
        pending_relationships = sum(
            relationship.status == RelationshipStatus.pending
            for relationship in relationships
        )
        rejected_relationships = sum(
            relationship.status == RelationshipStatus.rejected
            for relationship in relationships
        )
        if pending_facts:
            policies.append(
                "pending_facts_withheld: "
                f"{pending_facts} pending facts excluded until verified"
            )
        if rejected_facts:
            policies.append(
                "rejected_facts_excluded: "
                f"{rejected_facts} rejected facts excluded from answers"
            )
        if pending_relationships:
            policies.append(
                "pending_relationships_withheld: "
                f"{pending_relationships} pending relationships excluded until verified"
            )
        if rejected_relationships:
            policies.append(
                "rejected_relationships_excluded: "
                f"{rejected_relationships} rejected relationships excluded from answers"
            )
        return policies

    @staticmethod
    def _relationship_item(
        relationship: Relationship, subject_id: uuid.UUID
    ) -> RelationshipItem:
        from app.domain.enums import invert_relationship_type

        if relationship.person_a_id == subject_id:
            other = relationship.person_b
            relationship_type = relationship.relationship_type
        else:
            other = relationship.person_a
            relationship_type = invert_relationship_type(relationship.relationship_type)
        return RelationshipItem(
            entity_id=relationship.id,
            other_display_name=other.display_name if other else "?",
            relationship_type=relationship_type,
            status=relationship.status,
            epistemic=epistemic_kind_for_relationship(relationship.status),
            related_person_id=other.id if other else None,
            source_id=relationship.source_id,
            source_type=(
                relationship.source.source_type.value
                if relationship.source is not None
                else None
            ),
        )

    @classmethod
    def _fact_item(cls, fact: Fact) -> FactItem:
        source = fact.source
        return FactItem(
            entity_id=fact.id,
            text=cls._fact_text(fact),
            status=fact.status.value,
            epistemic=epistemic_kind_for_fact(fact.status),
            fact_type=fact.fact_type,
            source_id=fact.source_id,
            source_type=source.source_type.value if source is not None else None,
        )

    @staticmethod
    def _fact_text(fact: Fact) -> str:
        parts = [part for part in (fact.value_text, fact.place_text) if part]
        return "; ".join(parts) if parts else f"({fact.fact_type})"

    @staticmethod
    def _event_item(event: Event) -> EventItem:
        return EventItem(
            entity_id=event.id,
            title=event.title,
            start_at=event.start_at,
            status=event.status,
        )
