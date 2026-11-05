"""Regression coverage for AUD-005: rejected and pending data visibility.

Rejected facts/relationships must never reach retrieval or answers. Pending
rows may still be fetched for context building, but they are withheld and
reported explicitly instead of being softened into ``probable``.
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import (
    EpistemicKind,
    FactStatus,
    Intent,
    RelationshipStatus,
    RelationshipType,
    SourceType,
)
from app.llm.deterministic import DeterministicResponder
from app.policies.trust import (
    epistemic_kind_for_fact,
    epistemic_kind_for_relationship,
    is_answerable_fact_status,
    is_answerable_relationship_status,
)
from app.repositories.fact import FactRepository
from app.repositories.person import PersonRepository
from app.repositories.relationship import RelationshipRepository
from app.repositories.source import SourceRepository
from app.retrieval.context_builder import (
    AnswerContext,
    ContextBuilder,
    FactItem,
    RelationshipItem,
)
from app.retrieval.entity_resolver import PersonCandidate, Resolution, ResolutionKind
from app.retrieval.structured_retriever import PersonData
from app.schemas.fact import FactCreate
from app.schemas.person import PersonCreate
from app.schemas.relationship import RelationshipCreate
from app.schemas.source import SourceCreate


async def _subject_other_source(
    db_session: AsyncSession,
) -> tuple[object, object, object]:
    person_repo = PersonRepository(db_session)
    subject = await person_repo.create(PersonCreate(first_name="Subject"))
    other = await person_repo.create(PersonCreate(first_name="Other"))
    source = await SourceRepository(db_session).create(
        SourceCreate(source_type=SourceType.system, description="test source")
    )
    return subject, other, source


async def _status_facts(db_session: AsyncSession, subject, source) -> dict:
    repo = FactRepository(db_session)
    return {
        status: await repo.create(
            FactCreate(
                subject_person_id=subject.id,
                fact_type="hobby",
                value_text=f"{status.value} fact",
                status=status,
                source_id=source.id,
            )
        )
        for status in FactStatus
    }


async def _status_relationships(db_session: AsyncSession, subject, other, source) -> dict:
    repo = RelationshipRepository(db_session)
    return {
        status: await repo.create(
            RelationshipCreate(
                person_a_id=subject.id,
                person_b_id=other.id,
                relationship_type=RelationshipType.friend_of,
                status=status,
                source_id=source.id,
            )
        )
        for status in RelationshipStatus
    }


async def test_fact_retrieval_excludes_rejected_rows(
    db_session: AsyncSession,
) -> None:
    subject, _, source = await _subject_other_source(db_session)
    rows = await _status_facts(db_session, subject, source)

    retrieved = await FactRepository(db_session).for_subject_person(subject.id)

    assert {fact.status for fact in retrieved} == {
        FactStatus.verified,
        FactStatus.probable,
        FactStatus.pending,
    }
    assert rows[FactStatus.rejected].id not in {fact.id for fact in retrieved}


async def test_relationship_retrieval_excludes_rejected_rows(
    db_session: AsyncSession,
) -> None:
    subject, other, source = await _subject_other_source(db_session)
    rows = await _status_relationships(db_session, subject, other, source)

    retrieved = await RelationshipRepository(db_session).for_person(subject.id)

    assert {relationship.status for relationship in retrieved} == {
        RelationshipStatus.verified,
        RelationshipStatus.probable,
        RelationshipStatus.pending,
    }
    assert rows[RelationshipStatus.rejected].id not in {
        relationship.id for relationship in retrieved
    }


async def test_context_withholds_pending_and_rejected_and_reports_them(
    db_session: AsyncSession,
) -> None:
    subject, other, source = await _subject_other_source(db_session)
    facts = await _status_facts(db_session, subject, source)
    relationships = await _status_relationships(db_session, subject, other, source)
    resolution = Resolution.from_people(subject.display_name, [subject])

    context = ContextBuilder().build_from_person_data(
        PersonData(
            person=subject,
            relationships=list(relationships.values()),
            facts=list(facts.values()),
            memories=[],
            events=[],
            media=[],
        ),
        resolution,
    )

    assert {fact.status for fact in context.facts} == {"verified", "probable"}
    assert {
        relationship.status
        for relationship in [*context.verified_relations, *context.probable_relations]
    } == {RelationshipStatus.verified, RelationshipStatus.probable}

    policy_text = " ".join(context.policies)
    assert "pending_facts_withheld: 1" in policy_text
    assert "rejected_facts_excluded: 1" in policy_text
    assert "pending_relationships_withheld: 1" in policy_text
    assert "rejected_relationships_excluded: 1" in policy_text


@pytest.mark.parametrize("status", [FactStatus.pending, FactStatus.rejected])
def test_fact_epistemic_kind_fails_closed_for_unanswerable_states(
    status: FactStatus,
) -> None:
    with pytest.raises(ValueError, match="not answerable"):
        epistemic_kind_for_fact(status)
    assert not is_answerable_fact_status(status)


@pytest.mark.parametrize(
    "status", [RelationshipStatus.pending, RelationshipStatus.rejected]
)
def test_relationship_epistemic_kind_fails_closed_for_unanswerable_states(
    status: RelationshipStatus,
) -> None:
    with pytest.raises(ValueError, match="not answerable"):
        epistemic_kind_for_relationship(status)
    assert not is_answerable_relationship_status(status)


def test_answerable_states_still_include_verified_and_probable() -> None:
    assert is_answerable_fact_status(FactStatus.verified)
    assert is_answerable_fact_status(FactStatus.probable)
    assert is_answerable_relationship_status(RelationshipStatus.verified)
    assert is_answerable_relationship_status(RelationshipStatus.probable)
    assert epistemic_kind_for_fact(FactStatus.verified) == EpistemicKind.verified_fact
    assert epistemic_kind_for_fact(FactStatus.probable) == EpistemicKind.probable_fact


def _person(name: str = "Subject") -> PersonCandidate:
    return PersonCandidate(id=uuid.uuid4(), display_name=name)


def _fact(text: str, status: str) -> FactItem:
    return FactItem(
        entity_id=uuid.uuid4(),
        text=text,
        status=status,
        epistemic=EpistemicKind.probable_fact,
    )


def _relation(status: RelationshipStatus) -> RelationshipItem:
    return RelationshipItem(
        entity_id=uuid.uuid4(),
        other_display_name="Other",
        relationship_type=RelationshipType.friend_of,
        status=status,
        epistemic=EpistemicKind.probable_fact,
    )


@pytest.mark.asyncio
async def test_responder_never_uses_pending_or_rejected_person_data() -> None:
    context = AnswerContext(
        intent=Intent.person_query,
        term="Subject",
        resolution=ResolutionKind.resolved,
        person=_person(),
        verified_relations=[
            _relation(RelationshipStatus.pending),
            _relation(RelationshipStatus.rejected),
        ],
        probable_relations=[
            _relation(RelationshipStatus.pending),
            _relation(RelationshipStatus.rejected),
        ],
        facts=[
            _fact("pending fact", "pending"),
            _fact("rejected fact", "rejected"),
        ],
    )

    response = await DeterministicResponder().respond(context)

    assert "pending fact" not in response.text
    assert "rejected fact" not in response.text
    assert "Posiblemente" not in response.text
    assert all(
        source.entity_type not in {"fact", "relationship"}
        for source in response.sources
    )
    assert response.text == "Tengo guardado a Subject, pero todavía no tengo detalles."


@pytest.mark.asyncio
async def test_responder_does_not_answer_a_fact_query_from_withheld_rows() -> None:
    context = AnswerContext(
        intent=Intent.fact_query,
        term="hobby",
        resolution=ResolutionKind.resolved,
        person=_person(),
        facts=[
            _fact("pending fact", "pending"),
            _fact("rejected fact", "rejected"),
        ],
    )

    response = await DeterministicResponder().respond(context)

    assert response.text == "No tengo guardado ningún dato sobre eso de Subject."
    assert "pending fact" not in response.text
    assert "rejected fact" not in response.text
    assert all(source.entity_type != "fact" for source in response.sources)
