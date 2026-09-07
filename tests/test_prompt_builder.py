import uuid
from datetime import UTC, date, datetime, timedelta

from app.domain.enums import (
    EpistemicKind,
    EventStatus,
    Intent,
    MemoryStatus,
    RelationshipStatus,
    RelationshipType,
)
from app.llm.prompt_builder import allowed_sources, build_prompt
from app.retrieval.context_builder import (
    AnswerContext,
    EventItem,
    FactItem,
    MediaItem,
    MemoryItem,
    RelationshipItem,
)
from app.retrieval.entity_resolver import PersonCandidate, ResolutionKind


def _person(name: str = "Francisco Navarro") -> PersonCandidate:
    return PersonCandidate(id=uuid.uuid4(), display_name=name)


def _verified_relation(
    other: str = "Antonio", relationship_type: RelationshipType = RelationshipType.friend_of
) -> RelationshipItem:
    return RelationshipItem(
        entity_id=uuid.uuid4(),
        other_display_name=other,
        relationship_type=relationship_type,
        status=RelationshipStatus.verified,
        epistemic=EpistemicKind.verified_fact,
    )


def _probable_relation() -> RelationshipItem:
    return RelationshipItem(
        entity_id=uuid.uuid4(),
        other_display_name="Carmen",
        relationship_type=RelationshipType.other,
        status=RelationshipStatus.probable,
        epistemic=EpistemicKind.probable_fact,
    )


def _fact() -> FactItem:
    return FactItem(
        entity_id=uuid.uuid4(),
        text="Vivió en Guardamar",
        status="verified",
        epistemic=EpistemicKind.verified_fact,
    )


def _memory() -> MemoryItem:
    return MemoryItem(
        entity_id=uuid.uuid4(),
        title="Pesca",
        narrative="Carmen recuerda ir a pescar con Antonio.",
        memory_status=MemoryStatus.consolidated,
        epistemic=EpistemicKind.memory,
    )


def _event() -> EventItem:
    return EventItem(
        entity_id=uuid.uuid4(),
        title="Visita de María",
        start_at=datetime.now(UTC) + timedelta(days=2),
        status=EventStatus.confirmed,
    )


def _photo() -> MediaItem:
    return MediaItem(
        entity_id=uuid.uuid4(),
        media_type="photo",
        title="Foto antigua",
        description=None,
        approx_date=date(1965, 7, 1),
        place_text="Santa Pola",
    )


def test_resolved_person_prompt_includes_data_and_allowed_ids() -> None:
    person = _person()
    relation = _verified_relation()
    fact = _fact()
    memory = _memory()
    photo = _photo()
    context = AnswerContext(
        intent=Intent.person_query,
        term="Francisco",
        resolution=ResolutionKind.resolved,
        person=person,
        verified_relations=[relation],
        facts=[fact],
        memories=[memory],
        media=[photo],
    )
    prompt = build_prompt(context)
    assert "Francisco Navarro" in prompt.user
    assert str(person.id) in prompt.user
    assert "amigo" in prompt.user or "friend_of" in prompt.user
    assert "Vivió en Guardamar" in prompt.user
    assert "Carmen recuerda ir a pescar" in prompt.user
    assert "Never describe what the photograph shows" in prompt.system
    assert "type photo" in prompt.user
    assert "title 'Foto antigua'" in prompt.user
    assert "Allowed source ids" in prompt.user
    for (entity_type, entity_id), usage in allowed_sources(context).items():
        assert f"{entity_type} {entity_id} ({usage})" in prompt.user
    assert allowed_sources(context)[("media", str(photo.entity_id))] == "media"


def test_allowed_sources_derive_usage_from_context() -> None:
    person = _person()
    context = AnswerContext(
        intent=Intent.person_query,
        term="Francisco",
        resolution=ResolutionKind.resolved,
        person=person,
        verified_relations=[_verified_relation()],
        probable_relations=[_probable_relation()],
        memories=[_memory()],
        events=[_event()],
    )
    allowed = allowed_sources(context)
    assert allowed[("person", str(person.id))] == "subject"
    assert set(allowed.values()) == {
        "subject",
        "verified_fact",
        "probable_fact",
        "memory",
        "confirmed",
    }


def test_ambiguous_person_lists_candidates_and_never_a_subject() -> None:
    a, b = _person("Francisco Navarro"), _person("Paco García")
    context = AnswerContext(
        intent=Intent.person_query,
        term="Paco",
        resolution=ResolutionKind.ambiguous,
        candidates=[a, b],
    )
    prompt = build_prompt(context)
    assert "Francisco Navarro" in prompt.user
    assert "Paco García" in prompt.user
    assert "Candidate people" in prompt.user
    assert "Resolution: ambiguous" in prompt.user
    allowed = allowed_sources(context)
    assert allowed[("person", str(a.id))] == "candidate"
    assert allowed[("person", str(b.id))] == "candidate"
    assert "subject" not in allowed.values()


def test_unknown_person_allows_no_sources() -> None:
    context = AnswerContext(
        intent=Intent.person_query,
        term="Eustaquio",
        resolution=ResolutionKind.not_found,
    )
    prompt = build_prompt(context)
    assert "Eustaquio" in prompt.user
    assert allowed_sources(context) == {}
    assert "never invent" in prompt.system


def test_system_instructions_enforce_trust_model() -> None:
    context = AnswerContext(intent=Intent.general_chat, term=None)
    system = build_prompt(context).system
    assert "Never invent" in system
    assert "Reply in Spanish" in system
    assert "present a planned or confirmed event as already completed" in system
    assert "candidates" in system
    assert "JSON" in system


def test_prompt_exposes_no_db_mechanism() -> None:
    context = AnswerContext(
        intent=Intent.person_query,
        term="Francisco",
        resolution=ResolutionKind.resolved,
        person=_person(),
    )
    prompt = build_prompt(context)
    combined = prompt.system + prompt.user
    for forbidden in ("SELECT", "sqlalchemy", "repository", "tool", "session"):
        assert forbidden.lower() not in combined.lower()


def test_event_query_prompt_keeps_event_status() -> None:
    event = _event()
    context = AnswerContext(
        intent=Intent.event_query,
        term="María",
        resolution=ResolutionKind.resolved,
        person=_person("María"),
        events=[event],
    )
    prompt = build_prompt(context)
    assert event.status.value in prompt.user
    assert str(event.entity_id) in prompt.user


def test_prompt_includes_trusted_clock_context() -> None:
    now = datetime(2026, 8, 31, 14, 5, tzinfo=UTC)
    context = AnswerContext(intent=Intent.orientation_query, now=now)
    prompt = build_prompt(context)
    assert now.isoformat() in prompt.user
    assert "code-provided clock" in prompt.user
    assert "code-provided clock" in prompt.system
    assert "never on your own knowledge" in prompt.system
