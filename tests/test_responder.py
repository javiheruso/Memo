import uuid
from datetime import UTC, date, datetime, timedelta

import pytest

from app.domain.enums import (
    EpistemicKind,
    EventStatus,
    Intent,
    MemoryStatus,
    RelationshipStatus,
    RelationshipType,
)
from app.llm.deterministic import DeterministicResponder
from app.retrieval.context_builder import (
    AnswerContext,
    EventItem,
    FactItem,
    MediaItem,
    MemoryItem,
    RelationshipItem,
)
from app.retrieval.entity_resolver import PersonCandidate, ResolutionKind

responder = DeterministicResponder()


def _person(name: str = "Francisco Navarro") -> PersonCandidate:
    return PersonCandidate(id=uuid.uuid4(), display_name=name)


def _event(
    title: str = "Visita de María",
    status: EventStatus = EventStatus.confirmed,
    start_at: datetime | None = None,
) -> EventItem:
    return EventItem(
        entity_id=uuid.uuid4(),
        title=title,
        start_at=start_at or datetime.now(UTC) + timedelta(days=2),
        status=status,
    )


def _verified_relation(
    other: str = "Antonio",
    relationship_type: RelationshipType = RelationshipType.friend_of,
) -> RelationshipItem:
    return RelationshipItem(
        entity_id=uuid.uuid4(),
        other_display_name=other,
        relationship_type=relationship_type,
        status=RelationshipStatus.verified,
        epistemic=EpistemicKind.verified_fact,
    )


def _probable_relation(
    other: str = "Carmen", relationship_type: RelationshipType = RelationshipType.other
) -> RelationshipItem:
    return RelationshipItem(
        entity_id=uuid.uuid4(),
        other_display_name=other,
        relationship_type=relationship_type,
        status=RelationshipStatus.probable,
        epistemic=EpistemicKind.probable_fact,
    )


def _memory(narrative: str = "Carmen remembers going fishing.") -> MemoryItem:
    return MemoryItem(
        entity_id=uuid.uuid4(),
        title="Fishing trips",
        narrative=narrative,
        memory_status=MemoryStatus.consolidated,
        epistemic=EpistemicKind.memory,
    )


def _fact(text: str = "Vivió en Guardamar") -> FactItem:
    return FactItem(
        entity_id=uuid.uuid4(),
        text=text,
        status="verified",
        epistemic=EpistemicKind.verified_fact,
    )


def _probable_fact(text: str = "Fue panadero") -> FactItem:
    return FactItem(
        entity_id=uuid.uuid4(),
        text=text,
        status="probable",
        epistemic=EpistemicKind.probable_fact,
    )


def _photo(
    media_type: str = "photo",
    approx_date: date | None = date(1965, 7, 1),
    place_text: str | None = "Santa Pola",
) -> MediaItem:
    return MediaItem(
        entity_id=uuid.uuid4(),
        media_type=media_type,
        title="Foto antigua",
        description=None,
        approx_date=approx_date,
        place_text=place_text,
    )


@pytest.mark.asyncio
async def test_known_person_uses_verified_relation_only():
    context = AnswerContext(
        intent=Intent.person_query,
        term="Francisco",
        resolution=ResolutionKind.resolved,
        person=_person(),
        verified_relations=[_verified_relation()],
        memories=[_memory()],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "amigo de Antonio" in answer
    # Progressive hints stop at the first working one: the memory is NOT surfaced.
    assert "Fishing trips" not in answer
    assert "recuerda" not in answer
    # Only the entities actually used are reported as sources.
    types = {source.entity_type for source in response.sources}
    assert types == {"person", "relationship"}


@pytest.mark.asyncio
async def test_ambiguous_person_is_not_collapsed():
    context = AnswerContext(
        intent=Intent.person_query,
        term="Paco",
        resolution=ResolutionKind.ambiguous,
        candidates=[_person("Francisco Navarro"), _person("Paco García")],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "Francisco Navarro" in answer
    assert "Paco García" in answer
    assert "¿A cuál te refieres?" in answer
    assert {source.entity_type for source in response.sources} == {"person"}


@pytest.mark.asyncio
async def test_unknown_person_is_admitted_without_invention():
    context = AnswerContext(
        intent=Intent.person_query,
        term="Eustaquio",
        resolution=ResolutionKind.not_found,
    )
    response = await responder.respond(context)
    assert response.text == "No tengo guardado quién era Eustaquio."
    assert response.sources == []


@pytest.mark.asyncio
async def test_partial_person_expresses_uncertainty():
    context = AnswerContext(
        intent=Intent.person_query,
        term="Paco García",
        resolution=ResolutionKind.resolved,
        person=_person("Paco García"),
        probable_relations=[_probable_relation()],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "No tengo claro" in answer
    assert "no está confirmado" in answer
    # A probable relation must not be stated as a verified fact.
    assert "era conocido de Carmen." not in answer


@pytest.mark.asyncio
async def test_memory_is_never_presented_as_verified_fact():
    context = AnswerContext(
        intent=Intent.person_query,
        term="Paco",
        resolution=ResolutionKind.resolved,
        person=_person("Paco Navarro"),
        memories=[_memory("Carmen remembers that Paco and Antonio fished together.")],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "Recuerdo guardado" in answer
    assert "Carmen remembers that Paco and Antonio fished together." in answer
    assert "amigo de" not in answer
    assert "era amigo" not in answer


@pytest.mark.asyncio
async def test_resolved_person_without_any_data_is_honest():
    context = AnswerContext(
        intent=Intent.person_query,
        term="Alguien",
        resolution=ResolutionKind.resolved,
        person=_person("Alguien"),
    )
    response = await responder.respond(context)
    assert "todavía no tengo detalles" in response.text


@pytest.mark.asyncio
async def test_facts_tier_surfaces_stored_facts_before_memories():
    # The dossier hint ladder: name + relation -> facts/context -> photo ->
    # memory. With no relation but a stored fact, the facts tier fires.
    context = AnswerContext(
        intent=Intent.person_query,
        term="Luisa",
        resolution=ResolutionKind.resolved,
        person=_person("Luisa"),
        facts=[_fact("Vivió en Guardamar")],
        memories=[_memory()],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "Dato: 'Vivió en Guardamar'" in answer
    assert "Recuerdo guardado" not in answer
    assert "amigo de" not in answer
    types = {source.entity_type for source in response.sources}
    assert types == {"person", "fact"}
    assert all(
        source.entity_type != "fact"
        or source.usage_type == EpistemicKind.verified_fact.value
        for source in response.sources
    )


@pytest.mark.asyncio
async def test_facts_tier_never_presents_probable_fact_as_verified():
    context = AnswerContext(
        intent=Intent.person_query,
        term="Luisa",
        resolution=ResolutionKind.resolved,
        person=_person("Luisa"),
        facts=[_probable_fact("Fue panadero")],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "Dato probable: 'Fue panadero'" in answer
    assert "Dato: 'Fue panadero'" not in answer
    sources = response.sources
    fact_sources = [s for s in sources if s.entity_type == "fact"]
    assert all(
        source.usage_type == EpistemicKind.probable_fact.value
        for source in fact_sources
    )


@pytest.mark.asyncio
async def test_photo_tier_fires_before_memories_with_metadata_only():
    context = AnswerContext(
        intent=Intent.person_query,
        term="Luisa",
        resolution=ResolutionKind.resolved,
        person=_person("Luisa"),
        media=[_photo()],
        memories=[_memory("A beach afternoon with Luisa.")],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "pero tengo guardada una fotografía de 1965, en Santa Pola." in answer
    # Progressive hints stop at the first working one: the memory is not used.
    assert "A beach afternoon with Luisa." not in answer
    # The text only renders stored metadata (year/place), never a caption or
    # any visual claim.
    assert "Foto antigua" not in answer
    types = {source.entity_type for source in response.sources}
    assert types == {"person", "media"}
    assert all(
        source.entity_type != "media" or source.usage_type == "media"
        for source in response.sources
    )


@pytest.mark.asyncio
async def test_photo_tier_without_date_or_place_renders_generic_phrase():
    context = AnswerContext(
        intent=Intent.person_query,
        term="Luisa",
        resolution=ResolutionKind.resolved,
        person=_person("Luisa"),
        media=[_photo(approx_date=None, place_text=None)],
    )
    response = await responder.respond(context)
    assert response.text == (
        "No tengo claro exactamente quién era Luisa, "
        "pero tengo guardada una fotografía."
    )


@pytest.mark.asyncio
async def test_non_photo_media_is_not_surfaced_as_a_photo_hint():
    context = AnswerContext(
        intent=Intent.person_query,
        term="Luisa",
        resolution=ResolutionKind.resolved,
        person=_person("Luisa"),
        media=[_photo(media_type="video")],
        memories=[_memory("Un baile con Luisa.")],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "fotografía" not in answer
    assert "Recuerdo guardado: 'Un baile con Luisa.'" in answer
    types = {source.entity_type for source in response.sources}
    assert types == {"person", "memory"}


@pytest.mark.asyncio
async def test_hint_ladder_stops_at_first_working_tier():
    # Full ladder context: relations outrank facts, which outrank photo, which
    # outranks memory. Only the highest working tier is rendered.
    context = AnswerContext(
        intent=Intent.person_query,
        term="Paco",
        resolution=ResolutionKind.resolved,
        person=_person("Paco García"),
        verified_relations=[_verified_relation(other="Antonio")],
        facts=[_fact()],
        media=[_photo()],
        memories=[_memory()],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "amigo de Antonio" in answer
    assert "Dato:" not in answer
    assert "fotografía" not in answer
    assert "Recuerdo guardado" not in answer

    without_relation = AnswerContext(
        intent=Intent.person_query,
        term="Luisa",
        resolution=ResolutionKind.resolved,
        person=_person("Luisa"),
        facts=[_fact("Vivió en Guardamar")],
        media=[_photo()],
        memories=[_memory()],
    )
    response = await responder.respond(without_relation)
    answer = response.text
    assert "Dato: 'Vivió en Guardamar'" in answer
    assert "fotografía" not in answer
    assert "Recuerdo guardado" not in answer

    with_only_photo_and_memory = AnswerContext(
        intent=Intent.person_query,
        term="Luisa",
        resolution=ResolutionKind.resolved,
        person=_person("Luisa"),
        media=[_photo()],
        memories=[_memory()],
    )
    response = await responder.respond(with_only_photo_and_memory)
    answer = response.text
    assert "fotografía" in answer
    assert "Recuerdo guardado" not in answer


@pytest.mark.asyncio
async def test_general_chat_does_not_touch_personal_context():
    context = AnswerContext(intent=Intent.general_chat, term=None)
    response = await responder.respond(context)
    assert (
        response.text == "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."
    )
    assert response.sources == []


@pytest.mark.asyncio
async def test_confirmed_event_is_phrased_as_confirmed_not_completed():
    context = AnswerContext(
        intent=Intent.event_query,
        term="María",
        resolution=ResolutionKind.resolved,
        person=_person("María"),
        events=[_event(title="Visita de María el domingo", status=EventStatus.confirmed)],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "está confirmado 'Visita de María el domingo'" in answer
    assert "se celebró" not in answer
    assert "planificado" not in answer
    assert {source.entity_type for source in response.sources} == {"person", "event"}
    assert all(source.entity_type != "event" or source.usage_type == "confirmed" for source in response.sources)


@pytest.mark.asyncio
async def test_planned_event_is_phrased_as_planned_not_completed():
    context = AnswerContext(
        intent=Intent.event_query,
        term="Antonio",
        resolution=ResolutionKind.resolved,
        person=_person("Antonio"),
        events=[_event(title="Cumpleaños de Antonio", status=EventStatus.planned)],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "está planificado 'Cumpleaños de Antonio'" in answer
    assert "se celebró" not in answer
    assert "confirmado" not in answer


@pytest.mark.asyncio
async def test_completed_event_uses_past_phrasing():
    context = AnswerContext(
        intent=Intent.event_query,
        term="Carmen",
        resolution=ResolutionKind.resolved,
        person=_person("Carmen"),
        events=[
            _event(
                title="Baile en la sala",
                status=EventStatus.completed,
                start_at=datetime.now(UTC) - timedelta(days=5),
            )
        ],
    )
    response = await responder.respond(context)
    assert "No tengo eventos próximos" in response.text
    assert "se celebró" not in response.text


@pytest.mark.asyncio
async def test_cancelled_event_is_not_presented_as_upcoming():
    context = AnswerContext(
        intent=Intent.event_query,
        term="Carmen",
        resolution=ResolutionKind.resolved,
        person=_person("Carmen"),
        events=[
            _event(title="Cita con el médico", status=EventStatus.cancelled),
            _event(title="Visita de María", status=EventStatus.confirmed),
        ],
    )
    response = await responder.respond(context)
    answer = response.text
    assert "Visita de María" in answer
    assert "Cita con el médico" not in answer


@pytest.mark.asyncio
async def test_event_query_without_person_asks_for_clarification():
    context = AnswerContext(intent=Intent.event_query, term=None)
    response = await responder.respond(context)
    assert "¿De quién quieres saber sus eventos?" in response.text
    assert response.sources == []


@pytest.mark.asyncio
async def test_event_query_unknown_person_is_admitted():
    context = AnswerContext(
        intent=Intent.event_query,
        term="Eustaquio",
        resolution=ResolutionKind.not_found,
    )
    response = await responder.respond(context)
    assert response.text == "No tengo guardado quién es Eustaquio."
    assert response.sources == []


@pytest.mark.asyncio
async def test_event_query_ambiguous_person_is_not_collapsed():
    context = AnswerContext(
        intent=Intent.event_query,
        term="Paco",
        resolution=ResolutionKind.ambiguous,
        candidates=[_person("Francisco Navarro"), _person("Paco García")],
    )
    response = await responder.respond(context)
    assert "Francisco Navarro" in response.text
    assert "Paco García" in response.text
    assert "¿A cuál te refieres?" in response.text


@pytest.mark.asyncio
async def test_unknown_intent_asks_for_repeat():
    context = AnswerContext(intent=Intent.unknown, term=None)
    response = await responder.respond(context)
    assert "No he entendido" in response.text


@pytest.mark.asyncio
async def test_orientation_reports_day_date_and_time_from_clock():
    context = AnswerContext(
        intent=Intent.orientation_query,
        term=None,
        now=datetime(2026, 8, 31, 14, 5, tzinfo=UTC),
    )
    response = await responder.respond(context)
    assert "lunes" in response.text
    assert "31 de agosto de 2026" in response.text
    assert "14:05" in response.text
    assert response.sources == []


@pytest.mark.asyncio
async def test_orientation_naive_clock_is_treated_as_utc():
    context = AnswerContext(
        intent=Intent.orientation_query,
        term=None,
        now=datetime(2026, 1, 1, 9, 0),  # noqa: DTZ001 - naive input is the case under test
    )
    response = await responder.respond(context)
    assert "jueves" in response.text
    assert "1 de enero de 2026" in response.text
    assert "09:00" in response.text
