from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import EventStatus
from app.repositories.event import EventRepository
from app.repositories.person import PersonRepository
from app.repositories.source import SourceRepository
from app.schemas.event import EventCreate, EventPersonCreate
from app.schemas.person import PersonCreate
from app.schemas.source import SourceCreate
from tests.fixtures.universe import SeedUniverse


async def test_event_create_with_people_roundtrip(db_session: AsyncSession) -> None:
    person = await PersonRepository(db_session).create(PersonCreate(first_name="María"))
    source = await SourceRepository(db_session).create(
        SourceCreate(source_type="system", description="test source")
    )
    start = datetime(2026, 9, 5, 10, 0, tzinfo=UTC)
    event = await EventRepository(db_session).create(
        EventCreate(
            title="Visita",
            description="María viene de visita.",
            start_at=start,
            status=EventStatus.confirmed,
            source_id=source.id,
            people=[EventPersonCreate(person_id=person.id, role="guest")],
        )
    )
    assert event.status == EventStatus.confirmed
    assert [(p.person_id, p.role) for p in event.people] == [(person.id, "guest")]

    fetched = await EventRepository(db_session).get(event.id)
    assert fetched is not None
    assert fetched.people[0].person.display_name == "María"


async def test_event_default_status_is_planned(db_session: AsyncSession) -> None:
    event = await EventRepository(db_session).create(
        EventCreate(
            title="Cita",
            start_at=datetime(2026, 9, 5, 10, 0, tzinfo=UTC),
        )
    )
    assert event.status == EventStatus.planned


async def test_for_person_returns_linked_events_ordered_by_start(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    repo = EventRepository(db_session)
    events = await repo.for_person(seed.person_ids["maria"])
    assert [event.title for event in events] == ["Visita de María el domingo"]
    events = await repo.for_person(seed.person_ids["carmen"])
    titles = [event.title for event in events]
    assert titles == [
        "Baile en la sala",
        "Cita con el médico",
        "Visita de María el domingo",
        "Cumpleaños de Antonio",
    ]


async def test_upcoming_for_person_filters_by_time_and_status(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    repo = EventRepository(db_session)
    now = datetime.now(UTC)
    events = await repo.upcoming_for_person(seed.person_ids["carmen"], now=now)
    titles = [event.title for event in events]
    assert titles == [
        "Visita de María el domingo",
        "Cumpleaños de Antonio",
    ]
    assert "Cita con el médico" not in titles
    assert "Baile en la sala" not in titles


async def test_upcoming_excludes_cancelled_and_completed(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    repo = EventRepository(db_session)
    events = await repo.upcoming_for_person(
        seed.person_ids["carmen"], now=datetime.now(UTC)
    )
    assert all(
        event.status in (EventStatus.planned, EventStatus.confirmed)
        for event in events
    )
    assert all(event.start_at >= datetime.now(UTC) - timedelta(hours=1) for event in events)
