import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.domain.enums import EventStatus
from app.models import Event, EventPerson
from app.repositories.base import SQLRepository
from app.schemas.event import EventCreate


class EventRepository(SQLRepository[Event]):
    model = Event
    load_options = (
        selectinload(Event.people).selectinload(EventPerson.person),
    )

    async def create(self, data: EventCreate) -> Event:
        return await self._create(
            lambda: Event(
                title=data.title,
                description=data.description,
                start_at=data.start_at,
                end_at=data.end_at,
                status=data.status,
                source_id=data.source_id,
                people=[
                    EventPerson(person_id=person.person_id, role=person.role)
                    for person in data.people
                ],
            )
        )

    async def for_person(self, person_id: uuid.UUID) -> list[Event]:
        stmt = (
            select(Event)
            .join(EventPerson, EventPerson.event_id == Event.id)
            .where(EventPerson.person_id == person_id)
            .options(*self.load_options)
            .order_by(Event.start_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())

    async def upcoming_for_person(
        self, person_id: uuid.UUID, *, now: datetime | None = None
    ) -> list[Event]:
        moment = now or datetime.now(UTC)
        stmt = (
            select(Event)
            .join(EventPerson, EventPerson.event_id == Event.id)
            .where(
                EventPerson.person_id == person_id,
                Event.start_at >= moment,
                Event.status.in_([EventStatus.planned, EventStatus.confirmed]),
            )
            .options(*self.load_options)
            .order_by(Event.start_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())
