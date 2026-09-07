import uuid

from sqlalchemy import func, or_, select
from sqlalchemy.orm import selectinload

from app.models import Person, PersonAlias
from app.repositories.base import SQLRepository
from app.schemas.person import PersonCreate


class PersonRepository(SQLRepository[Person]):
    model = Person
    load_options = (selectinload(Person.aliases),)

    async def create(self, data: PersonCreate) -> Person:
        display_name = data.display_name or " ".join(
            part for part in (data.first_name, data.last_name) if part
        )
        return await self._create(
            lambda: Person(
                first_name=data.first_name,
                last_name=data.last_name,
                display_name=display_name,
                birth_date=data.birth_date,
                death_date=data.death_date,
                is_deceased=data.is_deceased,
                description=data.description,
                aliases=[
                    PersonAlias(alias=alias.alias, alias_type=alias.alias_type)
                    for alias in data.aliases
                ],
            )
        )

    async def set_primary_media(
        self, person_id: uuid.UUID, media_id: uuid.UUID | None
    ) -> Person | None:
        """Point a person's photo-hint tier at a media row, or clear it.

        ``media_id`` may be ``None`` to clear the current primary. The media
        row is not required to have a ``media_people`` link; retrieval falls
        back to ``persons.primary_media_id`` directly.
        """
        person = await self.get(person_id)
        if person is None:
            return None
        person.primary_media_id = media_id
        await self._session.commit()
        return await self.get(person_id)

    async def search_by_name(self, name: str) -> list[Person]:
        term = name.strip().lower()
        if not term:
            return []
        stmt = (
            select(Person)
            .outerjoin(PersonAlias, PersonAlias.person_id == Person.id)
            .where(
                or_(
                    func.lower(PersonAlias.alias) == term,
                    func.lower(Person.first_name) == term,
                    func.lower(Person.last_name) == term,
                    func.lower(Person.display_name) == term,
                )
            )
            .options(selectinload(Person.aliases))
            .order_by(Person.id)
            .distinct()
        )
        result = await self._session.execute(stmt)
        return list(result.scalars())
