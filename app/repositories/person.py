import unicodedata
import uuid

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.models import Person, PersonAlias
from app.repositories.base import SQLRepository
from app.schemas.person import PersonCreate


def _fold(value: str | None) -> str:
    """Lowercase and strip accents for accent-insensitive matching."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value).lower()
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


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
        """Resolve a person by name/alias, accent- and token-aware.

        The match is done in Python (the dataset is a personal, small corpus)
        so accents can be folded ("garcia" matches "García") and multi-word
        terms can match across fields ("paco navarro" matches the alias "Paco"
        plus the last name "Navarro"). Exact field matches are returned before
        token-subset matches.
        """
        term = _fold(name).strip()
        if not term:
            return []
        persons = await self._list_all()
        term_tokens = set(term.split())

        exact: list[Person] = []
        token_matches: list[Person] = []
        for person in persons:
            fields = {
                _fold(person.first_name),
                _fold(person.last_name),
                _fold(person.display_name),
            }
            for alias in person.aliases:
                fields.add(_fold(alias.alias))
            fields = {field for field in fields if field}

            if term in fields:
                exact.append(person)
                continue

            person_tokens: set[str] = set()
            for field in fields:
                person_tokens.update(field.split())
            if term_tokens and term_tokens.issubset(person_tokens):
                token_matches.append(person)

        result = exact + token_matches
        seen: set[uuid.UUID] = set()
        unique: list[Person] = []
        for person in result:
            if person.id not in seen:
                seen.add(person.id)
                unique.append(person)
        return unique

    async def _list_all(self) -> list[Person]:
        stmt = select(Person).options(*self.load_options).order_by(Person.id)
        result = await self._session.execute(stmt)
        return list(result.scalars())
