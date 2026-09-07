import uuid
from dataclasses import dataclass, field

from app.models import Event, Fact, Media, Memory, Person, Relationship
from app.repositories.event import EventRepository
from app.repositories.fact import FactRepository
from app.repositories.media import MediaRepository
from app.repositories.memory import MemoryRepository
from app.repositories.person import PersonRepository
from app.repositories.relationship import RelationshipRepository


@dataclass(frozen=True)
class PersonData:
    person: Person
    relationships: list[Relationship]
    facts: list[Fact]
    memories: list[Memory]
    events: list[Event]
    media: list[Media] = field(default_factory=list)


class StructuredRetriever:
    """Fetch the structured data about one resolved person.

    Only exact, sourced entities are returned; each row keeps its database
    status so the context builder can assign the right epistemic kind.
    Media rows are metadata only and are never interpreted here.
    Semantic/similarity retrieval is intentionally out of scope for now.
    """

    def __init__(
        self,
        *,
        person_repo: PersonRepository,
        relationship_repo: RelationshipRepository,
        fact_repo: FactRepository,
        memory_repo: MemoryRepository,
        event_repo: EventRepository,
        media_repo: MediaRepository,
    ) -> None:
        self._person_repo = person_repo
        self._relationship_repo = relationship_repo
        self._fact_repo = fact_repo
        self._memory_repo = memory_repo
        self._event_repo = event_repo
        self._media_repo = media_repo

    async def retrieve(self, person_id: uuid.UUID) -> PersonData:
        person = await self._person_repo.get(person_id)
        if person is None:
            raise ValueError(f"Unknown person id: {person_id}")
        relationships = await self._relationship_repo.for_person(person_id)
        facts = await self._fact_repo.for_subject_person(person_id)
        memories = await self._memory_repo.for_person(person_id)
        events = await self._event_repo.for_person(person_id)
        media = await self._media_repo.for_person(person_id)
        if person.primary_media_id is not None and not any(
            item.id == person.primary_media_id for item in media
        ):
            primary = await self._media_repo.get(person.primary_media_id)
            if primary is not None:
                media.insert(0, primary)
        return PersonData(
            person=person,
            relationships=relationships,
            facts=facts,
            memories=memories,
            events=events,
            media=media,
        )
