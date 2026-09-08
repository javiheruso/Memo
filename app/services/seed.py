"""Deterministic fictional universe used to seed example data.

Shared between the test fixtures (thin re-export from
``tests.fixtures.universe``) and the admin "load example data" feature, so the
production app can be populated with the same sample persons and records
without hand-crafting API calls. No real family data is used.

The data is aligned with the dossier fixtures (doc 07): Carmen, Antonio, María,
Javi, Paco/Francisco, and an ambiguity case where two people share the alias
"Paco".
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import (
    EventStatus,
    FactStatus,
    MediaType,
    MemoryStatus,
    RelationshipStatus,
    RelationshipType,
    SensitivityLevel,
    SourceType,
)
from app.models import Person
from app.repositories.event import EventRepository
from app.repositories.fact import FactRepository
from app.repositories.media import MediaRepository
from app.repositories.memory import MemoryRepository
from app.repositories.person import PersonRepository
from app.repositories.relationship import RelationshipRepository
from app.repositories.source import SourceRepository
from app.schemas.event import EventCreate, EventPersonCreate
from app.schemas.fact import FactCreate
from app.schemas.media import MediaCreate, MediaPersonCreate
from app.schemas.memory import MemoryCreate, MemoryPersonCreate
from app.schemas.person import PersonAliasCreate, PersonCreate
from app.schemas.relationship import RelationshipCreate
from app.schemas.source import SourceCreate


@dataclass
class SeedCounts:
    """How many rows of each reported entity the seed created."""

    persons: int = 0
    relationships: int = 0
    facts: int = 0
    memories: int = 0
    media: int = 0


@dataclass
class SeedUniverse:
    person_ids: dict[str, uuid.UUID] = field(default_factory=dict)
    source_ids: dict[str, uuid.UUID] = field(default_factory=dict)
    event_ids: dict[str, uuid.UUID] = field(default_factory=dict)
    media_ids: dict[str, uuid.UUID] = field(default_factory=dict)
    created: SeedCounts = field(default_factory=SeedCounts)


class AlreadySeededError(RuntimeError):
    """Raised when example data is requested but persons already exist."""

    def __init__(self) -> None:
        super().__init__("already seeded")


async def has_persons(session: AsyncSession) -> bool:
    """True when at least one person row exists in the database."""
    result = await session.execute(select(func.count()).select_from(Person))
    return result.scalar_one() > 0


async def seed_if_empty(session: AsyncSession) -> SeedUniverse:
    """Seed the fictional universe when the database holds no persons yet.

    Refuses with :class:`AlreadySeededError` when any person already exists so
    repeated calls never duplicate data.
    """
    if await has_persons(session):
        raise AlreadySeededError()
    return await seed_universe(session)


async def seed_universe(session: AsyncSession) -> SeedUniverse:
    universe = SeedUniverse()

    persons = {
        "carmen": PersonCreate(
            first_name="Carmen",
            last_name="Gómez",
            display_name="Carmen",
            description="Primary user; lives in Santa Pola.",
        ),
        "antonio": PersonCreate(
            first_name="Antonio",
            last_name="Gómez",
            display_name="Antonio",
            is_deceased=True,
            description="Carmen's late husband.",
        ),
        "paco_navarro": PersonCreate(
            first_name="Francisco",
            last_name="Navarro",
            display_name="Francisco Navarro",
            aliases=[PersonAliasCreate(alias="Paco", alias_type="nickname")],
            description="Old friend of Antonio.",
        ),
        "paco_garcia": PersonCreate(
            first_name="Paco",
            last_name="García",
            display_name="Paco García",
            aliases=[PersonAliasCreate(alias="Paco", alias_type="nickname")],
            description="Local baker, known as 'Paco el panadero'.",
        ),
        "maria": PersonCreate(
            first_name="María",
            last_name="Gómez",
            display_name="María",
            description="Daughter of Carmen.",
        ),
        "javi": PersonCreate(
            first_name="Javi",
            last_name="Gómez",
            display_name="Javi",
            description="Grandson of Carmen.",
        ),
        "luisa": PersonCreate(
            first_name="Luisa",
            last_name=None,
            display_name="Luisa",
            is_deceased=True,
            description=(
                "Reachable only through her stored photograph; no relations, "
                "facts or memories are recorded for her."
            ),
        ),
    }
    person_repo = PersonRepository(session)
    for key, data in persons.items():
        person = await person_repo.create(data)
        universe.person_ids[key] = person.id

    sources = {
        "family_records": SourceCreate(
            source_type=SourceType.family_member,
            person_id=universe.person_ids["maria"],
            description="Family records provided by María",
        ),
        "shared_by_carmen": SourceCreate(
            source_type=SourceType.user,
            person_id=universe.person_ids["carmen"],
            description="Shared directly by Carmen",
        ),
        "recorded_conversation": SourceCreate(
            source_type=SourceType.conversation,
            description="Extracted from a recorded conversation",
        ),
        "old_photos": SourceCreate(
            source_type=SourceType.photo,
            description="Old photo album",
        ),
        "seed": SourceCreate(
            source_type=SourceType.system,
            description="Fictional example-data universe seed",
        ),
    }
    source_repo = SourceRepository(session)
    for key, data in sources.items():
        source = await source_repo.create(data)
        universe.source_ids[key] = source.id

    media_repo = MediaRepository(session)
    media_items = [
        (
            "luisa_photo",
            MediaCreate(
                media_type=MediaType.photo,
                storage_path="seeds/luisa_old_photo.jpg",
                title="Foto antigua de Luisa",
                approx_date=date(1965, 7, 1),
                place_text="Santa Pola",
                source_id=universe.source_ids["old_photos"],
                people=[
                    MediaPersonCreate(
                        person_id=universe.person_ids["luisa"], is_primary=True
                    )
                ],
            ),
        ),
    ]
    for key, data in media_items:
        media = await media_repo.create(data)
        universe.media_ids[key] = media.id

    relationship_repo = RelationshipRepository(session)
    relationships = [
        (
            "carmen_antonio",
            RelationshipCreate(
                person_a_id=universe.person_ids["carmen"],
                person_b_id=universe.person_ids["antonio"],
                relationship_type=RelationshipType.spouse_of,
                status=RelationshipStatus.verified,
                source_id=universe.source_ids["family_records"],
            ),
        ),
        (
            "maria_carmen",
            RelationshipCreate(
                person_a_id=universe.person_ids["maria"],
                person_b_id=universe.person_ids["carmen"],
                relationship_type=RelationshipType.child_of,
                status=RelationshipStatus.verified,
                source_id=universe.source_ids["family_records"],
            ),
        ),
        (
            "javi_carmen",
            RelationshipCreate(
                person_a_id=universe.person_ids["javi"],
                person_b_id=universe.person_ids["carmen"],
                relationship_type=RelationshipType.grandchild_of,
                status=RelationshipStatus.verified,
                source_id=universe.source_ids["family_records"],
            ),
        ),
        (
            "paco_antonio",
            RelationshipCreate(
                person_a_id=universe.person_ids["paco_navarro"],
                person_b_id=universe.person_ids["antonio"],
                relationship_type=RelationshipType.friend_of,
                status=RelationshipStatus.verified,
                source_id=universe.source_ids["old_photos"],
            ),
        ),
        (
            "paco_garcia_carmen",
            RelationshipCreate(
                person_a_id=universe.person_ids["paco_garcia"],
                person_b_id=universe.person_ids["carmen"],
                relationship_type=RelationshipType.other,
                status=RelationshipStatus.probable,
                source_id=universe.source_ids["recorded_conversation"],
            ),
        ),
    ]
    for _key, data in relationships:
        await relationship_repo.create(data)

    fact_repo = FactRepository(session)
    facts = [
        FactCreate(
            subject_person_id=universe.person_ids["antonio"],
            fact_type="residence",
            value_text="Guardamar de la Safor",
            valid_from=date(1960, 1, 1),
            status=FactStatus.verified,
            source_id=universe.source_ids["family_records"],
        ),
        FactCreate(
            subject_person_id=universe.person_ids["carmen"],
            fact_type="move",
            value_text="Moved to Santa Pola",
            value_date=date(1985, 1, 1),
            status=FactStatus.verified,
            source_id=universe.source_ids["shared_by_carmen"],
        ),
        FactCreate(
            subject_person_id=universe.person_ids["antonio"],
            fact_type="hobby",
            value_text="Fishing",
            status=FactStatus.verified,
            source_id=universe.source_ids["shared_by_carmen"],
        ),
        FactCreate(
            subject_person_id=universe.person_ids["paco_garcia"],
            fact_type="occupation",
            value_text="Baker",
            status=FactStatus.probable,
            source_id=universe.source_ids["recorded_conversation"],
        ),
        FactCreate(
            subject_person_id=universe.person_ids["maria"],
            fact_type="residence",
            value_text="Alicante",
            status=FactStatus.verified,
            source_id=universe.source_ids["family_records"],
        ),
    ]
    for data in facts:
        await fact_repo.create(data)

    memory_repo = MemoryRepository(session)
    memories = [
        MemoryCreate(
            title="Fishing trips",
            narrative="Carmen remembers that Paco and Antonio used to go fishing together.",
            approx_period_text="late 1970s",
            place_text="Guardamar",
            memory_status=MemoryStatus.consolidated,
            sensitivity_level=SensitivityLevel.low,
            source_id=universe.source_ids["shared_by_carmen"],
            people=[
                MemoryPersonCreate(person_id=universe.person_ids["paco_navarro"], role="friend"),
                MemoryPersonCreate(person_id=universe.person_ids["antonio"], role="husband"),
            ],
        ),
        MemoryCreate(
            title="Dancing evenings",
            narrative="Carmen and Antonio went dancing at the local hall on Saturday nights.",
            approx_period_text="1970s",
            place_text="Santa Pola",
            memory_status=MemoryStatus.recurring,
            sensitivity_level=SensitivityLevel.low,
            source_id=universe.source_ids["shared_by_carmen"],
            people=[MemoryPersonCreate(person_id=universe.person_ids["antonio"])],
        ),
        MemoryCreate(
            title="First day of school",
            narrative="Carmen walked María to school on her first day.",
            memory_status=MemoryStatus.consolidated,
            sensitivity_level=SensitivityLevel.low,
            source_id=universe.source_ids["shared_by_carmen"],
            people=[MemoryPersonCreate(person_id=universe.person_ids["maria"])],
        ),
        MemoryCreate(
            title="Moving house",
            narrative="The family moved from Guardamar to Santa Pola in the mid-eighties.",
            approx_period_text="1985",
            memory_status=MemoryStatus.consolidated,
            sensitivity_level=SensitivityLevel.low,
            source_id=universe.source_ids["family_records"],
            people=[MemoryPersonCreate(person_id=universe.person_ids["antonio"])],
        ),
        MemoryCreate(
            title="Antonio's passing",
            narrative="The family remembers the time around Antonio's passing.",
            memory_status=MemoryStatus.consolidated,
            sensitivity_level=SensitivityLevel.high,
            source_id=universe.source_ids["family_records"],
            people=[MemoryPersonCreate(person_id=universe.person_ids["antonio"])],
        ),
        MemoryCreate(
            title="Bread from the bakery",
            narrative="Carmen remembers buying bread from Paco el panadero.",
            memory_status=MemoryStatus.candidate,
            sensitivity_level=SensitivityLevel.low,
            source_id=universe.source_ids["recorded_conversation"],
            people=[MemoryPersonCreate(person_id=universe.person_ids["paco_garcia"])],
        ),
        MemoryCreate(
            title="Summer at the beach",
            narrative="The family spent a summer at the beach with little Javi.",
            memory_status=MemoryStatus.candidate,
            sensitivity_level=SensitivityLevel.low,
            source_id=universe.source_ids["old_photos"],
            people=[MemoryPersonCreate(person_id=universe.person_ids["javi"])],
        ),
    ]
    for data in memories:
        await memory_repo.create(data)

    now = datetime.now(UTC)
    event_repo = EventRepository(session)
    events = [
        (
            "maria_sunday_visit",
            EventCreate(
                title="Visita de María el domingo",
                description="María viene a visitar a Carmen.",
                start_at=now + timedelta(days=2),
                end_at=now + timedelta(days=2, hours=4),
                status=EventStatus.confirmed,
                source_id=universe.source_ids["family_records"],
                people=[
                    EventPersonCreate(
                        person_id=universe.person_ids["maria"], role="guest"
                    ),
                    EventPersonCreate(
                        person_id=universe.person_ids["carmen"], role="host"
                    ),
                ],
            ),
        ),
        (
            "antonio_birthday",
            EventCreate(
                title="Cumpleaños de Antonio",
                start_at=now + timedelta(days=10),
                status=EventStatus.planned,
                source_id=universe.source_ids["family_records"],
                people=[
                    EventPersonCreate(
                        person_id=universe.person_ids["antonio"], role="celebrant"
                    ),
                    EventPersonCreate(
                        person_id=universe.person_ids["carmen"], role="host"
                    ),
                ],
            ),
        ),
        (
            "fishing_trip",
            EventCreate(
                title="Salida de pesca",
                description="Paco y Antonio salen a pescar a Guardamar.",
                start_at=now + timedelta(days=4),
                status=EventStatus.planned,
                source_id=universe.source_ids["shared_by_carmen"],
                people=[
                    EventPersonCreate(
                        person_id=universe.person_ids["paco_navarro"], role="friend"
                    ),
                    EventPersonCreate(
                        person_id=universe.person_ids["antonio"], role="friend"
                    ),
                ],
            ),
        ),
        (
            "saturday_dance",
            EventCreate(
                title="Baile en la sala",
                description="Baile del sábado en el salón local.",
                start_at=now - timedelta(days=5),
                status=EventStatus.completed,
                source_id=universe.source_ids["shared_by_carmen"],
                people=[
                    EventPersonCreate(
                        person_id=universe.person_ids["carmen"], role="attendant"
                    ),
                    EventPersonCreate(
                        person_id=universe.person_ids["antonio"], role="attendant"
                    ),
                ],
            ),
        ),
        (
            "doctor_appointment",
            EventCreate(
                title="Cita con el médico",
                start_at=now + timedelta(days=1),
                status=EventStatus.cancelled,
                source_id=universe.source_ids["shared_by_carmen"],
                people=[
                    EventPersonCreate(
                        person_id=universe.person_ids["carmen"], role="patient"
                    )
                ],
            ),
        ),
    ]
    for key, data in events:
        event = await event_repo.create(data)
        universe.event_ids[key] = event.id

    universe.created = SeedCounts(
        persons=len(persons),
        relationships=len(relationships),
        facts=len(facts),
        memories=len(memories),
        media=len(media_items),
    )

    return universe


__all__ = [
    "AlreadySeededError",
    "SeedCounts",
    "SeedUniverse",
    "has_persons",
    "seed_if_empty",
    "seed_universe",
]
