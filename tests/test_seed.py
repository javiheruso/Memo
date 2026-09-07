from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import RelationshipStatus, RelationshipType, SensitivityLevel
from app.models import (
    Event,
    Fact,
    Media,
    Memory,
    MemoryPerson,
    Person,
    Relationship,
    Source,
)
from app.repositories.media import MediaRepository
from app.repositories.memory import MemoryRepository
from app.repositories.person import PersonRepository
from app.repositories.relationship import RelationshipRepository
from tests.fixtures.universe import SeedUniverse


async def test_seed_counts(db_session: AsyncSession, seed: SeedUniverse) -> None:
    async def count(model) -> int:
        result = await db_session.execute(select(func.count()).select_from(model))
        return result.scalar_one()

    assert await count(Person) == 7
    assert await count(Source) == 5
    assert await count(Relationship) == 5
    assert await count(Fact) == 5
    assert await count(Memory) == 7
    assert await count(Event) == 5
    assert await count(Media) == 1


async def test_seed_luisa_is_reachable_only_through_her_photo(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    """Luisa has no relations, facts, memories or events; only a photo."""
    relationship_repo = RelationshipRepository(db_session)
    relationships = await relationship_repo.for_person(seed.person_ids["luisa"])
    assert relationships == []

    fact_results = await db_session.execute(
        select(Fact).where(Fact.subject_person_id == seed.person_ids["luisa"])
    )
    assert fact_results.scalars().all() == []

    memory_results = await db_session.execute(
        select(Memory)
        .join(MemoryPerson, MemoryPerson.memory_id == Memory.id)
        .where(MemoryPerson.person_id == seed.person_ids["luisa"])
    )
    assert memory_results.scalars().all() == []

    media = await MediaRepository(db_session).for_person(seed.person_ids["luisa"])
    assert len(media) == 1
    assert media[0].id == seed.media_ids["luisa_photo"]
    assert media[0].media_type == "photo"
    assert media[0].title == "Foto antigua de Luisa"


async def test_seed_paco_is_ambiguous(db_session: AsyncSession, seed: SeedUniverse) -> None:
    results = await PersonRepository(db_session).search_by_name("paco")
    assert len(results) == 2
    assert {p.id for p in results} == {
        seed.person_ids["paco_navarro"],
        seed.person_ids["paco_garcia"],
    }


async def test_seed_paco_navarro_verified_friend_of_antonio(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    relationships = await RelationshipRepository(db_session).list()
    matches = [
        r
        for r in relationships
        if r.person_a_id == seed.person_ids["paco_navarro"]
        and r.person_b_id == seed.person_ids["antonio"]
        and r.relationship_type == RelationshipType.friend_of
    ]
    assert len(matches) == 1
    assert matches[0].status == RelationshipStatus.verified


async def test_seed_paco_garcia_relationship_is_only_probable(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    relationships = await RelationshipRepository(db_session).list()
    matches = [
        r
        for r in relationships
        if r.person_a_id == seed.person_ids["paco_garcia"]
        and r.relationship_type == RelationshipType.other
    ]
    assert len(matches) == 1
    assert matches[0].status == RelationshipStatus.probable


async def test_seed_has_one_high_sensitivity_memory(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    memories = await MemoryRepository(db_session).list()
    sensitive = [m for m in memories if m.sensitivity_level == SensitivityLevel.high]
    assert len(sensitive) == 1
    assert sensitive[0].title == "Antonio's passing"
