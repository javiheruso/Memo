import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.person import PersonRepository
from app.retrieval.entity_resolver import EntityResolver, ResolutionKind
from tests.fixtures.universe import SeedUniverse


@pytest.fixture
def resolver(db_session: AsyncSession) -> EntityResolver:
    return EntityResolver(PersonRepository(db_session))


async def test_resolve_known_person_by_first_name(resolver, seed: SeedUniverse):
    resolution = await resolver.resolve("Francisco")
    assert resolution.kind == ResolutionKind.resolved
    assert resolution.person is not None
    assert resolution.person.id == seed.person_ids["paco_navarro"]


async def test_resolve_by_display_name(resolver, seed: SeedUniverse):
    resolution = await resolver.resolve("Paco García")
    assert resolution.kind == ResolutionKind.resolved
    assert resolution.person is not None
    assert resolution.person.id == seed.person_ids["paco_garcia"]


async def test_resolve_alias_is_ambiguous_with_all_candidates(
    resolver, seed: SeedUniverse
):
    resolution = await resolver.resolve("Paco")
    assert resolution.kind == ResolutionKind.ambiguous
    assert resolution.person is None
    assert {candidate.id for candidate in resolution.candidates} == {
        seed.person_ids["paco_navarro"],
        seed.person_ids["paco_garcia"],
    }


async def test_resolve_unknown_person(resolver):
    resolution = await resolver.resolve("Eustaquio")
    assert resolution.kind == ResolutionKind.not_found
    assert resolution.candidates == []
    assert resolution.person is None
