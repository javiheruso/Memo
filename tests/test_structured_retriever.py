import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import EpistemicKind, MediaType, RelationshipType
from app.repositories.event import EventRepository
from app.repositories.fact import FactRepository
from app.repositories.media import MediaRepository
from app.repositories.memory import MemoryRepository
from app.repositories.person import PersonRepository
from app.repositories.relationship import RelationshipRepository
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.entity_resolver import EntityResolver, ResolutionKind
from app.retrieval.structured_retriever import StructuredRetriever
from app.schemas.media import MediaCreate
from tests.fixtures.universe import SeedUniverse


@pytest.fixture
def retriever(db_session: AsyncSession) -> StructuredRetriever:
    return StructuredRetriever(
        person_repo=PersonRepository(db_session),
        relationship_repo=RelationshipRepository(db_session),
        fact_repo=FactRepository(db_session),
        memory_repo=MemoryRepository(db_session),
        event_repo=EventRepository(db_session),
        media_repo=MediaRepository(db_session),
    )


@pytest.fixture
def resolver(db_session: AsyncSession) -> EntityResolver:
    return EntityResolver(PersonRepository(db_session))


async def test_retrieve_relationships_facts_memories(retriever, seed: SeedUniverse):
    data = await retriever.retrieve(seed.person_ids["antonio"])
    assert len(data.relationships) == 2
    assert len(data.facts) == 2
    assert len(data.memories) == 4
    assert {fact.fact_type for fact in data.facts} == {"residence", "hobby"}
    assert len(data.events) == 3


async def test_retrieve_includes_events(retriever, seed: SeedUniverse):
    data = await retriever.retrieve(seed.person_ids["maria"])
    assert [event.title for event in data.events] == ["Visita de María el domingo"]


async def test_paco_navarro_has_verified_friend_relationship(
    retriever, seed: SeedUniverse
):
    data = await retriever.retrieve(seed.person_ids["paco_navarro"])
    assert len(data.relationships) == 1
    assert data.relationships[0].relationship_type == RelationshipType.friend_of
    assert data.memories  # the fishing-trip memory is linked to Paco Navarro


async def test_context_marks_memories_as_memory_not_fact(
    retriever, resolver, seed: SeedUniverse
):
    resolution = await resolver.resolve("Antonio")
    assert resolution.kind == ResolutionKind.resolved
    data = await retriever.retrieve(resolution.person.id)
    context = ContextBuilder().build_from_person_data(data, resolution)

    assert all(memory.epistemic == EpistemicKind.memory for memory in context.memories)
    assert all(
        relation.epistemic == EpistemicKind.verified_fact
        for relation in context.verified_relations
    )
    assert any(
        relation.relationship_type == RelationshipType.spouse_of
        for relation in context.verified_relations
    )


async def test_sensitive_memories_filtered_out_of_context(
    retriever, resolver, seed: SeedUniverse
):
    resolution = await resolver.resolve("Antonio")
    data = await retriever.retrieve(resolution.person.id)
    context = ContextBuilder().build_from_person_data(data, resolution)

    titles = {memory.title for memory in context.memories}
    assert "Antonio's passing" not in titles
    assert any("sensitive" in policy for policy in context.policies)


async def test_luisa_is_reachable_only_through_her_photo_metadata(
    retriever, resolver, seed: SeedUniverse
):
    resolution = await resolver.resolve("Luisa")
    assert resolution.kind == ResolutionKind.resolved
    data = await retriever.retrieve(resolution.person.id)

    assert data.relationships == []
    assert data.facts == []
    assert data.memories == []
    assert len(data.media) == 1
    photo = data.media[0]
    assert photo.media_type == "photo"
    assert photo.approx_date is not None
    assert photo.place_text == "Santa Pola"

    context = ContextBuilder().build_from_person_data(data, resolution)
    assert len(context.media) == 1
    assert context.media[0].media_type == "photo"
    assert context.media[0].place_text == "Santa Pola"


async def test_primary_media_used_without_media_people_row(
    retriever, db_session: AsyncSession, seed: SeedUniverse
):
    """A primary photo with no ``media_people`` link still reaches retrieval.

    ``persons.primary_media_id`` is the only association, so the retriever must
    fall back to it and surface the row first in ``PersonData.media``.
    """
    person_repo = PersonRepository(db_session)
    photo = await MediaRepository(db_session).create(
        MediaCreate(
            media_type=MediaType.photo,
            storage_path="albums/javi_primary.jpg",
            title="Foto de Javi",
        )
    )
    person = await person_repo.set_primary_media(seed.person_ids["javi"], photo.id)
    assert person is not None
    assert person.primary_media_id == photo.id

    data = await retriever.retrieve(seed.person_ids["javi"])
    assert [item.id for item in data.media] == [photo.id]
