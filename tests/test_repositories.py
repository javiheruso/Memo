import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import (
    FactStatus,
    MemoryStatus,
    RelationshipStatus,
    RelationshipType,
    SensitivityLevel,
    SourceType,
)
from app.models import Source
from app.repositories.fact import FactRepository
from app.repositories.memory import MemoryRepository
from app.repositories.person import PersonRepository
from app.repositories.relationship import RelationshipRepository
from app.repositories.source import SourceRepository
from app.schemas.fact import FactCreate
from app.schemas.memory import MemoryCreate, MemoryPersonCreate
from app.schemas.person import PersonAliasCreate, PersonCreate
from app.schemas.relationship import RelationshipCreate
from app.schemas.source import SourceCreate


async def _make_source(db_session: AsyncSession, *, source_type: SourceType = SourceType.system) -> Source:
    return await SourceRepository(db_session).create(
        SourceCreate(source_type=source_type, description="test source")
    )


async def test_person_create_derives_display_name(db_session: AsyncSession) -> None:
    person = await PersonRepository(db_session).create(
        PersonCreate(first_name="Carmen", last_name="Gómez")
    )
    assert person.display_name == "Carmen Gómez"
    assert person.created_at is not None
    assert person.is_deceased is False


async def test_person_create_with_aliases_roundtrip(db_session: AsyncSession) -> None:
    repo = PersonRepository(db_session)
    person = await repo.create(
        PersonCreate(
            first_name="Francisco",
            last_name="Navarro",
            aliases=[PersonAliasCreate(alias="Paco", alias_type="nickname")],
        )
    )
    assert [alias.alias for alias in person.aliases] == ["Paco"]

    fetched = await repo.get(person.id)
    assert fetched is not None
    assert [alias.alias for alias in fetched.aliases] == ["Paco"]


async def test_person_search_by_alias(db_session: AsyncSession) -> None:
    repo = PersonRepository(db_session)
    created = await repo.create(
        PersonCreate(
            first_name="Francisco",
            last_name="Navarro",
            aliases=[PersonAliasCreate(alias="Paco", alias_type="nickname")],
        )
    )
    results = await repo.search_by_name("paco")
    assert [p.id for p in results] == [created.id]


async def test_person_search_by_first_name_case_insensitive(db_session: AsyncSession) -> None:
    repo = PersonRepository(db_session)
    created = await repo.create(PersonCreate(first_name="Carmen"))
    results = await repo.search_by_name("CARMEN")
    assert [p.id for p in results] == [created.id]


async def test_person_search_returns_ambiguous_matches(db_session: AsyncSession) -> None:
    repo = PersonRepository(db_session)
    first = await repo.create(
        PersonCreate(
            first_name="Francisco",
            last_name="Navarro",
            aliases=[PersonAliasCreate(alias="Paco", alias_type="nickname")],
        )
    )
    second = await repo.create(
        PersonCreate(
            first_name="Paco",
            last_name="García",
            aliases=[PersonAliasCreate(alias="Paco", alias_type="nickname")],
        )
    )
    results = await repo.search_by_name("paco")
    assert sorted(r.id for r in results) == sorted([first.id, second.id])


async def test_person_search_is_accent_insensitive(db_session: AsyncSession) -> None:
    repo = PersonRepository(db_session)
    created = await repo.create(
        PersonCreate(first_name="Paco", last_name="García")
    )
    assert [p.id for p in await repo.search_by_name("garcia")] == [created.id]
    assert [p.id for p in await repo.search_by_name("GARCÍA")] == [created.id]


async def test_person_search_matches_alias_plus_last_name(db_session: AsyncSession) -> None:
    repo = PersonRepository(db_session)
    navarro = await repo.create(
        PersonCreate(
            first_name="Francisco",
            last_name="Navarro",
            aliases=[PersonAliasCreate(alias="Paco", alias_type="nickname")],
        )
    )
    garcia = await repo.create(
        PersonCreate(
            first_name="Paco",
            last_name="García",
            aliases=[PersonAliasCreate(alias="Paco", alias_type="nickname")],
        )
    )
    # "paco navarro" must resolve to the Navarro person (alias + last name),
    # and "paco garcia" to the García person (alias + accent-folded last name).
    assert [p.id for p in await repo.search_by_name("paco navarro")] == [navarro.id]
    assert [p.id for p in await repo.search_by_name("paco garcia")] == [garcia.id]


async def test_person_search_unknown_returns_empty(db_session: AsyncSession) -> None:
    results = await PersonRepository(db_session).search_by_name("nobody")
    assert results == []


async def test_person_search_empty_term_returns_empty(db_session: AsyncSession) -> None:
    results = await PersonRepository(db_session).search_by_name("   ")
    assert results == []


async def test_source_create_and_list(db_session: AsyncSession) -> None:
    repo = SourceRepository(db_session)
    source = await repo.create(SourceCreate(source_type=SourceType.photo, description="album"))
    assert source.source_type == SourceType.photo
    listed = await repo.list()
    assert [s.id for s in listed] == [source.id]


async def test_relationship_create_defaults_to_verified(db_session: AsyncSession) -> None:
    person_repo = PersonRepository(db_session)
    a = await person_repo.create(PersonCreate(first_name="Antonio"))
    b = await person_repo.create(PersonCreate(first_name="Paco"))
    source = await _make_source(db_session)

    rel = await RelationshipRepository(db_session).create(
        RelationshipCreate(
            person_a_id=a.id,
            person_b_id=b.id,
            relationship_type=RelationshipType.friend_of,
            source_id=source.id,
        )
    )
    assert rel.status == RelationshipStatus.verified
    listed = await RelationshipRepository(db_session).list()
    assert len(listed) == 1
    assert listed[0].id == rel.id


async def test_relationship_rejects_self_reference(db_session: AsyncSession) -> None:
    person = await PersonRepository(db_session).create(PersonCreate(first_name="Carmen"))
    with pytest.raises(ValueError):
        await RelationshipRepository(db_session).create(
            RelationshipCreate(
                person_a_id=person.id,
                person_b_id=person.id,
                relationship_type=RelationshipType.other,
            )
        )


async def test_fact_default_status_is_pending(db_session: AsyncSession) -> None:
    person = await PersonRepository(db_session).create(PersonCreate(first_name="Antonio"))
    source = await _make_source(db_session)
    fact = await FactRepository(db_session).create(
        FactCreate(
            subject_person_id=person.id,
            fact_type="hobby",
            value_text="Fishing",
            source_id=source.id,
        )
    )
    assert fact.status == FactStatus.pending
    assert fact.value_text == "Fishing"


async def test_memory_defaults_and_people_roundtrip(db_session: AsyncSession) -> None:
    person = await PersonRepository(db_session).create(PersonCreate(first_name="Paco"))
    source = await _make_source(db_session)
    repo = MemoryRepository(db_session)
    memory = await repo.create(
        MemoryCreate(
            narrative="We went fishing together.",
            source_id=source.id,
            people=[MemoryPersonCreate(person_id=person.id, role="friend")],
        )
    )
    assert memory.memory_status == MemoryStatus.candidate
    assert memory.sensitivity_level == SensitivityLevel.low
    assert [(mp.person_id, mp.role) for mp in memory.people] == [(person.id, "friend")]

    fetched = await repo.get(memory.id)
    assert fetched is not None
    assert len(fetched.people) == 1
