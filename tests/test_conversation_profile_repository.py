from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.person import PersonRepository
from app.repositories.profile import ConversationProfileRepository
from app.schemas.person import PersonCreate
from app.schemas.profile import ConversationProfileCreate, SensitiveTopicCreate


async def _person(session: AsyncSession, first_name: str):
    return await PersonRepository(session).create(PersonCreate(first_name=first_name))


async def test_create_active_profile_deactivates_previous_profile(
    db_session: AsyncSession,
) -> None:
    first_person = await _person(db_session, "Elena")
    second_person = await _person(db_session, "Carmen")
    profiles = ConversationProfileRepository(db_session)

    first = await profiles.create(
        ConversationProfileCreate(
            person_id=first_person.id,
            preferred_name="Elena",
            sensitive_topics=[
                SensitiveTopicCreate(topic="fallecimiento de Miguel")
            ],
            is_active=True,
        )
    )
    assert await profiles.has_active() is True
    assert (await profiles.get_active()).id == first.id
    assert (await profiles.get_by_person(first_person.id)).id == first.id

    second = await profiles.create(
        ConversationProfileCreate(person_id=second_person.id, is_active=True)
    )
    assert (await profiles.get_active()).id == second.id
    assert (await profiles.get(first.id)).is_active is False
    assert (await profiles.get(second.id)).is_active is True


async def test_activate_switches_the_single_active_profile(
    db_session: AsyncSession,
) -> None:
    first_person = await _person(db_session, "Elena")
    second_person = await _person(db_session, "Carmen")
    profiles = ConversationProfileRepository(db_session)
    first = await profiles.create(
        ConversationProfileCreate(person_id=first_person.id, is_active=True)
    )
    second = await profiles.create(
        ConversationProfileCreate(person_id=second_person.id)
    )
    assert (await profiles.get_active()).id == first.id

    activated = await profiles.activate(second.id)

    assert activated is not None
    assert activated.is_active is True
    assert (await profiles.get_active()).id == second.id
    assert (await profiles.get(first.id)).is_active is False
