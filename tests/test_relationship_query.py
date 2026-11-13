"""Explicit relationship-role questions over the shared conversation flow.

These tests cover the reported "mi marido" / "mi esposo" case: after a person
has been safely resolved in the current conversation, an explicit role question
must answer from that person's stored relationships, never from the role words
themselves and never by guessing across sessions or trust states.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import Intent, RelationshipStatus, RelationshipType, SourceType
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.deterministic import DeterministicResponder
from app.repositories.conversation import ConversationAccessDenied
from app.repositories.person import PersonRepository
from app.repositories.profile import ConversationProfileRepository
from app.repositories.relationship import RelationshipRepository
from app.repositories.source import SourceRepository
from app.schemas.person import PersonCreate
from app.schemas.profile import ConversationProfileCreate, SensitiveTopicCreate
from app.schemas.relationship import RelationshipCreate
from app.schemas.source import SourceCreate
from app.services.conversation import ConversationService
from app.voice.service import VoiceService


def _service(db: AsyncSession) -> ConversationService:
    return ConversationService(
        db, DeterministicResponder(), embedder=DeterministicEmbedder()
    )


async def _seed_elena(
    db: AsyncSession,
    *other_names: str,
    relationship_type: RelationshipType = RelationshipType.spouse_of,
    status: RelationshipStatus = RelationshipStatus.verified,
):
    persons = PersonRepository(db)
    elena = await persons.create(PersonCreate(first_name="Elena"))
    others = [
        await persons.create(PersonCreate(first_name=name)) for name in other_names
    ]
    source = await SourceRepository(db).create(
        SourceCreate(
            source_type=SourceType.family_member,
            person_id=elena.id,
            description="Family records",
        )
    )
    relationships = []
    for other in others:
        relationships.append(
            await RelationshipRepository(db).create(
                RelationshipCreate(
                    person_a_id=elena.id,
                    person_b_id=other.id,
                    relationship_type=relationship_type,
                    status=status,
                    source_id=source.id,
                )
            )
        )
    return elena, others, relationships, source


async def _resolve_elena(service: ConversationService):
    first = await service.handle_user_message("¿Quién era Elena?")
    assert first.intent == Intent.person_query
    return first


async def test_active_person_spouse_query_returns_miguel_with_provenance(
    db_session: AsyncSession,
) -> None:
    _, _, relationships, seed_source = await _seed_elena(db_session, "Miguel")
    service = _service(db_session)
    first = await _resolve_elena(service)

    result = await service.handle_user_message(
        "¿Quién es mi marido?", conversation_id=first.conversation_id
    )

    assert result.intent == Intent.person_query
    assert "Miguel" in result.answer
    assert "marido" in result.answer
    assert "no está confirmado" not in result.answer
    assert "Elena" not in result.answer

    usage_types = {(source.entity_type, source.usage_type) for source in result.sources}
    assert ("person", "subject") in usage_types
    assert ("person", "related") in usage_types
    assert ("relationship", "verified_fact") in usage_types
    assert ("source", "provenance") in usage_types
    assert any(source.entity_id == relationships[0].id for source in result.sources)
    assert any(source.entity_id == seed_source.id for source in result.sources)


async def test_esposo_variant_and_no_claimed_verification_when_probable(
    db_session: AsyncSession,
) -> None:
    await _seed_elena(
        db_session,
        "Miguel",
        status=RelationshipStatus.probable,
    )
    service = _service(db_session)
    first = await _resolve_elena(service)

    result = await service.handle_user_message(
        "¿Quién es mi esposo?", conversation_id=first.conversation_id
    )

    assert "Miguel" in result.answer
    assert "Creo que" in result.answer
    assert "no está confirmado" in result.answer
    usage_types = {(source.entity_type, source.usage_type) for source in result.sources}
    assert ("relationship", "probable_fact") in usage_types
    assert ("relationship", "verified_fact") not in usage_types


async def test_direct_name_still_resolves_by_name(
    db_session: AsyncSession,
) -> None:
    await _seed_elena(db_session, "Miguel")
    service = _service(db_session)
    first = await _resolve_elena(service)

    result = await service.handle_user_message(
        "¿Quién era Miguel?", conversation_id=first.conversation_id
    )

    assert result.intent == Intent.person_query
    assert "Miguel" in result.answer
    assert "No tengo guardado" not in result.answer
    assert "Tu marido" not in result.answer


async def test_no_active_person_never_guesses_a_spouse(
    db_session: AsyncSession,
) -> None:
    await _seed_elena(db_session, "Miguel")
    service = _service(db_session)

    result = await service.handle_user_message("¿Quién es mi marido?")

    assert "Miguel" not in result.answer
    assert "Antes de responder" in result.answer
    assert result.gap is None
    assert result.sources == []


async def test_absent_relationship_is_reported_without_invention(
    db_session: AsyncSession,
) -> None:
    await _seed_elena(db_session)
    service = _service(db_session)
    first = await _resolve_elena(service)

    result = await service.handle_user_message(
        "¿Quién es mi marido?", conversation_id=first.conversation_id
    )

    assert result.answer == "No tengo guardado quién es tu marido."
    assert [(source.entity_type, source.usage_type) for source in result.sources] == [
        ("person", "subject")
    ]
    assert result.gap is None


async def test_multiple_verified_spouses_are_not_collapsed(
    db_session: AsyncSession,
) -> None:
    await _seed_elena(db_session, "Miguel", "Pedro")
    service = _service(db_session)
    first = await _resolve_elena(service)

    result = await service.handle_user_message(
        "¿Quién es mi marido?", conversation_id=first.conversation_id
    )

    assert "Miguel" in result.answer
    assert "Pedro" in result.answer
    assert "No sé cuál es" in result.answer
    assert "Tu marido es " not in result.answer


async def test_pending_relationship_is_withheld(
    db_session: AsyncSession,
) -> None:
    await _seed_elena(
        db_session,
        "Miguel",
        status=RelationshipStatus.pending,
    )
    service = _service(db_session)
    first = await _resolve_elena(service)

    result = await service.handle_user_message(
        "¿Quién es mi marido?", conversation_id=first.conversation_id
    )

    assert "Miguel" not in result.answer
    assert result.answer == "No tengo guardado quién es tu marido."
    assert [(source.entity_type, source.usage_type) for source in result.sources] == [
        ("person", "subject")
    ]


async def test_relationship_query_is_isolated_by_owner_session(
    db_session: AsyncSession,
) -> None:
    await _seed_elena(db_session, "Miguel")
    owner = _service(db_session)
    first = await owner.handle_user_message(
        "¿Quién era Elena?", owner_session_id="session-a"
    )

    with pytest.raises(ConversationAccessDenied):
        await _service(db_session).handle_user_message(
            "¿Quién es mi marido?",
            conversation_id=first.conversation_id,
            owner_session_id="session-b",
        )


async def test_sensitive_topic_control_still_wins_over_relationship_lookup(
    db_session: AsyncSession,
) -> None:
    elena, _, _, _ = await _seed_elena(db_session, "Miguel")
    await ConversationProfileRepository(db_session).create(
        ConversationProfileCreate(
            person_id=elena.id,
            sensitive_topics=[
                SensitiveTopicCreate(
                    topic="marido",
                    handling="Do not invent details.",
                    triggers=["marido"],
                )
            ],
            is_active=True,
        )
    )
    service = _service(db_session)
    first = await _resolve_elena(service)

    result = await service.handle_user_message(
        "¿Quién es mi marido?", conversation_id=first.conversation_id
    )

    assert "No quiero inventar detalles" in result.answer
    assert "Miguel" not in result.answer
    assert result.sources == []


async def test_voice_flow_shares_relationship_resolution(
    db_session: AsyncSession,
) -> None:
    await _seed_elena(db_session, "Miguel")

    class FakeSTT:
        async def transcribe(self, audio: bytes, *, content_type=None) -> str:
            return "¿Quién era Elena?"

    voice = VoiceService(
        db_session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
        stt=FakeSTT(),
    )
    first = await voice.process(audio=b"\x00wav", audio_content_type="audio/wav")
    second = await voice.process(
        text="¿Quién es mi marido?",
        conversation_id=first.conversation_id,
    )

    assert second.transcript is None
    assert "Miguel" in second.answer
    assert "marido" in second.answer


# --------------------------------------------------------------------------
# State-gated relative follow-ups ("¿Y su hija/madre?")
# --------------------------------------------------------------------------


async def _seed_relative(
    db: AsyncSession,
    *,
    relationship_type: RelationshipType,
    status: RelationshipStatus = RelationshipStatus.verified,
    active_name: str = "Carmen",
    other_name: str = "María",
    active_is_person_a: bool = True,
):
    persons = PersonRepository(db)
    active = await persons.create(PersonCreate(first_name=active_name))
    other = await persons.create(PersonCreate(first_name=other_name))
    source = await SourceRepository(db).create(
        SourceCreate(
            source_type=SourceType.family_member,
            person_id=active.id,
            description="Family records",
        )
    )
    person_a, person_b = (
        (active, other) if active_is_person_a else (other, active)
    )
    relationship = await RelationshipRepository(db).create(
        RelationshipCreate(
            person_a_id=person_a.id,
            person_b_id=person_b.id,
            relationship_type=relationship_type,
            status=status,
            source_id=source.id,
        )
    )
    return active, other, relationship, source


async def _resolve_active(service: ConversationService, name: str = "Carmen"):
    result = await service.handle_user_message(f"¿Quién era {name}?")
    assert result.intent == Intent.person_query
    return result


def _source_keys(result):
    return {(source.entity_type, str(source.entity_id)) for source in result.sources}


async def test_child_relative_follow_up_is_neutral_and_keeps_direction(
    db_session: AsyncSession,
) -> None:
    active, child, relationship, seed_source = await _seed_relative(
        db_session,
        relationship_type=RelationshipType.child_of,
        active_is_person_a=False,
    )
    service = _service(db_session)
    first = await _resolve_active(service, active.first_name)

    daughter = await service.handle_user_message(
        "¿Y su hija?", conversation_id=first.conversation_id
    )
    son = await service.handle_user_message(
        "¿Y su hijo?", conversation_id=first.conversation_id
    )

    for result in (daughter, son):
        assert result.intent == Intent.person_query
        assert result.answer.startswith(
            "Tengo guardada a María como descendiente de Carmen"
        )
        assert "el parentesco guardado no especifica el sexo ni el rol" in (
            result.answer
        )
        assert "hija" not in result.answer
        assert "hijo" not in result.answer
        keys = _source_keys(result)
        assert ("person", str(active.id)) in keys
        assert ("person", str(child.id)) in keys
        assert ("relationship", str(relationship.id)) in keys
        assert ("source", str(seed_source.id)) in keys


async def test_child_relative_follow_up_does_not_confirm_asked_daughter_role(
    db_session: AsyncSession,
) -> None:
    """A verified child_of row is proof of a descendant, not of a daughter.

    The user asked for ``hija`` but the stored relation carries no sex or role
    evidence, so the answer must state that limit instead of affirming the
    gendered role. Direction and provenance stay intact.
    """
    active, child, relationship, seed_source = await _seed_relative(
        db_session,
        relationship_type=RelationshipType.child_of,
        active_is_person_a=False,
    )
    service = _service(db_session)
    first = await _resolve_active(service, active.first_name)

    result = await service.handle_user_message(
        "¿Y su hija?", conversation_id=first.conversation_id
    )

    assert "descendiente de Carmen" in result.answer
    assert "el parentesco guardado no especifica el sexo ni el rol" in result.answer
    assert "es hija" not in result.answer.lower()
    assert "es hijo" not in result.answer.lower()
    assert "hija de Carmen" not in result.answer.lower()
    keys = _source_keys(result)
    assert ("person", str(active.id)) in keys
    assert ("person", str(child.id)) in keys
    assert ("relationship", str(relationship.id)) in keys
    assert ("source", str(seed_source.id)) in keys


async def test_parent_relative_follow_up_is_neutral_and_not_reversed(
    db_session: AsyncSession,
) -> None:
    active, parent, relationship, _ = await _seed_relative(
        db_session,
        relationship_type=RelationshipType.child_of,
        active_is_person_a=True,
    )
    service = _service(db_session)
    first = await _resolve_active(service, active.first_name)

    mother = await service.handle_user_message(
        "¿Y su madre?", conversation_id=first.conversation_id
    )
    father = await service.handle_user_message(
        "¿Y su padre?", conversation_id=first.conversation_id
    )
    # The stored row says Carmen is the child, not the parent, so a daughter
    # question must not turn the parent into a daughter.
    daughter = await service.handle_user_message(
        "¿Y su hija?", conversation_id=first.conversation_id
    )

    for result in (mother, father):
        assert result.intent == Intent.person_query
        assert result.answer.startswith(
            "Tengo guardada a María como progenitor de Carmen"
        )
        assert "madre" not in result.answer
        assert "padre" not in result.answer
        assert ("relationship", str(relationship.id)) in _source_keys(result)

    assert daughter.answer == "No tengo guardado quién es su hija."
    assert "María" not in daughter.answer


async def test_relative_follow_up_requires_previous_resolved_person(
    db_session: AsyncSession,
) -> None:
    await _seed_relative(
        db_session,
        relationship_type=RelationshipType.child_of,
        active_is_person_a=False,
    )
    service = _service(db_session)

    result = await service.handle_user_message("¿Y su hija?")

    assert result.intent == Intent.general_chat
    assert result.sources == []
    assert "María" not in result.answer


async def test_relative_follow_up_does_not_carry_across_a_stale_turn(
    db_session: AsyncSession,
) -> None:
    await _seed_relative(
        db_session,
        relationship_type=RelationshipType.child_of,
        active_is_person_a=False,
    )
    service = _service(db_session)
    first = await _resolve_active(service)

    stale = await service.handle_user_message(
        "Hoy hace buen día.", conversation_id=first.conversation_id
    )
    result = await service.handle_user_message(
        "¿Y su hija?", conversation_id=stale.conversation_id
    )

    assert result.intent == Intent.general_chat
    assert result.sources == []
    assert "María" not in result.answer


async def test_relative_follow_up_does_not_collapse_multiple_people(
    db_session: AsyncSession,
) -> None:
    active, child, first_relationship, _ = await _seed_relative(
        db_session,
        relationship_type=RelationshipType.child_of,
        active_is_person_a=False,
    )
    second_child = await PersonRepository(db_session).create(
        PersonCreate(first_name="Pedro")
    )
    second_relationship = await RelationshipRepository(db_session).create(
        RelationshipCreate(
            person_a_id=second_child.id,
            person_b_id=active.id,
            relationship_type=RelationshipType.child_of,
            status=RelationshipStatus.verified,
        )
    )
    service = _service(db_session)
    first = await _resolve_active(service)

    result = await service.handle_user_message(
        "¿Y su hija?", conversation_id=first.conversation_id
    )

    assert result.intent == Intent.person_query
    assert "María" in result.answer
    assert "Pedro" in result.answer
    assert "No sé cuál es" in result.answer
    keys = _source_keys(result)
    assert ("relationship", str(first_relationship.id)) in keys
    assert ("relationship", str(second_relationship.id)) in keys


async def test_relative_follow_up_reports_honest_absence(
    db_session: AsyncSession,
) -> None:
    active, _, _, _ = await _seed_relative(
        db_session,
        relationship_type=RelationshipType.friend_of,
        other_name="Miguel",
    )
    service = _service(db_session)
    first = await _resolve_active(service, active.first_name)

    result = await service.handle_user_message(
        "¿Y su madre?", conversation_id=first.conversation_id
    )

    assert result.answer == "No tengo guardado quién es su madre."
    assert [(source.entity_type, source.usage_type) for source in result.sources] == [
        ("person", "subject")
    ]


async def test_relative_follow_up_withholds_pending_relation(
    db_session: AsyncSession,
) -> None:
    active, child, _, _ = await _seed_relative(
        db_session,
        relationship_type=RelationshipType.child_of,
        status=RelationshipStatus.pending,
        active_is_person_a=False,
    )
    service = _service(db_session)
    first = await _resolve_active(service)

    result = await service.handle_user_message(
        "¿Y su hija?", conversation_id=first.conversation_id
    )

    assert result.answer == "No tengo guardado quién es su hija."
    assert child.display_name not in result.answer
    assert [(source.entity_type, source.usage_type) for source in result.sources] == [
        ("person", "subject")
    ]


async def test_relative_follow_up_marks_probable_relation_as_unconfirmed(
    db_session: AsyncSession,
) -> None:
    await _seed_relative(
        db_session,
        relationship_type=RelationshipType.child_of,
        status=RelationshipStatus.probable,
        active_is_person_a=False,
    )
    service = _service(db_session)
    first = await _resolve_active(service)

    result = await service.handle_user_message(
        "¿Y su hija?", conversation_id=first.conversation_id
    )

    assert "María" in result.answer
    assert "descendiente de Carmen" in result.answer
    assert "no está confirmado" in result.answer
