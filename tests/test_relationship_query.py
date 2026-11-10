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
