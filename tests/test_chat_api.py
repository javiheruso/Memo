import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models import Conversation, KnowledgeGap, Memory, Message, ResponseSource
from tests.fixtures.universe import SeedUniverse

API = "/api/v1/chat"

# The memory-query API tests assert on seed narratives that only carry
# similarity scores through the SQLite in-memory fallback (the seed universe
# stores no embeddings, by design). The PostgreSQL profile covers the pgvector
# path with dedicated smoke tests.
_SKIP_ON_POSTGRES = pytest.mark.skipif(
    settings.test_database_url is not None,
    reason="relies on the SQLite semantic fallback; covered by postgres smoke tests",
)


async def test_known_person_flow(app_client: AsyncClient, seed: SeedUniverse):
    response = await app_client.post(API, json={"message": "¿Quién era Francisco?"})
    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "person_query"
    assert "amigo de Antonio" in body["answer"]
    assert body["ambiguity"] is None
    assert body["gap"] is None

    sources = body["sources"]
    person_sources = [s for s in sources if s["entity_type"] == "person"]
    assert any(
        s["entity_id"] == str(seed.person_ids["paco_navarro"])
        and s["usage_type"] == "subject"
        for s in person_sources
    )
    assert any(s["entity_type"] == "relationship" for s in sources)


async def test_fact_query_returns_relevant_fact_with_provenance(
    app_client: AsyncClient, seed: SeedUniverse
):
    response = await app_client.post(
        API, json={"message": "¿Qué aficiones tiene Antonio?"}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "fact_query"
    assert "Fishing" in body["answer"]
    assert "Guardamar de la Safor" not in body["answer"]
    fact_sources = [
        source for source in body["sources"] if source["entity_type"] == "fact"
    ]
    assert len(fact_sources) == 1
    assert fact_sources[0]["usage_type"] == "verified_fact"
    assert any(
        source["entity_type"] == "source"
        and source["usage_type"] == "provenance"
        for source in body["sources"]
    )


async def test_ambiguous_person_is_not_collapsed(
    app_client: AsyncClient, seed: SeedUniverse
):
    response = await app_client.post(API, json={"message": "¿Quién era Paco?"})
    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "person_query"
    assert body["ambiguity"] is not None
    assert body["ambiguity"]["term"] == "Paco"
    candidate_ids = {c["id"] for c in body["ambiguity"]["candidates"]}
    assert candidate_ids == {
        str(seed.person_ids["paco_navarro"]),
        str(seed.person_ids["paco_garcia"]),
    }
    assert "Francisco Navarro" in body["answer"]
    assert "Paco García" in body["answer"]
    assert body["gap"] is None

    candidate_source_ids = {
        s["entity_id"]
        for s in body["sources"]
        if s["entity_type"] == "person" and s["usage_type"] == "candidate"
    }
    assert candidate_source_ids == candidate_ids


async def test_unknown_person_creates_knowledge_gap(
    app_client: AsyncClient,
):
    response = await app_client.post(API, json={"message": "¿Quién era Eustaquio?"})
    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "No tengo guardado quién era Eustaquio."
    assert body["sources"] == []
    assert body["ambiguity"] is None
    assert body["gap"] is not None
    assert body["gap"]["gap_type"] == "person_unknown"
    assert body["gap"]["canonical_reference"] == "eustaquio"
    assert body["gap"]["status"] == "observed"


async def test_partial_person_expresses_uncertainty(
    app_client: AsyncClient, seed: SeedUniverse
):
    response = await app_client.post(API, json={"message": "¿Quién es Paco García?"})
    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "person_query"
    assert "No tengo claro" in body["answer"]
    assert "no está confirmado" in body["answer"]
    assert body["gap"] is None


async def test_general_chat_touches_no_personal_data(
    app_client: AsyncClient, db_session: AsyncSession
):
    response = await app_client.post(
        API, json={"message": "Me encanta la tortilla de patatas"}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "general_chat"
    assert body["sources"] == []
    assert body["ambiguity"] is None
    assert body["gap"] is None

    conversation = await db_session.get(
        Conversation, uuid.UUID(body["conversation_id"])
    )
    assert conversation is not None
    messages = (
        (
            await db_session.execute(
                select(Message).where(Message.conversation_id == conversation.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(messages) == 2
    assert {message.speaker for message in messages} == {"user", "assistant"}
    sources = (
        (
            await db_session.execute(
                select(ResponseSource)
                .join(Message, Message.id == ResponseSource.message_id)
                .where(Message.conversation_id == conversation.id)
            )
        )
        .scalars()
        .all()
    )
    assert sources == []


async def test_event_query_lists_confirmed_upcoming_event(
    app_client: AsyncClient, seed: SeedUniverse
):
    response = await app_client.post(API, json={"message": "¿Viene María hoy?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "event_query"
    assert "está confirmado 'Visita de María el domingo'" in body["answer"]
    assert "se celebró" not in body["answer"]
    assert body["gap"] is None

    event_sources = [
        s for s in body["sources"] if s["entity_type"] == "event"
    ]
    assert len(event_sources) == 1
    assert event_sources[0]["entity_id"] == str(seed.event_ids["maria_sunday_visit"])
    assert event_sources[0]["usage_type"] == "confirmed"


async def test_event_query_planned_event_is_not_phrased_as_completed(
    app_client: AsyncClient, seed: SeedUniverse
):
    response = await app_client.post(
        API, json={"message": "¿Qué eventos tiene Antonio esta semana?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "event_query"
    assert "está planificado 'Cumpleaños de Antonio'" in body["answer"]
    assert "se celebró" not in body["answer"]
    assert "confirmado" not in body["answer"]


async def test_event_query_unknown_person_creates_gap(
    app_client: AsyncClient,
):
    response = await app_client.post(
        API, json={"message": "¿Viene Eustaquio hoy?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "event_query"
    assert body["answer"] == "No tengo guardado quién es Eustaquio."
    assert body["gap"] is not None
    assert body["gap"]["canonical_reference"] == "eustaquio"


async def test_event_query_without_person_asks_for_clarification(
    app_client: AsyncClient,
):
    response = await app_client.post(API, json={"message": "¿Cuándo hay eventos?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "event_query"
    assert "¿De quién quieres saber sus eventos?" in body["answer"]
    assert body["sources"] == []


async def test_persistence_records_response_sources(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    response = await app_client.post(API, json={"message": "¿Quién era Francisco?"})
    conversation_id = uuid.UUID(response.json()["conversation_id"])

    messages = (
        (
            await db_session.execute(
                select(Message).where(Message.conversation_id == conversation_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(messages) == 2
    assistant = next(m for m in messages if m.speaker == "assistant")
    assert assistant.intent == "person_query"

    sources = (
        (
            await db_session.execute(
                select(ResponseSource).where(ResponseSource.message_id == assistant.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(sources) >= 2
    assert {source.source_entity_type for source in sources} >= {
        "person",
        "relationship",
    }


async def test_gap_deduplicates_across_calls(
    app_client: AsyncClient, db_session: AsyncSession
):
    await app_client.post(API, json={"message": "¿Quién era Eustaquio?"})
    await app_client.post(API, json={"message": "¿Quién era Eustaquio?"})

    gap = (
        await db_session.execute(
            select(KnowledgeGap).where(
                KnowledgeGap.gap_type == "person_unknown",
                KnowledgeGap.canonical_reference == "eustaquio",
            )
        )
    ).scalar_one()
    assert gap.occurrence_count == 2
    assert gap.distinct_sessions == 2


@_SKIP_ON_POSTGRES
async def test_memory_query_returns_stored_memory_as_memory(
    app_client: AsyncClient, seed: SeedUniverse
):
    response = await app_client.post(
        API, json={"message": "Do you remember when we went dancing?"}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "memory_query"
    assert "dancing" in body["answer"].lower()
    assert body["ambiguity"] is None
    assert body["gap"] is None
    assert body["sources"]
    assert all(
        source["entity_type"] == "memory" and source["usage_type"] == "memory"
        for source in body["sources"]
    )


async def test_memory_query_without_match_is_honest(
    app_client: AsyncClient, seed: SeedUniverse
):
    response = await app_client.post(
        API, json={"message": "¿Te acuerdas de cuando íbamos a bailar?"}
    )
    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "memory_query"
    assert body["answer"] == "No tengo guardado ese recuerdo todavía."
    assert body["sources"] == []
    assert body["gap"] is None


@_SKIP_ON_POSTGRES
async def test_memory_query_never_surfaces_sensitive_memory(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    response = await app_client.post(
        API, json={"message": "Do you remember Antonio passing away?"}
    )
    assert response.status_code == 200
    body = response.json()

    passing = (
        await db_session.execute(
            select(Memory).where(Memory.title == "Antonio's passing")
        )
    ).scalar_one()
    assert "passing" not in body["answer"].lower()
    assert all(
        source["entity_id"] != str(passing.id) for source in body["sources"]
    )
