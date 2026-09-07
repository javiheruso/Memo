import uuid

from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.base import ClaimCandidate
from app.domain.enums import ClaimStatus
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.base import Responder, Response
from app.llm.deterministic import DeterministicResponder
from app.models import Claim
from app.repositories.claim import ClaimRepository
from app.repositories.conversation import ConversationRepository
from app.retrieval.context_builder import AnswerContext
from app.services.conversation import ConversationService
from tests.fixtures.universe import SeedUniverse

API = "/api/v1/chat"

_GENERAL_REPLY = "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."


class StubResponder(Responder):
    """Fake responder used to prove ingestion never changes the answer."""

    def __init__(self, text: str = "stub answer") -> None:
        self._text = text

    async def respond(self, context: AnswerContext) -> Response:
        return Response(text=self._text)


class BrokenExtractor:
    """Extractor that always fails, used to prove fail-safe behavior."""

    def extract(self, text: str) -> list[ClaimCandidate]:
        raise RuntimeError("extractor exploded")


async def _claim_count(db_session: AsyncSession) -> int:
    result = await db_session.execute(select(func.count()).select_from(Claim))
    return result.scalar_one()


async def _claims_for_message(db_session: AsyncSession, message_id: uuid.UUID) -> list[Claim]:
    return await ClaimRepository(db_session).list_for_message(message_id)


async def test_add_many_persists_traceable_claim(db_session: AsyncSession):
    conversations = ConversationRepository(db_session)
    conversation = await conversations.get_or_create(None)
    message = await conversations.add_message(
        conversation.id, speaker="user", text="Vivíamos en Madrid."
    )

    claims = await ClaimRepository(db_session).add_many(
        [ClaimCandidate(original_text="Vivíamos en Madrid.")],
        conversation_id=conversation.id,
        message_id=message.id,
    )

    assert len(claims) == 1
    stored = claims[0]
    assert stored.original_text == "Vivíamos en Madrid."
    assert stored.conversation_id == conversation.id
    assert stored.message_id == message.id
    assert stored.status == ClaimStatus.candidate
    assert stored.first_seen_at is not None
    assert stored.speaker_person_id is None
    assert stored.canonical_subject is None
    assert stored.canonical_predicate is None
    assert stored.canonical_object is None


async def test_list_for_message_returns_persisted_claims(db_session: AsyncSession):
    conversations = ConversationRepository(db_session)
    conversation = await conversations.get_or_create(None)
    message = await conversations.add_message(
        conversation.id, speaker="user", text="Vivíamos en Madrid. Teníamos un perro."
    )
    repo = ClaimRepository(db_session)
    await repo.add_many(
        [
            ClaimCandidate(original_text="Vivíamos en Madrid."),
            ClaimCandidate(original_text="Teníamos un perro."),
        ],
        conversation_id=conversation.id,
        message_id=message.id,
    )

    stored = await _claims_for_message(db_session, message.id)
    assert {claim.original_text for claim in stored} == {
        "Vivíamos en Madrid.",
        "Teníamos un perro.",
    }
    assert all(claim.message_id == message.id for claim in stored)


async def test_list_for_conversation_returns_claims_across_messages(
    db_session: AsyncSession,
):
    conversations = ConversationRepository(db_session)
    conversation = await conversations.get_or_create(None)
    first = await conversations.add_message(
        conversation.id, speaker="user", text="Vivíamos en Madrid."
    )
    second = await conversations.add_message(
        conversation.id, speaker="user", text="Recuerdo que teníamos un perro."
    )
    repo = ClaimRepository(db_session)
    await repo.add_many(
        [ClaimCandidate(original_text="Vivíamos en Madrid.")],
        conversation_id=conversation.id,
        message_id=first.id,
    )
    await repo.add_many(
        [ClaimCandidate(original_text="Recuerdo que teníamos un perro.")],
        conversation_id=conversation.id,
        message_id=second.id,
    )

    stored = await repo.list_for_conversation(conversation.id)
    assert {claim.original_text for claim in stored} == {
        "Vivíamos en Madrid.",
        "Recuerdo que teníamos un perro.",
    }
    assert all(claim.conversation_id == conversation.id for claim in stored)


async def test_general_chat_persists_extracted_claim(
    app_client: AsyncClient, db_session: AsyncSession
):
    response = await app_client.post(
        API, json={"message": "Vivíamos en Madrid en 1972."}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "general_chat"
    assert body["answer"] == _GENERAL_REPLY
    assert body["sources"] == []

    conversation_id = uuid.UUID(body["conversation_id"])
    claims = (
        (
            await db_session.execute(
                select(Claim).where(Claim.conversation_id == conversation_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(claims) == 1
    assert claims[0].original_text == "Vivíamos en Madrid en 1972"
    assert claims[0].status == ClaimStatus.candidate


async def test_general_chat_without_signal_persists_nothing(
    app_client: AsyncClient, db_session: AsyncSession
):
    response = await app_client.post(
        API, json={"message": "Me encanta la tortilla de patatas"}
    )
    assert response.status_code == 200
    assert response.json()["intent"] == "general_chat"

    conversation_id = uuid.UUID(response.json()["conversation_id"])
    claims = (
        (
            await db_session.execute(
                select(Claim).where(Claim.conversation_id == conversation_id)
            )
        )
        .scalars()
        .all()
    )
    assert claims == []
    assert await _claim_count(db_session) == 0


async def test_query_intent_skips_claim_ingestion(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    response = await app_client.post(API, json={"message": "¿Quién era Francisco?"})
    assert response.status_code == 200
    assert response.json()["intent"] == "person_query"
    assert await _claim_count(db_session) == 0


async def test_extraction_failure_does_not_affect_response(
    db_session: AsyncSession, seed: SeedUniverse
):
    service = ConversationService(
        db_session,
        StubResponder("respuesta de prueba"),
        embedder=DeterministicEmbedder(),
        claim_extractor=BrokenExtractor(),
    )
    result = await service.handle_user_message("Vivíamos en Madrid.")
    assert result.answer == "respuesta de prueba"
    assert result.intent.value == "general_chat"
    assert await _claim_count(db_session) == 0

async def test_ingestion_never_promotes_to_fact_or_memory(
    db_session: AsyncSession,
):
    from app.models import Fact, Memory

    service = ConversationService(
        db_session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
    )
    result = await service.handle_user_message("Vivíamos en Madrid en 1972.")
    assert result.answer == _GENERAL_REPLY

    fact_count = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()
    memory_count = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()
    assert fact_count == 0
    assert memory_count == 0
