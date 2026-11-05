"""Regressions for AUD-007: one conversation turn persists atomically.

The user message, assistant reply, response sources and turn-derived gap are
one transaction. An intermediate failure must leave no misleading partial turn
(a stored user message without a reply, or a reply without its sources).
Claim ingestion stays outside that boundary because the domain documents it as
fail-safe and independently persisted.
"""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.base import ClaimCandidate
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.base import Responder, Response
from app.llm.deterministic import DeterministicResponder
from app.models import Claim, Conversation, Message, ResponseSource
from app.repositories.conversation import ConversationRepository
from app.repositories.knowledge_gap import KnowledgeGapRepository
from app.retrieval.context_builder import AnswerContext
from app.services.conversation import ConversationService
from tests.fixtures.universe import SeedUniverse


class RaisingResponder(Responder):
    async def respond(self, context: AnswerContext) -> Response:
        raise RuntimeError("responder exploded")


class StubResponder(Responder):
    async def respond(self, context: AnswerContext) -> Response:
        return Response(text="respuesta de prueba")


class BrokenExtractor:
    def extract(self, text: str) -> list[ClaimCandidate]:
        raise RuntimeError("extractor exploded")


async def _count(db_session: AsyncSession, model) -> int:
    return (
        await db_session.execute(select(func.count()).select_from(model))
    ).scalar_one()


async def _messages_for(
    db_session: AsyncSession, conversation_id: uuid.UUID
) -> list[Message]:
    return list(
        (
            await db_session.execute(
                select(Message).where(
                    Message.conversation_id == conversation_id
                )
            )
        ).scalars()
    )


async def test_responder_failure_rolls_back_the_whole_turn(
    db_session: AsyncSession,
) -> None:
    service = ConversationService(
        db_session, RaisingResponder(), embedder=DeterministicEmbedder()
    )

    with pytest.raises(RuntimeError, match="responder exploded"):
        await service.handle_user_message("Hola")

    assert await _count(db_session, Conversation) == 0
    assert await _count(db_session, Message) == 0


async def test_source_failure_rolls_back_user_and_assistant_messages(
    db_session: AsyncSession,
    seed: SeedUniverse,
    monkeypatch,
) -> None:
    conversation = await ConversationRepository(db_session).get_or_create(None)
    conversation_id = conversation.id

    async def failing_sources(self, message_id, sources, *, commit=True):
        raise RuntimeError("sources exploded")

    monkeypatch.setattr(
        ConversationRepository, "add_response_sources", failing_sources
    )
    service = ConversationService(
        db_session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
    )

    with pytest.raises(RuntimeError, match="sources exploded"):
        await service.handle_user_message(
            "¿Quién era Francisco?", conversation_id=conversation_id
        )

    assert await _messages_for(db_session, conversation_id) == []
    sources = (
        await db_session.execute(
            select(ResponseSource)
            .join(Message, Message.id == ResponseSource.message_id)
            .where(Message.conversation_id == conversation_id)
        )
    ).scalars()
    assert list(sources) == []


async def test_gap_failure_also_rolls_back_the_turn(
    db_session: AsyncSession,
    monkeypatch,
) -> None:
    async def failing_upsert(self, **kwargs):
        raise RuntimeError("gap exploded")

    monkeypatch.setattr(KnowledgeGapRepository, "upsert", failing_upsert)
    service = ConversationService(
        db_session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
    )

    with pytest.raises(RuntimeError, match="gap exploded"):
        await service.handle_user_message("¿Quién era Eustaquio?")

    assert await _count(db_session, Conversation) == 0
    assert await _count(db_session, Message) == 0


async def test_claim_ingestion_failure_does_not_undo_a_committed_turn(
    db_session: AsyncSession,
) -> None:
    service = ConversationService(
        db_session,
        StubResponder(),
        embedder=DeterministicEmbedder(),
        claim_extractor=BrokenExtractor(),
    )

    result = await service.handle_user_message("Vivíamos en Madrid.")

    assert result.answer == "respuesta de prueba"
    messages = await _messages_for(db_session, result.conversation_id)
    assert [message.speaker for message in messages] == ["user", "assistant"]
    assert await _count(db_session, Claim) == 0
