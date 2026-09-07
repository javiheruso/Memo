"""Public server-rendered chat UI: pages render and the pipeline works end to end.

Every message goes through the same ``ConversationService`` behind the JSON
chat API (``POST /api/v1/chat``), so these tests assert the HTML surface
renders, that the form drives that service, that multi-turn continuity is kept
via the conversation cookie, and that the ambiguity/gap signals the pipeline
produces surface back to the user.
"""

import uuid

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Conversation, KnowledgeGap, Message
from tests.fixtures.universe import SeedUniverse

UI = "/chat"
COOKIE = "memo_chat_conversation"

_GENERAL_REPLY = "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."


async def _cookie_conversation_id(client: AsyncClient) -> uuid.UUID | None:
    raw = client.cookies.get(COOKIE)
    return uuid.UUID(raw) if raw else None


async def _stored_conversation(
    db_session: AsyncSession, conversation_id: uuid.UUID
) -> Conversation:
    return (
        await db_session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
    ).scalar_one()


async def _stored_messages(
    db_session: AsyncSession, conversation_id: uuid.UUID
) -> list[Message]:
    return list(
        (
            await db_session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            )
        ).scalars()
    )


# --------------------------------------------------------------------------
# Page render
# --------------------------------------------------------------------------


async def test_empty_chat_page_renders(app_client: AsyncClient) -> None:
    response = await app_client.get(UI)
    assert response.status_code == 200
    assert "Escribe tu mensaje" in response.text
    assert 'name="message"' in response.text
    assert "Enviar" in response.text
    assert "soy Memo" in response.text
    assert app_client.cookies.get(COOKIE) is None


async def test_page_shows_stored_history(app_client: AsyncClient) -> None:
    sent = await app_client.post(
        UI, data={"message": "Me encanta la tortilla de patatas"}
    )
    assert sent.status_code == 303
    conversation_id = await _cookie_conversation_id(app_client)
    assert conversation_id is not None

    response = await app_client.get(UI)
    assert response.status_code == 200
    assert "Me encanta la tortilla de patatas" in response.text
    assert _GENERAL_REPLY in response.text
    assert "Tú" in response.text
    assert "Memo" in response.text


# --------------------------------------------------------------------------
# Message submission drives the real pipeline
# --------------------------------------------------------------------------


async def test_message_submission_runs_conversation_service(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    response = await app_client.post(
        UI, data={"message": "Me encanta la tortilla de patatas"}
    )
    assert response.status_code == 303
    conversation_id = await _cookie_conversation_id(app_client)
    assert conversation_id is not None

    conversation = await _stored_conversation(db_session, conversation_id)
    assert conversation is not None
    messages = await _stored_messages(db_session, conversation_id)
    assert [message.speaker for message in messages] == ["user", "assistant"]
    assert messages[0].text == "Me encanta la tortilla de patatas"
    assert messages[1].text == _GENERAL_REPLY


async def test_blank_message_is_refused(app_client: AsyncClient) -> None:
    response = await app_client.post(UI, data={"message": "   "}, follow_redirects=True)
    assert response.status_code == 200
    assert "Escribe un mensaje antes de enviar." in response.text
    assert app_client.cookies.get(COOKIE) is None


# --------------------------------------------------------------------------
# Multi-turn continuity
# --------------------------------------------------------------------------


async def test_multi_turn_continuity(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    first = await app_client.post(UI, data={"message": "Hola"})
    assert first.status_code == 303
    conversation_id = await _cookie_conversation_id(app_client)
    assert conversation_id is not None

    second = await app_client.post(
        UI, data={"message": "Me encanta la tortilla de patatas"}
    )
    assert second.status_code == 303
    assert await _cookie_conversation_id(app_client) == conversation_id

    page = await app_client.get(UI)
    assert page.status_code == 200
    assert page.text.count("Tú") == 2
    assert page.text.index("Hola") < page.text.index(
        "Me encanta la tortilla de patatas"
    )
    assert page.text.index("Hola") < page.text.index(_GENERAL_REPLY)

    messages = await _stored_messages(db_session, conversation_id)
    assert [message.speaker for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert {message.text for message in messages} >= {
        "Hola",
        "Me encanta la tortilla de patatas",
        _GENERAL_REPLY,
    }


async def test_reset_starts_a_fresh_conversation(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    await app_client.post(UI, data={"message": "Me encanta la tortilla de patatas"})
    old_id = await _cookie_conversation_id(app_client)
    assert old_id is not None

    response = await app_client.post(f"{UI}/reset", follow_redirects=True)
    assert response.status_code == 200
    assert app_client.cookies.get(COOKIE) is None
    assert "soy Memo" in response.text

    stored = list((await db_session.execute(select(Conversation))).scalars())
    assert len(stored) == 1
    assert stored[0].id == old_id


# --------------------------------------------------------------------------
# Gap and ambiguity signals still surface
# --------------------------------------------------------------------------


async def test_unknown_person_gap_surfaces_on_page(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    response = await app_client.post(
        UI, data={"message": "¿Quién era Eustaquio?"}, follow_redirects=True
    )
    assert response.status_code == 200
    assert "No tengo guardado quién era Eustaquio." in response.text
    assert "Todavía no conozco a esa persona" in response.text

    conversation_id = await _cookie_conversation_id(app_client)
    assert conversation_id is not None
    gap = (
        await db_session.execute(
            select(KnowledgeGap)
            .where(
                KnowledgeGap.gap_type == "person_unknown",
                KnowledgeGap.canonical_reference == "eustaquio",
            )
            .options(selectinload(KnowledgeGap.sessions))
        )
    ).scalar_one()
    session_links = list(gap.sessions)
    assert [link.conversation_id for link in session_links] == [conversation_id]

    messages = await _stored_messages(db_session, conversation_id)
    assert len(messages) == 2
    assert messages[1].text == "No tengo guardado quién era Eustaquio."


async def test_ambiguous_person_surfaces_candidates_and_notice(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    response = await app_client.post(
        UI, data={"message": "¿Quién era Paco?"}, follow_redirects=True
    )
    assert response.status_code == 200
    assert "Hay varias personas llamadas Paco" in response.text
    assert "Francisco Navarro" in response.text
    assert "Paco García" in response.text
    assert "Hay varias personas con ese nombre" in response.text
    assert await _cookie_conversation_id(app_client) is not None
