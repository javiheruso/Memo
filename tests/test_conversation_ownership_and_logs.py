"""Regressions for AUD-003 (conversation ownership) and AUD-004 (log privacy).

A conversation is bound to an opaque, signed browser session. Knowing another
client's ``conversation_id`` must not be enough to read or continue it. Error
paths must keep technical identifiers in logs without echoing user text.
"""

import logging
import uuid

from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.models import Message
from app.security import issue_chat_session_cookie, read_chat_session_id
from app.services.conversation import ConversationService

API = "/api/v1/chat"
UI = "/chat"
CONVERSATION_COOKIE = "memo_chat_conversation"
SESSION_COOKIE = "memo_chat_session"


def _second_client() -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    )


async def _message_count(
    db_session: AsyncSession, conversation_id: uuid.UUID
) -> int:
    return (
        await db_session.execute(
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
        )
    ).scalar_one()


def test_chat_session_cookie_rejects_tampering() -> None:
    session_id = uuid.uuid4()
    value = issue_chat_session_cookie(session_id)
    raw, _, signature = value.partition(".")

    assert read_chat_session_id(value) == session_id
    assert read_chat_session_id(str(session_id)) is None
    tampered = f"{raw}.{'0' if signature[0] != '0' else '1'}{signature[1:]}"
    assert read_chat_session_id(tampered) is None


async def test_api_rejects_a_conversation_owned_by_another_client(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    created = await app_client.post(
        API, json={"message": "Me encanta la tortilla de patatas"}
    )
    assert created.status_code == 200
    conversation_id = uuid.UUID(created.json()["conversation_id"])
    assert app_client.cookies.get(SESSION_COOKIE) is not None

    async with _second_client() as other:
        stolen = await other.post(
            API,
            json={
                "message": "¿Qué me habías contado?",
                "conversation_id": str(conversation_id),
            },
        )

    assert stolen.status_code == 404
    assert stolen.json()["detail"] == "Conversation not found"
    assert await _message_count(db_session, conversation_id) == 2

    follow_up = await app_client.post(
        API,
        json={
            "message": "Hola de nuevo",
            "conversation_id": str(conversation_id),
        },
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["conversation_id"] == str(conversation_id)
    assert await _message_count(db_session, conversation_id) == 4


async def test_web_chat_ignores_a_foreign_conversation_cookie(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    victim_marker = "Marcador privado de la conversación de Ana"
    sent = await app_client.post(UI, data={"message": victim_marker})
    assert sent.status_code == 303
    conversation_id = uuid.UUID(app_client.cookies.get(CONVERSATION_COOKIE))

    async with _second_client() as intruder:
        intruder.cookies.set(
            CONVERSATION_COOKIE, str(conversation_id), domain="test", path="/"
        )
        page = await intruder.get(UI)
        assert page.status_code == 200
        assert victim_marker not in page.text

        continued = await intruder.post(
            UI,
            data={"message": "Quiero continuar esta conversación"},
        )

    assert continued.status_code == 303
    new_conversation_id = uuid.UUID(
        continued.cookies.get(CONVERSATION_COOKIE)
    )
    assert new_conversation_id != conversation_id
    assert await _message_count(db_session, conversation_id) == 2

    victim_page = await app_client.get(UI)
    assert victim_marker in victim_page.text
    assert "Quiero continuar esta conversación" not in victim_page.text


async def test_chat_error_log_omits_user_content(
    app_client: AsyncClient,
    monkeypatch,
    caplog,
) -> None:
    marker = "MI-MARCADOR-PRIVADO-12345"

    async def failing_turn(
        self,
        text: str,
        conversation_id=None,
        *,
        owner_session_id=None,
    ):
        raise RuntimeError(f"No se pudo guardar el texto del usuario: {text}")

    monkeypatch.setattr(ConversationService, "handle_user_message", failing_turn)
    caplog.set_level(logging.ERROR, logger="app.web.chat")

    response = await app_client.post(
        UI, data={"message": marker}, follow_redirects=True
    )

    assert response.status_code == 200
    assert "Algo ha ido mal al guardar el mensaje." in response.text
    assert "chat turn failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert marker not in caplog.text
