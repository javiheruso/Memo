"""Public server-rendered chat surface.

A minimal, large-text conversation page so a human can talk to the companion
in text mode and exercise the whole pipeline end to end without calling the
JSON API. It reuses ``ConversationService`` unchanged (the exact class behind
``POST /api/v1/chat``), so intent routing, retrieval, trust policies, claim
ingestion and knowledge-gap persistence behave identically here.

This surface adds no auth, no STT/TTS, no streaming and no frontend toolchain:
messages render with Jinja2 and forms post back to this router (POST/redirect/
GET, like the admin UI). Multi-turn continuity is carried by a session cookie
that holds the ``conversation_id``; the conversation itself lives in the
database, so clearing the cookie never deletes history.

UI copy is Spanish because the deterministic responder and the product
audience (older Spanish-speaking adults) are Spanish; code stays in English.
"""

import logging
import os
import uuid
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    ClaimContradictionDetectorDep,
    ClaimExtractorDep,
    ClaimGrouperDep,
    EmbedderDep,
    ResponderDep,
    get_db,
)
from app.repositories.conversation import ConversationRepository
from app.security import cookie_secure
from app.services.conversation import ChatResult, ConversationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat-ui"], include_in_schema=False)

CONVERSATION_COOKIE = "memo_chat_conversation"
_COOKIE_PATH = "/"

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

DbSession = Annotated[AsyncSession, Depends(get_db)]


def _conversation_id_from_cookie(request: Request) -> uuid.UUID | None:
    raw = request.cookies.get(CONVERSATION_COOKIE)
    if not raw:
        return None
    try:
        return uuid.UUID(raw)
    except ValueError:
        return None


def _back(*, notice: str | None = None, error: str | None = None) -> RedirectResponse:
    query: dict[str, str] = {}
    if notice:
        query["notice"] = notice
    if error:
        query["error"] = error
    url = f"/chat?{urlencode(query)}" if query else "/chat"
    return RedirectResponse(url=url, status_code=303)


def _notice_for(result: ChatResult) -> str | None:
    """Map pipeline signals to a single user-facing notice code.

    Resolution is either ambiguous or not-found in a given turn, so at most one
    notice applies. The reply text itself is already rendered as the assistant
    message; this banner is the structured signal, not a duplicate.
    """
    if result.ambiguity is not None:
        return "ambiguous"
    if result.gap is not None:
        return "unknown_person"
    return None


def _set_conversation_cookie(
    response: RedirectResponse, conversation_id: uuid.UUID
) -> None:
    response.set_cookie(
        CONVERSATION_COOKIE,
        str(conversation_id),
        httponly=True,
        samesite="lax",
        secure=cookie_secure(),
        path=_COOKIE_PATH,
    )


@router.get("", response_class=HTMLResponse)
async def chat_page(request: Request, db: DbSession) -> HTMLResponse:
    conversation_id = _conversation_id_from_cookie(request)
    messages = []
    if conversation_id is not None:
        messages = await ConversationRepository(db).list_messages(conversation_id)
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "messages": messages,
            "notice": request.query_params.get("notice"),
            "error": request.query_params.get("error"),
        },
    )


@router.post("")
async def chat_send(
    request: Request,
    db: DbSession,
    responder: ResponderDep,
    embedder: EmbedderDep,
    claim_extractor: ClaimExtractorDep,
    claim_grouper: ClaimGrouperDep,
    claim_contradiction_detector: ClaimContradictionDetectorDep,
) -> RedirectResponse:
    form = await request.form()
    text = str(form.get("message") or "").strip()
    if not text:
        return _back(error="Escribe un mensaje antes de enviar.")

    conversation_id = _conversation_id_from_cookie(request)
    service = ConversationService(
        db,
        responder,
        embedder=embedder,
        claim_extractor=claim_extractor,
        claim_grouper=claim_grouper,
        claim_contradiction_detector=claim_contradiction_detector,
    )
    try:
        result = await service.handle_user_message(
            text=text, conversation_id=conversation_id
        )
    except Exception:
        await db.rollback()
        logger.exception("chat turn failed for user message %r", text[:200])
        return _back(error="Algo ha ido mal al guardar el mensaje. Inténtalo de nuevo.")

    response = _back(notice=_notice_for(result))
    _set_conversation_cookie(response, result.conversation_id)
    return response


@router.post("/reset")
async def chat_reset() -> RedirectResponse:
    """Start a fresh conversation: drop the pointer cookie, keep stored history."""
    response = RedirectResponse(url="/chat", status_code=303)
    response.delete_cookie(CONVERSATION_COOKIE, path=_COOKIE_PATH)
    return response
