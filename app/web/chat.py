"""Public voice-first chat surface.

A minimal, large-text conversation page so a human can talk to the companion
with natural speech. The primary interaction is a push-to-talk microphone that
sends one recording to ``POST /api/v1/chat/voice`` through a small vanilla
JavaScript asset; a text form remains as an explicit secondary fallback. The
page reuses ``ConversationService`` unchanged (the exact class behind
``POST /api/v1/chat``), so intent routing, retrieval, trust policies, claim
ingestion and knowledge-gap persistence behave identically here.

The server surface adds no auth, no streaming, no STT/TTS implementation and no
frontend toolchain: history renders with Jinja2 and the fallback form posts back
to this router (POST/redirect/GET, like the admin UI). Multi-turn continuity is
carried by a signed owner cookie plus a ``conversation_id`` pointer cookie; the
conversation itself lives in the database, so clearing the cookie never deletes
history. The voice endpoint sets both cookies too, so spoken and typed turns
share one thread.

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
    InterpreterDep,
    ResponderDep,
    get_db,
)
from app.repositories.conversation import (
    ConversationAccessDenied,
    ConversationRepository,
)
from app.security import (
    CHAT_CONVERSATION_COOKIE_NAME,
    CHAT_SESSION_COOKIE_NAME,
    read_chat_conversation_id,
    read_chat_session_id,
    resolve_chat_session_id,
    set_chat_conversation_cookie,
    set_chat_session_cookie,
)
from app.services.conversation import ChatResult, ConversationService
from app.web.assets import asset_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat-ui"], include_in_schema=False)

CONVERSATION_COOKIE = CHAT_CONVERSATION_COOKIE_NAME
_COOKIE_PATH = "/"

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)
templates.env.globals["asset_url"] = asset_url

DbSession = Annotated[AsyncSession, Depends(get_db)]


def _conversation_id_from_cookie(request: Request) -> uuid.UUID | None:
    return read_chat_conversation_id(request.cookies.get(CONVERSATION_COOKIE))


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


@router.get("", response_class=HTMLResponse)
async def chat_page(request: Request, db: DbSession) -> HTMLResponse:
    conversation_id = _conversation_id_from_cookie(request)
    owner_session_id = read_chat_session_id(
        request.cookies.get(CHAT_SESSION_COOKIE_NAME)
    )
    messages = []
    active_conversation_id = ""
    if conversation_id is not None and owner_session_id is not None:
        repository = ConversationRepository(db)
        conversation = await repository.get_owned(
            conversation_id, str(owner_session_id)
        )
        if conversation is not None:
            messages = await repository.list_messages(conversation.id)
            active_conversation_id = str(conversation.id)
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "messages": messages,
            "conversation_id": active_conversation_id,
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
    interpreter: InterpreterDep,
    claim_extractor: ClaimExtractorDep,
    claim_grouper: ClaimGrouperDep,
    claim_contradiction_detector: ClaimContradictionDetectorDep,
) -> RedirectResponse:
    form = await request.form()
    text = str(form.get("message") or "").strip()
    if not text:
        return _back(error="Escribe un mensaje antes de enviar.")

    owner_session_id = resolve_chat_session_id(request)
    conversation_id = _conversation_id_from_cookie(request)
    service = ConversationService(
        db,
        responder,
        embedder=embedder,
        claim_extractor=claim_extractor,
        claim_grouper=claim_grouper,
        claim_contradiction_detector=claim_contradiction_detector,
        interpreter=interpreter,
    )
    try:
        result = await service.handle_user_message(
            text=text,
            conversation_id=conversation_id,
            owner_session_id=str(owner_session_id),
        )
    except ConversationAccessDenied:
        # A forged or foreign conversation cookie must not grant access. Keep
        # the turn usable by starting a fresh conversation owned by this
        # session instead of exposing the other conversation's history.
        logger.warning(
            "chat conversation access denied; starting a new conversation "
            "(conversation_id=%s)",
            conversation_id,
        )
        try:
            result = await service.handle_user_message(
                text=text,
                conversation_id=None,
                owner_session_id=str(owner_session_id),
            )
        except Exception as exc:  # noqa: BLE001 - safe UI error path
            await db.rollback()
            logger.error(
                "chat turn failed (%s, conversation_id=%s)",
                type(exc).__name__,
                conversation_id,
            )
            return _back(
                error="Algo ha ido mal al guardar el mensaje. Inténtalo de nuevo."
            )
    except Exception as exc:  # noqa: BLE001 - safe UI error path
        await db.rollback()
        logger.error(
            "chat turn failed (%s, conversation_id=%s)",
            type(exc).__name__,
            conversation_id,
        )
        return _back(error="Algo ha ido mal al guardar el mensaje. Inténtalo de nuevo.")

    response = _back(notice=_notice_for(result))
    set_chat_session_cookie(response, owner_session_id)
    set_chat_conversation_cookie(response, result.conversation_id)
    return response


@router.post("/reset")
async def chat_reset() -> RedirectResponse:
    """Start a fresh conversation: drop the pointer cookie, keep stored history."""
    response = RedirectResponse(url="/chat", status_code=303)
    response.delete_cookie(CONVERSATION_COOKIE, path=_COOKIE_PATH)
    return response
