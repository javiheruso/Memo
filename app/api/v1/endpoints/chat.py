from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    ClaimContradictionDetectorDep,
    ClaimExtractorDep,
    ClaimGrouperDep,
    EmbedderDep,
    ResponderDep,
    get_db,
)
from app.schemas.chat import (
    AmbiguityInfo,
    CandidateRef,
    ChatRequest,
    ChatResponse,
    GapRef,
    SourceRef,
)
from app.services.conversation import ConversationService

router = APIRouter(tags=["chat"])

DbSession = Annotated[AsyncSession, Depends(get_db)]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    db: DbSession,
    responder: ResponderDep,
    embedder: EmbedderDep,
    claim_extractor: ClaimExtractorDep,
    claim_grouper: ClaimGrouperDep,
    claim_contradiction_detector: ClaimContradictionDetectorDep,
    payload: ChatRequest,
) -> ChatResponse:
    result = await ConversationService(
        db,
        responder,
        embedder=embedder,
        claim_extractor=claim_extractor,
        claim_grouper=claim_grouper,
        claim_contradiction_detector=claim_contradiction_detector,
    ).handle_user_message(
        text=payload.message, conversation_id=payload.conversation_id
    )

    ambiguity = None
    if result.ambiguity is not None:
        ambiguity = AmbiguityInfo(
            term=result.ambiguity.term,
            candidates=[
                CandidateRef(id=candidate.id, display_name=candidate.display_name)
                for candidate in result.ambiguity.candidates
            ],
        )

    gap = None
    if result.gap is not None:
        gap = GapRef(
            id=result.gap.id,
            gap_type=result.gap.gap_type,
            canonical_reference=result.gap.canonical_reference,
            status=result.gap.status.value,
        )

    return ChatResponse(
        conversation_id=result.conversation_id,
        intent=result.intent,
        answer=result.answer,
        sources=[SourceRef(**source.__dict__) for source in result.sources],
        ambiguity=ambiguity,
        gap=gap,
    )
