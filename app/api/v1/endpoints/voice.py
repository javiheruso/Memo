import base64
import uuid
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    ClaimContradictionDetectorDep,
    ClaimExtractorDep,
    ClaimGrouperDep,
    EmbedderDep,
    ResponderDep,
    SttDep,
    TtsDep,
    get_db,
)
from app.core.config import settings
from app.repositories.conversation import ConversationAccessDenied
from app.schemas.chat import (
    AmbiguityInfo,
    CandidateRef,
    GapRef,
    SourceRef,
)
from app.schemas.voice import VoiceResponse
from app.security import (
    resolve_chat_session_id,
    set_chat_conversation_cookie,
    set_chat_session_cookie,
)
from app.voice.audio import (
    is_allowed_audio_content_type,
    normalize_audio_content_type,
)
from app.voice.service import VoiceInputError, VoiceService

router = APIRouter(tags=["voice"])

DbSession = Annotated[AsyncSession, Depends(get_db)]

# OpenAI-compatible /audio/speech endpoints return MP3 by default; the voice
# endpoint declares that codec for any synthesized audio it hands back.
_AUDIO_CONTENT_TYPE = "audio/mpeg"


@router.post("/chat/voice", response_model=VoiceResponse)
async def chat_voice(
    request: Request,
    response: Response,
    db: DbSession,
    responder: ResponderDep,
    embedder: EmbedderDep,
    claim_extractor: ClaimExtractorDep,
    claim_grouper: ClaimGrouperDep,
    claim_contradiction_detector: ClaimContradictionDetectorDep,
    stt: SttDep,
    tts: TtsDep,
    audio: Annotated[UploadFile | None, File()] = None,
    text: Annotated[str | None, Form()] = None,
    conversation_id: Annotated[uuid.UUID | None, Form()] = None,
) -> VoiceResponse:
    audio_bytes = None
    audio_content_type = None
    if audio is not None:
        audio_content_type = normalize_audio_content_type(audio.content_type)
        if not is_allowed_audio_content_type(audio_content_type):
            raise HTTPException(status_code=415, detail="Unsupported audio format")
        # Read one byte past the limit so an oversized upload is detected
        # without buffering an unbounded request body.
        audio_bytes = await audio.read(settings.voice_max_audio_bytes + 1)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload")
        if len(audio_bytes) > settings.voice_max_audio_bytes:
            raise HTTPException(status_code=413, detail="Audio upload too large")

    owner_session_id = resolve_chat_session_id(request)
    service = VoiceService(
        db,
        responder,
        embedder=embedder,
        claim_extractor=claim_extractor,
        claim_grouper=claim_grouper,
        claim_contradiction_detector=claim_contradiction_detector,
        stt=stt,
        tts=tts,
    )
    try:
        result = await service.process(
            text=text,
            audio=audio_bytes,
            audio_content_type=audio_content_type,
            conversation_id=conversation_id,
            owner_session_id=str(owner_session_id),
        )
    except ConversationAccessDenied as exc:
        raise HTTPException(status_code=404, detail="Conversation not found") from exc
    except VoiceInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    set_chat_session_cookie(response, owner_session_id)
    set_chat_conversation_cookie(response, result.conversation_id)

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

    audio_encoded = None
    audio_content_type_value = None
    if result.audio:
        audio_encoded = base64.b64encode(result.audio).decode("ascii")
        audio_content_type_value = _AUDIO_CONTENT_TYPE

    return VoiceResponse(
        conversation_id=result.conversation_id,
        intent=result.intent,
        transcript=result.transcript,
        text=result.text,
        answer=result.answer,
        sources=[SourceRef(**source.__dict__) for source in result.sources],
        audio=audio_encoded,
        audio_content_type=audio_content_type_value,
        ambiguity=ambiguity,
        gap=gap,
    )
