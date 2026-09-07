"""Voice orchestration over the conversation pipeline.

``VoiceService`` wraps ``ConversationService`` so a voice turn runs the exact
same intent routing, retrieval, policies, responder and persistence as the
text ``POST /chat`` endpoint, and then layers speech on top:

- STT (optional): when audio is provided and an STT provider is configured,
  the audio is transcribed first and the recognized text drives the turn. If
  STT is unconfigured or the provider fails, the turn degrades to the explicit
  ``text`` when one is present.
- TTS (optional): when a TTS provider is configured, the reply text is
  synthesized into audio bytes. If it fails, the turn still returns the text
  reply with no audio.

Fail-closed by design: a transcription is never empty or invented (empty or
malformed provider output raises and degrades instead), synthesized audio is
never fabricated (an empty audio body raises and degrades instead), and a voice
feature that is not configured simply does not run. The text-only path is the
zero-config default and is exercised by the same tests as ``POST /chat``.
"""

import logging
import uuid
from dataclasses import dataclass, field

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.base import ClaimExtractor
from app.claims.contradiction import ContradictionDetector
from app.claims.grouping import ClaimGrouper
from app.domain.enums import Intent
from app.embeddings.base import Embedder
from app.llm.base import Responder
from app.models import KnowledgeGap
from app.repositories.conversation import ResponseSourceRef
from app.retrieval.entity_resolver import Resolution
from app.services.conversation import ConversationService
from app.voice.base import SpeechToText, TextToSpeech
from app.voice.openai_compatible import InvalidSpeech, InvalidTranscription

logger = logging.getLogger(__name__)

_VOICE_PROVIDER_ERRORS = (
    httpx.HTTPError,
    InvalidTranscription,
    InvalidSpeech,
)


class VoiceInputError(ValueError):
    """Raised when a voice turn has no usable text to answer."""


@dataclass(frozen=True)
class VoiceResult:
    conversation_id: uuid.UUID
    intent: Intent
    transcript: str | None
    text: str | None
    answer: str
    sources: list[ResponseSourceRef] = field(default_factory=list)
    audio: bytes | None = None
    ambiguity: Resolution | None = None
    gap: KnowledgeGap | None = None
    gap_created: bool = False


class VoiceService:
    """Runs optional STT -> the conversation pipeline -> optional TTS."""

    def __init__(
        self,
        session: AsyncSession,
        responder: Responder,
        embedder: Embedder,
        claim_extractor: ClaimExtractor | None = None,
        claim_grouper: ClaimGrouper | None = None,
        claim_contradiction_detector: ContradictionDetector | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
    ) -> None:
        self._conversation = ConversationService(
            session,
            responder,
            embedder=embedder,
            claim_extractor=claim_extractor,
            claim_grouper=claim_grouper,
            claim_contradiction_detector=claim_contradiction_detector,
        )
        self._stt = stt
        self._tts = tts

    async def process(
        self,
        *,
        text: str | None = None,
        audio: bytes | None = None,
        audio_content_type: str | None = None,
        conversation_id: uuid.UUID | None = None,
    ) -> VoiceResult:
        text_input = text.strip() if text and text.strip() else None

        transcript = await self._transcribe(audio, audio_content_type)
        user_text = transcript or text_input
        if not user_text:
            raise VoiceInputError(
                "No usable text: no text was provided and the audio could not "
                "be transcribed."
            )

        result = await self._conversation.handle_user_message(
            user_text, conversation_id=conversation_id
        )
        audio_bytes = await self._synthesize(result.answer)

        return VoiceResult(
            conversation_id=result.conversation_id,
            intent=result.intent,
            transcript=transcript,
            text=text_input,
            answer=result.answer,
            sources=result.sources,
            audio=audio_bytes,
            ambiguity=result.ambiguity,
            gap=result.gap,
            gap_created=result.gap_created,
        )

    async def _transcribe(
        self, audio: bytes | None, content_type: str | None
    ) -> str | None:
        """Recognize audio when both audio and an STT provider are present.

        Returns ``None`` when there is nothing to transcribe or when the
        provider cannot be trusted for this turn, so the caller falls back to
        the explicit text.
        """
        if not audio or self._stt is None:
            return None
        try:
            return (await self._stt.transcribe(audio, content_type=content_type)).strip()
        except _VOICE_PROVIDER_ERRORS:
            logger.warning(
                "STT transcription failed; degrading this turn to text input",
                exc_info=True,
            )
            return None

    async def _synthesize(self, answer: str) -> bytes | None:
        """Synthesize the reply when a TTS provider is configured.

        Returns ``None`` when no provider is configured or when the provider
        cannot be trusted for this turn, so the reply stays text-only.
        """
        if self._tts is None:
            return None
        try:
            return await self._tts.synthesize(answer)
        except _VOICE_PROVIDER_ERRORS:
            logger.warning(
                "TTS synthesis failed; returning the text reply without audio",
                exc_info=True,
            )
            return None
