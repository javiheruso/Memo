import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import Intent
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.deterministic import DeterministicResponder
from app.models import Message
from app.voice.openai_compatible import InvalidTranscription
from app.voice.service import VoiceInputError, VoiceService
from tests.fixtures.universe import SeedUniverse


class FakeSTT:
    def __init__(self, transcript: str = "¿Quién era Francisco?", error=None) -> None:
        self._transcript = transcript
        self._error = error

    async def transcribe(self, audio: bytes, *, content_type=None) -> str:
        if self._error is not None:
            raise self._error
        return self._transcript


class FakeTTS:
    def __init__(self, audio: bytes = b"\x00MP3fake", error=None) -> None:
        self._audio = audio
        self._error = error

    async def synthesize(self, text: str) -> bytes:
        if self._error is not None:
            raise self._error
        return self._audio


def _service(
    db: AsyncSession,
    *,
    stt=None,
    tts=None,
) -> VoiceService:
    return VoiceService(
        db,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
        stt=stt,
        tts=tts,
    )


async def _user_message_text(
    db: AsyncSession, conversation_id
) -> list[str]:
    messages = (
        (
            await db.execute(
                select(Message).where(Message.conversation_id == conversation_id)
            )
        )
        .scalars()
        .all()
    )
    return [message.text for message in messages if message.speaker == "user"]


async def test_text_only_turn_needs_no_voice_provider(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    result = await _service(db_session).process(text="¿Quién era Eustaquio?")
    assert result.transcript is None
    assert result.text == "¿Quién era Eustaquio?"
    assert result.answer == "No tengo guardado quién era Eustaquio."
    assert result.audio is None
    assert result.intent == Intent.person_query
    assert await _user_message_text(db_session, result.conversation_id) == [
        "¿Quién era Eustaquio?"
    ]


async def test_voice_round_trip_transcribes_and_synthesizes(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(
        db_session,
        stt=FakeSTT("¿Quién era Francisco?"),
        tts=FakeTTS(b"\x00MP3voice"),
    )
    result = await service.process(
        text=None, audio=b"\x00wav", audio_content_type="audio/wav"
    )

    assert result.transcript == "¿Quién era Francisco?"
    assert result.text is None
    assert result.audio == b"\x00MP3voice"
    assert result.intent == Intent.person_query
    assert "amigo de Antonio" in result.answer
    assert await _user_message_text(db_session, result.conversation_id) == [
        "¿Quién era Francisco?"
    ]


async def test_stt_transcript_wins_over_explicit_text(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session, stt=FakeSTT("¿Quién era Francisco?"))
    result = await service.process(
        text="¿Quién era Eustaquio?",
        audio=b"\x00wav",
        audio_content_type="audio/wav",
    )
    assert result.transcript == "¿Quién era Francisco?"
    assert result.text == "¿Quién era Eustaquio?"
    assert "amigo de Antonio" in result.answer


async def test_stt_failure_degrades_to_text(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(
        db_session,
        stt=FakeSTT(error=httpx.ConnectError("connection refused")),
    )
    result = await service.process(
        text="¿Quién era Eustaquio?",
        audio=b"\x00wav",
        audio_content_type="audio/wav",
    )
    assert result.transcript is None
    assert result.answer == "No tengo guardado quién era Eustaquio."
    assert await _user_message_text(db_session, result.conversation_id) == [
        "¿Quién era Eustaquio?"
    ]


async def test_stt_invalid_output_degrades_to_text(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(
        db_session, stt=FakeSTT(error=InvalidTranscription("empty"))
    )
    result = await service.process(
        text="¿Quién era Eustaquio?", audio=b"\x00wav"
    )
    assert result.transcript is None
    assert result.answer == "No tengo guardado quién era Eustaquio."


async def test_stt_failure_without_text_raises_input_error(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(db_session, stt=FakeSTT(error=httpx.ConnectError("refused")))
    with pytest.raises(VoiceInputError):
        await service.process(audio=b"\x00wav", audio_content_type="audio/wav")


async def test_audio_without_stt_and_without_text_raises_input_error(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    with pytest.raises(VoiceInputError):
        await _service(db_session).process(audio=b"\x00wav", audio_content_type="audio/wav")


async def test_tts_failure_keeps_text_reply_without_audio(
    db_session: AsyncSession, seed: SeedUniverse
) -> None:
    service = _service(
        db_session,
        tts=FakeTTS(error=httpx.ConnectError("connection refused")),
    )
    result = await service.process(text="¿Quién era Eustaquio?")
    assert result.audio is None
    assert result.answer == "No tengo guardado quién era Eustaquio."
