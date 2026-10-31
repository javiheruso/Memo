from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import Intent, MemoryStatus, SensitivityLevel, SourceType
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.base import Responder, Response
from app.llm.deterministic import DeterministicResponder
from app.policies.sensitive_topics import SPONTANEOUS_SUPPRESSION_REPLY
from app.repositories.memory import MemoryRepository
from app.repositories.person import PersonRepository
from app.repositories.profile import ConversationProfileRepository
from app.repositories.source import SourceRepository
from app.retrieval.context_builder import AnswerContext
from app.schemas.memory import MemoryCreate, MemoryPersonCreate
from app.schemas.person import PersonCreate
from app.schemas.profile import ConversationProfileCreate, SensitiveTopicCreate
from app.schemas.source import SourceCreate
from app.services.conversation import ConversationService


async def _seed_family_memories(session: AsyncSession):
    persons = PersonRepository(session)
    elena = await persons.create(PersonCreate(first_name="Elena"))
    await persons.create(PersonCreate(first_name="Miguel", is_deceased=True))
    source = await SourceRepository(session).create(
        SourceCreate(
            source_type=SourceType.user,
            person_id=elena.id,
            description="Test recollections",
        )
    )
    memories = MemoryRepository(session)
    await memories.create(
        MemoryCreate(
            title="Una despedida",
            narrative="Elena recuerda el fallecimiento de Miguel cada primavera.",
            memory_status=MemoryStatus.consolidated,
            sensitivity_level=SensitivityLevel.low,
            source_id=source.id,
            people=[MemoryPersonCreate(person_id=elena.id, role="protagonista")],
        )
    )
    await memories.create(
        MemoryCreate(
            title="El día exacto",
            narrative="Miguel falleció el 3 de marzo de 1998.",
            memory_status=MemoryStatus.consolidated,
            sensitivity_level=SensitivityLevel.high,
            source_id=source.id,
            people=[MemoryPersonCreate(person_id=elena.id, role="protagonista")],
        )
    )
    await ConversationProfileRepository(session).create(
        ConversationProfileCreate(
            person_id=elena.id,
            preferred_name="Elena",
            positive_topics=["costura", "familia"],
            sensitive_topics=[
                SensitiveTopicCreate(
                    topic="fallecimiento de Miguel",
                    handling="Do not initiate; redirect gently if asked.",
                    triggers=["murió"],
                )
            ],
            is_active=True,
        )
    )
    return elena


async def test_service_does_not_initiate_sensitive_topic_on_person_query(
    db_session: AsyncSession,
) -> None:
    await _seed_family_memories(db_session)
    service = ConversationService(
        db_session, DeterministicResponder(), embedder=DeterministicEmbedder()
    )

    result = await service.handle_user_message("¿Quién era Elena?")

    assert result.intent == Intent.person_query
    assert "fallecimiento" not in result.answer.lower()
    assert "detalles" in result.answer


async def test_service_redirects_when_person_explicitly_raises_topic(
    db_session: AsyncSession,
) -> None:
    await _seed_family_memories(db_session)
    service = ConversationService(
        db_session, DeterministicResponder(), embedder=DeterministicEmbedder()
    )

    result = await service.handle_user_message("Miguel murió, ¿verdad?")

    assert "No quiero inventar detalles" in result.answer
    assert "costura" in result.answer
    assert result.sources == []


async def test_explicit_topic_does_not_bypass_high_sensitivity_filter(
    db_session: AsyncSession,
) -> None:
    await _seed_family_memories(db_session)

    class CapturingResponder(Responder):
        def __init__(self) -> None:
            self.context: AnswerContext | None = None

        async def respond(self, context: AnswerContext) -> Response:
            self.context = context
            return Response(text="respuesta segura")

    responder = CapturingResponder()
    service = ConversationService(
        db_session, responder, embedder=DeterministicEmbedder()
    )

    result = await service.handle_user_message(
        "¿Te acuerdas del fallecimiento de Miguel?"
    )

    assert result.answer == "respuesta segura"
    assert responder.context is not None
    assert [item.title for item in responder.context.memories] == ["Una despedida"]
    assert "3 de marzo" not in " ".join(
        item.narrative for item in responder.context.memories
    )


async def test_service_suppresses_responder_that_initiates_blocked_topic(
    db_session: AsyncSession,
) -> None:
    await _seed_family_memories(db_session)

    class LeakyResponder(Responder):
        async def respond(self, context: AnswerContext) -> Response:
            return Response(text="El fallecimiento de Miguel fue en 1998.")

    service = ConversationService(
        db_session, LeakyResponder(), embedder=DeterministicEmbedder()
    )

    result = await service.handle_user_message("Me apetece charlar un rato")

    assert result.answer == SPONTANEOUS_SUPPRESSION_REPLY
    assert "fallecimiento" not in result.answer.lower()
    assert result.sources == []
