"""Conversational retrieval of imported facts and preferences.

The slice under test recognizes a scoped person/fact question, retrieves only
the facts that belong to that question family, and answers with explicit
status and provenance. Sensitive-topic policy still runs first: blocked facts
are withheld and an explicit sensitive question keeps the careful redirect.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import Intent
from app.embeddings.deterministic import DeterministicEmbedder
from app.importers.onboarding_demo import OnboardingDemoImporter
from app.llm.base import Responder, Response
from app.llm.deterministic import DeterministicResponder
from app.models import Fact
from app.policies.sensitive_topics import SPONTANEOUS_SUPPRESSION_REPLY
from app.retrieval.context_builder import AnswerContext
from app.retrieval.fact_topics import FactTopic, fact_matches_topic
from app.services.conversation import ConversationService

PREFERENCE_DOCUMENT = {
    "schema_version": "1.0",
    "fictional": True,
    "not_real_person": True,
    "profile": {
        "id": "perfil_elena",
        "person_id": "elena",
        "preferred_name": "Elena",
        "positive_topics": ["costura"],
        "sensitive_topics": [
            {
                "topic": "fallecimiento de Miguel",
                "handling": "Do not initiate; redirect gently if asked.",
                "triggers": ["murió"],
            }
        ],
        "data_origin": "Fictional for tests",
    },
    "persons": [
        {
            "id": "elena",
            "first_name": "Elena",
            "last_name": "Martínez",
            "display_name": "Elena Martínez",
        },
        {
            "id": "miguel",
            "first_name": "Miguel",
            "last_name": "Soler",
            "display_name": "Miguel Soler",
            "is_deceased": True,
        },
    ],
    "sources": [
        {
            "id": "src_elena",
            "source_type": "user",
            "person_id": "elena",
            "description": "Elena's own recollections.",
        },
        {
            "id": "src_carmen",
            "source_type": "family_member",
            "person_id": None,
            "description": "Family testimony.",
        },
    ],
    "relationships": [],
    "facts": [
        {
            "id": "fact_hobby",
            "subject_person_id": "elena",
            "fact_type": "aficion",
            "value_text": "Coser",
            "status": "verified",
            "source_id": "src_elena",
        },
        {
            "id": "fact_food",
            "subject_person_id": "elena",
            "fact_type": "comida_favorita",
            "value_text": "Arroz al horno",
            "status": "probable",
            "source_id": "src_carmen",
        },
        {
            "id": "fact_residence",
            "subject_person_id": "elena",
            "fact_type": "residencia",
            "value_text": "Santa Pola",
            "status": "verified",
            "source_id": "src_elena",
        },
        {
            "id": "fact_parentesco",
            "subject_person_id": "elena",
            "fact_type": "parentesco",
            "value_text": "usuaria",
            "status": "verified",
            "source_id": "src_carmen",
        },
        {
            "id": "fact_dislike",
            "subject_person_id": "elena",
            "fact_type": "desagrado",
            "value_text": "Que la corrijan de forma brusca",
            "status": "probable",
            "source_id": "src_carmen",
        },
        {
            "id": "fact_death",
            "subject_person_id": "elena",
            "fact_type": "fallecimiento",
            "value_text": "Miguel murió en 1998",
            "status": "probable",
            "source_id": "src_carmen",
        },
    ],
    "memories": [],
}


async def _import_document(session: AsyncSession) -> None:
    await OnboardingDemoImporter(session).run(PREFERENCE_DOCUMENT)


async def _fact(session: AsyncSession, value_text: str) -> Fact:
    result = await session.execute(select(Fact).where(Fact.value_text == value_text))
    return result.scalar_one()


def _service(
    session: AsyncSession, responder: Responder | None = None
) -> ConversationService:
    return ConversationService(
        session,
        responder or DeterministicResponder(),
        embedder=DeterministicEmbedder(),
    )


def test_topic_relevance_uses_fact_type_not_subject() -> None:
    assert fact_matches_topic("comida_favorita", topic=FactTopic.food)
    assert fact_matches_topic("aficion", topic=FactTopic.preference)
    assert not fact_matches_topic("parentesco", topic=FactTopic.preference)
    assert not fact_matches_topic("desagrado", topic=FactTopic.preference)
    assert not fact_matches_topic(None, topic=FactTopic.preference)
    assert fact_matches_topic(None, topic=FactTopic.general)


async def test_preference_query_retrieves_only_relevant_facts_with_provenance(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)

    result = await _service(db_session).handle_user_message("¿Qué le gusta a Elena?")

    assert result.intent == Intent.fact_query
    answer = result.answer
    assert "Coser" in answer
    assert "Arroz al horno" in answer
    assert "usuaria" not in answer
    assert "Que la corrijan" not in answer
    assert "Miguel murió" not in answer
    assert "probable, sin confirmar" in answer
    assert "tengo como confirmado" not in answer
    assert "según lo que me contó la propia persona" in answer
    assert "según un familiar" in answer

    hobby = await _fact(db_session, "Coser")
    food = await _fact(db_session, "Arroz al horno")
    parentesco = await _fact(db_session, "usuaria")
    fact_ids = {
        source.entity_id for source in result.sources if source.entity_type == "fact"
    }
    assert fact_ids == {hobby.id, food.id}
    assert parentesco.id not in fact_ids
    assert any(
        source.entity_type == "source" and source.usage_type == "provenance"
        for source in result.sources
    )
    assert all(
        source.usage_type == "probable_fact"
        for source in result.sources
        if source.entity_type == "fact"
    )


async def test_specific_question_narrows_to_one_fact_family(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)

    result = await _service(db_session).handle_user_message(
        "¿Cuál es la comida favorita de Elena?"
    )

    assert result.intent == Intent.fact_query
    assert "Arroz al horno" in result.answer
    assert "Coser" not in result.answer


async def test_fact_query_without_relevant_facts_is_honest(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)

    result = await _service(db_session).handle_user_message(
        "¿Qué música le gusta a Elena?"
    )

    assert result.intent == Intent.fact_query
    assert result.answer == "No tengo guardado ningún dato sobre eso de Elena Martínez."
    assert all(source.entity_type != "fact" for source in result.sources)


async def test_verified_fact_is_presented_as_confirmed_with_source(
    db_session: AsyncSession, seed
) -> None:
    result = await _service(db_session).handle_user_message(
        "¿Qué aficiones tiene Antonio?"
    )

    assert result.intent == Intent.fact_query
    assert "Fishing" in result.answer
    assert "tengo como confirmado" in result.answer
    assert "según lo que me contó la propia persona" in result.answer
    # A fact query never drifts into the person hint ladder.
    assert "Guardamar de la Safor" not in result.answer
    assert "pareja de" not in result.answer
    assert any(
        source.entity_type == "fact" and source.usage_type == "verified_fact"
        for source in result.sources
    )
    assert any(
        source.entity_type == "source" and source.usage_type == "provenance"
        for source in result.sources
    )


async def test_ambiguous_subject_is_not_collapsed(
    db_session: AsyncSession, seed
) -> None:
    result = await _service(db_session).handle_user_message(
        "¿Qué aficiones tiene Paco?"
    )

    assert result.intent == Intent.fact_query
    assert result.ambiguity is not None
    assert "¿A cuál te refieres?" in result.answer
    assert all(source.entity_type != "fact" for source in result.sources)


async def test_blocked_sensitive_topic_withholds_matching_fact(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)

    result = await _service(db_session).handle_user_message(
        "¿Qué datos tienes de Elena?"
    )

    assert result.intent == Intent.fact_query
    assert "Coser" in result.answer
    assert "1998" not in result.answer
    assert "murió" not in result.answer
    death = await _fact(db_session, "Miguel murió en 1998")
    assert death.id not in {
        source.entity_id for source in result.sources if source.entity_type == "fact"
    }


async def test_explicit_sensitive_topic_keeps_careful_redirect(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)

    result = await _service(db_session).handle_user_message(
        "¿Qué datos tienes del fallecimiento de Miguel?"
    )

    assert "No quiero inventar detalles" in result.answer
    assert "costura" in result.answer
    assert "1998" not in result.answer
    assert result.sources == []


async def test_leaky_fact_responder_cannot_bypass_blocked_topic(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)

    class LeakyResponder(Responder):
        async def respond(self, context: AnswerContext) -> Response:
            return Response(text="Miguel murió en 1998.")

    result = await _service(db_session, LeakyResponder()).handle_user_message(
        "¿Qué datos tienes de Elena?"
    )

    assert result.answer == SPONTANEOUS_SUPPRESSION_REPLY
    assert "murió" not in result.answer
    assert result.sources == []


async def test_food_follow_up_keeps_previous_subject(db_session: AsyncSession) -> None:
    await _import_document(db_session)
    service = _service(db_session)

    first = await service.handle_user_message("¿Qué le gusta a Elena?")
    follow = await service.handle_user_message(
        "¿Y de comer?", conversation_id=first.conversation_id
    )

    assert follow.intent == Intent.fact_query
    assert "Arroz al horno" in follow.answer
    assert "Coser" not in follow.answer
    assert "probable, sin confirmar" in follow.answer
    assert "según un familiar" in follow.answer
    food = await _fact(db_session, "Arroz al horno")
    assert food.id in {
        source.entity_id for source in follow.sources if source.entity_type == "fact"
    }


async def test_place_follow_up_keeps_previous_subject(db_session: AsyncSession) -> None:
    await _import_document(db_session)
    service = _service(db_session)

    first = await service.handle_user_message("¿Qué le gusta a Elena?")
    follow = await service.handle_user_message(
        "¿Y dónde vivió?", conversation_id=first.conversation_id
    )

    assert follow.intent == Intent.fact_query
    assert "Santa Pola" in follow.answer
    assert "probable, sin confirmar" in follow.answer
    assert "según lo que me contó la propia persona" in follow.answer
    assert "Arroz al horno" not in follow.answer


async def test_follow_up_without_matching_data_is_honest(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)
    service = _service(db_session)

    first = await service.handle_user_message("¿Qué le gusta a Elena?")
    follow = await service.handle_user_message(
        "¿Y qué música le gusta?", conversation_id=first.conversation_id
    )

    assert follow.intent == Intent.fact_query
    assert follow.answer == "No tengo guardado ningún dato sobre eso de Elena Martínez."
    assert all(source.entity_type != "fact" for source in follow.sources)


async def test_follow_up_in_new_conversation_does_not_carry_subject(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)
    service = _service(db_session)

    first = await service.handle_user_message("¿Qué le gusta a Elena?")
    follow = await service.handle_user_message("¿Y de comer?")

    assert follow.conversation_id != first.conversation_id
    assert follow.intent == Intent.general_chat
    assert "Arroz al horno" not in follow.answer


async def test_follow_up_after_unrelated_turn_does_not_carry_subject(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)
    service = _service(db_session)

    first = await service.handle_user_message("¿Qué le gusta a Elena?")
    unrelated = await service.handle_user_message(
        "Hoy hace buen día para pasear", conversation_id=first.conversation_id
    )
    assert unrelated.intent == Intent.general_chat

    follow = await service.handle_user_message(
        "¿Y de comer?", conversation_id=first.conversation_id
    )

    assert follow.intent == Intent.general_chat
    assert "Arroz al horno" not in follow.answer


async def test_follow_up_after_ambiguous_fact_query_does_not_guess(
    db_session: AsyncSession, seed
) -> None:
    service = _service(db_session)

    first = await service.handle_user_message("¿Qué aficiones tiene Paco?")
    assert first.ambiguity is not None

    follow = await service.handle_user_message(
        "¿Y de comer?", conversation_id=first.conversation_id
    )

    assert follow.intent == Intent.general_chat
    assert follow.ambiguity is None
    assert "Baker" not in follow.answer


async def test_explicit_subject_in_follow_up_overrides_carried_subject(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)
    service = _service(db_session)

    first = await service.handle_user_message("¿Qué le gusta a Elena?")
    follow = await service.handle_user_message(
        "¿Y dónde vivió Miguel?", conversation_id=first.conversation_id
    )

    assert follow.intent == Intent.fact_query
    assert "Santa Pola" not in follow.answer
    assert follow.answer == "No tengo guardado ningún dato sobre eso de Miguel Soler."


async def test_follow_up_keeps_blocked_topic_fact_withheld(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)
    service = _service(db_session)

    first = await service.handle_user_message("¿Qué datos tienes de Elena?")
    assert "Coser" in first.answer

    follow = await service.handle_user_message(
        "¿Y qué más datos tienes?", conversation_id=first.conversation_id
    )

    assert follow.intent == Intent.fact_query
    assert "Coser" in follow.answer
    assert "Santa Pola" in follow.answer
    assert "1998" not in follow.answer
    assert "murió" not in follow.answer
    death = await _fact(db_session, "Miguel murió en 1998")
    assert death.id not in {
        source.entity_id for source in follow.sources if source.entity_type == "fact"
    }


async def test_sensitive_follow_up_does_not_bypass_careful_redirect(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)
    service = _service(db_session)

    first = await service.handle_user_message("¿Qué le gusta a Elena?")
    follow = await service.handle_user_message(
        "¿Y sobre la muerte?", conversation_id=first.conversation_id
    )

    assert "No quiero inventar detalles" in follow.answer
    assert "1998" not in follow.answer
    assert follow.sources == []


async def test_follow_up_with_event_marker_stays_a_fact_query(
    db_session: AsyncSession,
) -> None:
    await _import_document(db_session)
    service = _service(db_session)

    first = await service.handle_user_message("¿Qué le gusta a Elena?")
    follow = await service.handle_user_message(
        "¿Y cuándo nació?", conversation_id=first.conversation_id
    )

    assert follow.intent == Intent.fact_query
    assert "Santa Pola" in follow.answer
