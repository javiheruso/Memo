import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.domain.conversation_context import ConversationStyle, ConversationTurn
from app.domain.enums import (
    EpistemicKind,
    EventStatus,
    Intent,
    RelationshipStatus,
    RelationshipType,
)
from app.llm.claim_grounding import audit_answer_claims
from app.llm.openai_compatible import OpenAICompatibleResponder
from app.retrieval.context_builder import (
    AnswerContext,
    EventItem,
    FactItem,
    RelationshipItem,
)
from app.retrieval.entity_resolver import PersonCandidate, ResolutionKind


def _person(name: str = "Francisco Navarro") -> PersonCandidate:
    return PersonCandidate(id=uuid.uuid4(), display_name=name)


def _fact() -> FactItem:
    return FactItem(
        entity_id=uuid.uuid4(),
        text="Vivió en Guardamar",
        status="verified",
        epistemic=EpistemicKind.verified_fact,
    )


def _event() -> EventItem:
    return EventItem(
        entity_id=uuid.uuid4(),
        title="Visita de María",
        start_at=datetime.now(UTC) + timedelta(days=2),
        status=EventStatus.confirmed,
    )


def _verified_relation(
    other: str = "Antonio", relationship_type: RelationshipType = RelationshipType.friend_of
) -> RelationshipItem:
    return RelationshipItem(
        entity_id=uuid.uuid4(),
        other_display_name=other,
        relationship_type=relationship_type,
        status=RelationshipStatus.verified,
        epistemic=EpistemicKind.verified_fact,
    )


def _person_context(
    person: PersonCandidate, relation: RelationshipItem | None = None
) -> AnswerContext:
    return AnswerContext(
        intent=Intent.person_query,
        term="Francisco",
        resolution=ResolutionKind.resolved,
        person=person,
        verified_relations=[relation] if relation else [],
    )


def _client_for(
    content: str,
    *,
    status: int = 200,
    error: Exception | None = None,
    capture: list | None = None,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(request)
        if error is not None:
            raise error
        return httpx.Response(
            status, json={"choices": [{"message": {"content": content}}]}
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _responder(client: httpx.AsyncClient, **kwargs) -> OpenAICompatibleResponder:
    return OpenAICompatibleResponder(
        base_url=kwargs.pop("base_url", "http://llm.test"),
        model=kwargs.pop("model", "test-model"),
        api_key=kwargs.pop("api_key", "secret"),
        client=client,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_valid_model_output_returns_text_and_validated_sources() -> None:
    person = _person("Francisco")
    fact = _fact()
    context = AnswerContext(
        intent=Intent.fact_query,
        term="Francisco",
        resolution=ResolutionKind.resolved,
        person=person,
        facts=[fact],
    )
    payload = (
        '{"answer": "Vivió en Guardamar.", '
        f'"sources": [{{"entity_type": "person", "entity_id": "{person.id}"}}, '
        f'{{"entity_type": "fact", "entity_id": "{fact.entity_id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)

    assert response.text == "Vivió en Guardamar."
    by_type = {source.entity_type: source for source in response.sources}
    assert str(by_type["person"].entity_id) == str(person.id)
    assert by_type["person"].usage_type == "subject"
    assert str(by_type["fact"].entity_id) == str(fact.entity_id)
    assert by_type["fact"].usage_type == "verified_fact"


@pytest.mark.asyncio
async def test_usage_type_is_derived_from_context_not_from_model() -> None:
    person = _person("Francisco")
    fact = _fact()
    context = AnswerContext(
        intent=Intent.fact_query,
        term="Francisco",
        resolution=ResolutionKind.resolved,
        person=person,
        facts=[fact],
    )
    payload = (
        '{"answer": "Vivió en Guardamar.", '
        f'"sources": [{{"entity_type": "fact", "entity_id": "{fact.entity_id}", "usage_type": "hallucinated"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)
    assert response.sources[0].usage_type == "verified_fact"


@pytest.mark.asyncio
async def test_invented_source_id_falls_back_to_deterministic() -> None:
    person = _person()
    context = _person_context(person, _verified_relation())
    invented = uuid.uuid4()
    payload = (
        '{"answer": "respuesta inventada", '
        f'"sources": [{{"entity_type": "relationship", "entity_id": "{invented}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)

    assert "era amigo de Antonio" in response.text
    assert all(str(source.entity_id) != str(invented) for source in response.sources)


@pytest.mark.asyncio
async def test_wrong_entity_type_for_known_id_falls_back() -> None:
    person = _person()
    context = _person_context(person, _verified_relation())
    payload = (
        '{"answer": "ok", '
        f'"sources": [{{"entity_type": "memory", "entity_id": "{person.id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)
    assert "era amigo de Antonio" in response.text


@pytest.mark.asyncio
async def test_non_json_output_falls_back() -> None:
    person = _person()
    context = _person_context(person, _verified_relation())
    async with _client_for("por supuesto que sí, claro que sí") as client:
        response = await _responder(client).respond(context)
    assert "era amigo de Antonio" in response.text


@pytest.mark.asyncio
async def test_json_wrapped_in_code_fences_is_parsed() -> None:
    person = _person()
    context = _person_context(person)
    payload = (
        '```json\n{"answer": "Francisco.", "sources": '
        f'[{{"entity_type": "person", "entity_id": "{person.id}"}}]}}\n```'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)
    assert response.text == "Francisco."
    assert len(response.sources) == 1


@pytest.mark.asyncio
async def test_empty_answer_falls_back() -> None:
    context = _person_context(_person(), _verified_relation())
    async with _client_for('{"answer": "   ", "sources": []}') as client:
        response = await _responder(client).respond(context)
    assert "era amigo de Antonio" in response.text


@pytest.mark.asyncio
async def test_malformed_source_entry_falls_back() -> None:
    context = _person_context(_person(), _verified_relation())
    async with _client_for(
        '{"answer": "ok", "sources": [{"entity_type": "person"}]}'
    ) as client:
        response = await _responder(client).respond(context)
    assert "era amigo de Antonio" in response.text


@pytest.mark.asyncio
async def test_sources_not_a_list_falls_back() -> None:
    context = _person_context(_person(), _verified_relation())
    async with _client_for('{"answer": "ok", "sources": "person"}') as client:
        response = await _responder(client).respond(context)
    assert "era amigo de Antonio" in response.text


@pytest.mark.asyncio
async def test_http_error_falls_back() -> None:
    context = _person_context(_person(), _verified_relation())
    async with _client_for("ignored", status=500) as client:
        response = await _responder(client).respond(context)
    assert "era amigo de Antonio" in response.text


@pytest.mark.asyncio
async def test_network_error_falls_back() -> None:
    context = _person_context(_person(), _verified_relation())
    async with _client_for(
        "ignored", error=httpx.ConnectError("connection refused")
    ) as client:
        response = await _responder(client).respond(context)
    assert "era amigo de Antonio" in response.text


@pytest.mark.asyncio
async def test_timeout_falls_back() -> None:
    context = _person_context(_person(), _verified_relation())
    async with _client_for(
        "ignored", error=httpx.TimeoutException("request timed out")
    ) as client:
        response = await _responder(client).respond(context)
    assert "era amigo de Antonio" in response.text


@pytest.mark.asyncio
async def test_http_fallback_logs_status_without_sensitive_fields(caplog) -> None:
    context = _person_context(_person(), _verified_relation())
    marker = "PRIVATE-RESPONSE-BODY-503"
    caplog.set_level(logging.WARNING, logger="app.llm.openai_compatible")
    async with _client_for(marker, status=503) as client:
        response = await _responder(client, api_key="super-secret-api-key").respond(
            context
        )

    assert "era amigo de Antonio" in response.text
    log = caplog.text
    assert "timeout_or_transport" in log
    assert "503" in log
    for forbidden in (
        marker,
        "super-secret-api-key",
        "Authorization",
        "Bearer",
        "Francisco",
    ):
        assert forbidden not in log


@pytest.mark.asyncio
async def test_invalid_model_output_logs_no_model_content(caplog) -> None:
    context = _person_context(_person(), _verified_relation())
    marker = "LEAKED-MODEL-MARKER-987"
    payload = (
        '{"answer": "ok", "sources": '
        f'[{{"entity_type": "person", "entity_id": "{marker}"}}]}}'
    )
    caplog.set_level(logging.WARNING, logger="app.llm.openai_compatible")
    async with _client_for(payload) as client:
        response = await _responder(client, api_key="super-secret-api-key").respond(
            context
        )

    assert "era amigo de Antonio" in response.text
    log = caplog.text
    assert "invalid_json_or_schema" in log
    for forbidden in (marker, "super-secret-api-key", "Authorization", "Bearer"):
        assert forbidden not in log


@pytest.mark.asyncio
async def test_network_error_logs_only_failure_class(caplog) -> None:
    context = _person_context(_person(), _verified_relation())
    marker = "https://user:pass@internal.example/secret?token=leaked"
    caplog.set_level(logging.WARNING, logger="app.llm.openai_compatible")
    async with _client_for("ignored", error=httpx.ConnectError(marker)) as client:
        response = await _responder(client, api_key="super-secret-api-key").respond(
            context
        )

    assert "era amigo de Antonio" in response.text
    log = caplog.text
    assert "timeout_or_transport" in log
    for forbidden in (
        "internal.example",
        "token=leaked",
        "user:pass",
        "super-secret-api-key",
        "Authorization",
        "Bearer",
    ):
        assert forbidden not in log


@pytest.mark.asyncio
async def test_request_targets_chat_completions_with_model_and_auth() -> None:
    person = _person()
    context = _person_context(person)
    capture: list[httpx.Request] = []
    async with _client_for('{"answer": "ok", "sources": []}', capture=capture) as client:
        await _responder(client).respond(context)
    request = capture[0]
    assert request.url == "http://llm.test/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert body["temperature"] == 0
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]
    assert schema["name"] == "memo_reply"
    assert schema["strict"] is True
    assert set(schema["schema"]["required"]) == {"answer", "sources"}
    assert request.headers["Authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_payload_has_no_tools_and_prompt_exposes_no_db_mechanism() -> None:
    person = _person()
    context = _person_context(person, _verified_relation())
    capture: list[httpx.Request] = []
    async with _client_for('{"answer": "ok", "sources": []}', capture=capture) as client:
        await _responder(client).respond(context)
    body = json.loads(capture[0].content)
    assert "tools" not in body
    combined = body["messages"][0]["content"] + body["messages"][1]["content"]
    for forbidden in ("SELECT", "sqlalchemy", "repository", "tool", "session"):
        assert forbidden.lower() not in combined.lower()


@pytest.mark.asyncio
async def test_fallback_sources_are_truthful() -> None:
    person = _person()
    relation = _verified_relation()
    context = _person_context(person, relation)
    async with _client_for("no json at all") as client:
        response = await _responder(client).respond(context)
    by_type = {source.entity_type: source for source in response.sources}
    assert str(by_type["person"].entity_id) == str(person.id)
    assert str(by_type["relationship"].entity_id) == str(relation.entity_id)


def test_base_url_and_model_are_required() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleResponder(base_url="", model="m")
    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleResponder(base_url="http://llm.test", model="")


def test_from_settings_maps_config() -> None:
    from app.core.config import Settings

    responder = OpenAICompatibleResponder.from_settings(
        Settings(
            responder="openai_compatible",
            llm_base_url="http://local",
            llm_model="local-model",
            llm_api_key=None,
            llm_timeout_seconds=5,
        )
    )
    assert responder._base_url == "http://local"
    assert responder._model == "local-model"
    assert responder._api_key is None
    assert responder._timeout == 5


@pytest.mark.asyncio
async def test_chat_api_falls_back_closed_on_invented_sources(
    app_client, seed, db_session
) -> None:
    from app.llm.provider import get_responder
    from app.main import app

    invented = uuid.uuid4()
    payload = (
        '{"answer": "respuesta inventada", "sources": '
        f'[{{"entity_type": "relationship", "entity_id": "{invented}"}}]}}'
    )
    async with _client_for(payload) as client:
        responder = _responder(client)
        app.dependency_overrides[get_responder] = lambda: responder
        try:
            response = await app_client.post(
                "/api/v1/chat", json={"message": "¿Quién era Francisco?"}
            )
        finally:
            app.dependency_overrides.pop(get_responder, None)

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "person_query"
    assert "amigo de Antonio" in body["answer"]
    assert all(source["entity_id"] != str(invented) for source in body["sources"])
    assert any(
        source["entity_type"] == "person"
        and source["entity_id"] == str(seed.person_ids["paco_navarro"])
        for source in body["sources"]
    )


@pytest.mark.asyncio
async def test_general_chat_follow_up_prompt_is_bounded_and_source_free() -> None:
    context = AnswerContext(intent=Intent.general_chat, term=None)
    capture: list[httpx.Request] = []
    async with _client_for(
        '{"answer": "¿Qué te apetece hacer hoy?", "sources": []}',
        capture=capture,
    ) as client:
        response = await _responder(client).respond(
            context, message="Hola, hoy hace un día precioso."
        )

    body = json.loads(capture[0].content)
    combined = body["messages"][0]["content"] + "\n" + body["messages"][1]["content"]
    assert "at most one brief, generic follow-up question" in combined
    assert "Stored data available this turn: none" in combined
    assert response.text == "¿Qué te apetece hacer hoy?"
    assert response.sources == []


@pytest.mark.asyncio
async def test_general_chat_invented_source_still_falls_back_closed() -> None:
    context = AnswerContext(intent=Intent.general_chat, term=None)
    invented = uuid.uuid4()
    payload = (
        '{"answer": "Claro, lo recuerdo perfectamente.", "sources": '
        f'[{{"entity_type": "memory", "entity_id": "{invented}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context, message="Hola")

    assert response.text == "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."
    assert response.sources == []
    assert all(str(source.entity_id) != str(invented) for source in response.sources)


@pytest.mark.asyncio
async def test_positive_topic_offer_is_rejected_when_not_allowed() -> None:
    context = AnswerContext(intent=Intent.general_chat, term=None)
    style = ConversationStyle(
        positive_topics=("la costura", "la playa"),
        offer_positive_topics=False,
    )
    async with _client_for(
        '{"answer": "Claro, ¿te apetece hablar de la costura?", "sources": []}'
    ) as client:
        response = await _responder(client).respond(
            context, message="No me apetece hablar de costura.", style=style
        )

    assert "costura" not in response.text
    assert "playa" not in response.text
    assert response.sources == []


@pytest.mark.asyncio
async def test_positive_topic_named_by_the_person_can_be_echoed_as_a_statement() -> None:
    context = AnswerContext(intent=Intent.general_chat, term=None)
    style = ConversationStyle(
        positive_topics=("la costura", "la playa"),
        offer_positive_topics=False,
    )
    async with _client_for(
        '{"answer": "Me alegra que te guste la costura.", "sources": []}'
    ) as client:
        response = await _responder(client).respond(
            context, message="Prefiero hablar de la costura.", style=style
        )

    assert response.text == "Me alegra que te guste la costura."
    assert response.sources == []


@pytest.mark.asyncio
async def test_positive_topic_named_by_the_person_cannot_be_reoffered() -> None:
    context = AnswerContext(intent=Intent.general_chat, term=None)
    style = ConversationStyle(
        positive_topics=("la costura", "la playa"),
        offer_positive_topics=False,
    )
    async with _client_for(
        '{"answer": "Claro, ¿te apetece hablar de la costura?", "sources": []}'
    ) as client:
        response = await _responder(client).respond(
            context, message="Prefiero hablar de la costura.", style=style
        )

    assert "costura" not in response.text
    assert response.sources == []


@pytest.mark.asyncio
async def test_positive_topic_offer_is_bounded_to_one_topic() -> None:
    context = AnswerContext(intent=Intent.general_chat, term=None)
    style = ConversationStyle(
        positive_topics=("la costura", "la playa"),
        offer_positive_topics=True,
    )
    async with _client_for(
        '{"answer": "¿Te apetece hablar de la costura o la playa?", "sources": []}'
    ) as client:
        response = await _responder(client).respond(
            context, message="Hola", style=style
        )

    assert "costura" not in response.text
    assert "playa" not in response.text


@pytest.mark.asyncio
async def test_single_allowed_positive_topic_offer_is_accepted() -> None:
    context = AnswerContext(intent=Intent.general_chat, term=None)
    style = ConversationStyle(
        positive_topics=("la costura", "la playa"),
        offer_positive_topics=True,
    )
    async with _client_for(
        '{"answer": "¿Te apetece hablar de la costura?", "sources": []}'
    ) as client:
        response = await _responder(client).respond(
            context, message="Hola", style=style
        )

    assert response.text == "¿Te apetece hablar de la costura?"
    assert response.sources == []


@pytest.mark.asyncio
async def test_no_data_model_cannot_claim_stored_projects_or_memories() -> None:
    context = AnswerContext(intent=Intent.general_chat, term=None)
    async with _client_for(
        '{"answer": "Tengo tus proyectos y recuerdos guardados.", "sources": []}'
    ) as client:
        response = await _responder(client).respond(context, message="Hola")

    assert "proyectos" not in response.text
    assert "recuerdos" not in response.text
    assert response.sources == []


@pytest.mark.asyncio
async def test_sensitive_redirect_question_is_accepted() -> None:
    context = AnswerContext(
        intent=Intent.memory_query,
        sensitive_topic_requested="la muerte de Antonio",
        sensitive_redirect_topic="la costura",
    )
    answer = (
        "Entiendo que quieres hablar de la muerte de Antonio. No quiero "
        "inventar detalles. Si te parece, podemos hablar de la costura."
    )
    async with _client_for(
        f'{{"answer": "{answer}", "sources": []}}'
    ) as client:
        response = await _responder(client).respond(
            context, message="¿Te acuerdas del hospital?"
        )

    expected = (
        "Entiendo que quieres hablar de la muerte de Antonio. No quiero "
        "inventar detalles, así que prefiero no extenderme. Si te parece, "
        "podemos hablar de la costura."
    )
    assert response.text == expected
    assert response.sources == []


@pytest.mark.asyncio
async def test_memory_answer_with_conversational_framing_is_grounded() -> None:
    memory = _memory_for_grounding()
    context = AnswerContext(
        intent=Intent.memory_query,
        memories=[memory],
    )
    answer = (
        "Tengo un recuerdo guardado que encaja con eso: "
        f"'{memory.narrative}'"
    )
    payload = (
        f'{{"answer": "{answer}", "sources": '
        f'[{{"entity_type": "memory", "entity_id": "{memory.entity_id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(
            context, message="¿Te acuerdas de las tardes de pesca?"
        )

    assert response.text == (
        "Tengo un recuerdo guardado que encaja con eso: "
        f"Recuerdo guardado: '{memory.narrative}'"
    )
    assert len(response.sources) == 1


@pytest.mark.asyncio
async def test_memory_answer_with_invented_detail_still_falls_back() -> None:
    memory = _memory_for_grounding()
    context = AnswerContext(
        intent=Intent.memory_query,
        memories=[memory],
    )
    payload = (
        '{"answer": "También iban a la playa en verano.", "sources": '
        f'[{{"entity_type": "memory", "entity_id": "{memory.entity_id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(
            context, message="¿Te acuerdas de las tardes de pesca?"
        )

    assert "playa" not in response.text
    assert response.text.startswith("Tengo un recuerdo guardado")


def _memory_for_grounding():
    from app.domain.enums import MemoryStatus
    from app.retrieval.context_builder import MemoryItem

    return MemoryItem(
        entity_id=uuid.uuid4(),
        title="Tardes de pesca",
        narrative=(
            "Carmen recuerda las tardes de pesca con Antonio y Paco en Guardamar."
        ),
        memory_status=MemoryStatus.consolidated,
        epistemic=EpistemicKind.memory,
    )


@pytest.mark.asyncio
async def test_json_schema_enumerates_only_allowed_source_ids() -> None:
    person = _person()
    relation = _verified_relation()
    context = _person_context(person, relation)
    capture: list[httpx.Request] = []
    payload = (
        '{"answer": "Francisco era amigo de Antonio.", "sources": '
        f'[{{"entity_type": "person", "entity_id": "{person.id}"}}]}}'
    )
    async with _client_for(payload, capture=capture) as client:
        await _responder(client).respond(context)

    schema = json.loads(capture[0].content)["response_format"]["json_schema"][
        "schema"
    ]
    item_schema = schema["properties"]["sources"]["items"]
    assert set(item_schema["properties"]["entity_id"]["enum"]) == {
        str(person.id),
        str(relation.entity_id),
    }
    assert item_schema["properties"]["entity_type"]["enum"] == [
        "person",
        "relationship",
    ]


@pytest.mark.asyncio
async def test_json_schema_forbids_sources_when_context_has_none() -> None:
    context = AnswerContext(intent=Intent.general_chat, term=None)
    capture: list[httpx.Request] = []
    async with _client_for(
        '{"answer": "Hola.", "sources": []}', capture=capture
    ) as client:
        response = await _responder(client).respond(context, message="Hola")

    schema = json.loads(capture[0].content)["response_format"]["json_schema"][
        "schema"
    ]
    assert schema["properties"]["sources"]["maxItems"] == 0
    assert response.text == "Hola."


def _ambiguous_context() -> AnswerContext:
    candidates = [_person("Carmen"), _person("Antonio")]
    return AnswerContext(
        intent=Intent.person_query,
        term="Gómez",
        resolution=ResolutionKind.ambiguous,
        candidates=candidates,
    )


@pytest.mark.asyncio
async def test_ambiguity_without_a_question_falls_back() -> None:
    context = _ambiguous_context()
    payload = (
        '{"answer": "Gómez era el marido de Carmen.", "sources": '
        f'[{{"entity_type": "person", "entity_id": "{context.candidates[0].id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)

    assert "Hay varias personas llamadas Gómez" in response.text
    assert "¿A cuál te refieres?" in response.text


@pytest.mark.asyncio
async def test_ambiguity_question_without_sources_gets_candidate_sources() -> None:
    context = _ambiguous_context()
    async with _client_for(
        '{"answer": "Hay varias personas. ¿A cuál te refieres?", "sources": []}'
    ) as client:
        response = await _responder(client).respond(context)

    assert response.text == "Hay varias personas. ¿A cuál te refieres?"
    assert {str(source.entity_id) for source in response.sources} == {
        str(candidate.id) for candidate in context.candidates
    }
    assert all(source.usage_type == "candidate" for source in response.sources)


@pytest.mark.asyncio
async def test_no_data_fact_with_only_subject_gets_subject_source() -> None:
    person = _person("Antonio")
    context = AnswerContext(
        intent=Intent.fact_query,
        term="Antonio",
        resolution=ResolutionKind.resolved,
        person=person,
    )
    async with _client_for(
        '{"answer": "No tengo información sobre eso de Antonio.", "sources": []}'
    ) as client:
        response = await _responder(client).respond(context)

    assert response.text == "No tengo información sobre eso de Antonio."
    assert [(source.entity_type, str(source.entity_id), source.usage_type) for source in response.sources] == [
        ("person", str(person.id), "subject")
    ]


@pytest.mark.asyncio
async def test_memory_claim_with_no_stored_memory_falls_back() -> None:
    context = AnswerContext(intent=Intent.memory_query, term=None)
    async with _client_for(
        '{"answer": "Sí, me acuerdo del crucero. Fue muy divertido.", "sources": []}'
    ) as client:
        response = await _responder(client).respond(
            context, message="¿Recuerdas aquel crucero?"
        )

    assert "crucero" not in response.text
    assert response.text == "No tengo guardado ese recuerdo todavía."


@pytest.mark.asyncio
async def test_acceptance_can_repeat_the_cited_memory_narrative() -> None:
    memory = _memory_for_grounding()
    context = AnswerContext(
        intent=Intent.memory_query,
        memories=[memory],
        accepted_offer=True,
    )
    text = (
        "Tengo un recuerdo guardado que encaja con eso: "
        f"'{memory.narrative}'"
    )
    payload = (
        f'{{"answer": "{text}", "sources": '
        f'[{{"entity_type": "memory", "entity_id": "{memory.entity_id}"}}]}}'
    )
    history = (
        ConversationTurn(role="assistant", text=text),
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(
            context, message="Sí.", history=history
        )

    assert response.text == (
        "Tengo un recuerdo guardado que encaja con eso: "
        f"Recuerdo guardado: '{memory.narrative}'"
    )


@pytest.mark.asyncio
async def test_acceptance_still_rejects_a_repeated_offer_question() -> None:
    context = AnswerContext(intent=Intent.general_chat, accepted_offer=True)
    offer = "¿Te gustaría hablar de la costura?"
    payload = f'{{"answer": "{offer}", "sources": []}}'
    history = (ConversationTurn(role="assistant", text=offer),)
    async with _client_for(payload) as client:
        response = await _responder(client).respond(
            context, message="Sí.", history=history
        )

    assert response.text != offer
    assert "?" not in response.text


@pytest.mark.asyncio
async def test_event_query_without_event_source_falls_back() -> None:
    person = _person("María")
    context = AnswerContext(
        intent=Intent.event_query,
        term="María",
        resolution=ResolutionKind.resolved,
        person=person,
        events=[_event()],
    )
    payload = (
        '{"answer": "María viene el domingo.", "sources": '
        f'[{{"entity_type": "person", "entity_id": "{person.id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)

    assert "Visita de María" in response.text
    assert "confirmado" in response.text


@pytest.mark.asyncio
async def test_fact_query_without_fact_source_falls_back() -> None:
    person = _person("Francisco")
    context = AnswerContext(
        intent=Intent.fact_query,
        term="Francisco",
        resolution=ResolutionKind.resolved,
        person=person,
        facts=[_fact()],
    )
    payload = (
        '{"answer": "Vivió en Guardamar.", "sources": '
        f'[{{"entity_type": "person", "entity_id": "{person.id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)

    assert "Vivió en Guardamar" in response.text
    assert any(
        source.entity_type == "fact" for source in response.sources
    )


@pytest.mark.asyncio
async def test_memory_query_without_memory_source_falls_back() -> None:
    person = _person("Carmen")
    memory = _memory_for_grounding()
    context = AnswerContext(
        intent=Intent.memory_query,
        person=person,
        memories=[memory],
    )
    payload = (
        f'{{"answer": "{memory.narrative}", "sources": '
        f'[{{"entity_type": "person", "entity_id": "{person.id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)

    assert "Tengo un recuerdo guardado" in response.text
    assert any(
        source.entity_type == "memory" for source in response.sources
    )


@pytest.mark.asyncio
async def test_sensitive_memory_query_cannot_accept_invented_personal_details() -> None:
    context = AnswerContext(
        intent=Intent.memory_query,
        sensitive_topic_requested="la enfermedad de Lucía",
        sensitive_redirect_topic="la jardinería",
    )
    payload = (
        '{"answer": "Lucía estuvo ingresada dos semanas en 1999.", '
        '"sources": []}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(
            context, message="¿Te acuerdas de aquella operación?"
        )

    assert response.text.startswith("Entiendo que quieres hablar de")
    assert "No quiero inventar detalles" in response.text
    assert "ingresada" not in response.text
    assert "1999" not in response.text
    assert response.sources == []


@pytest.mark.asyncio
async def test_person_fact_attributed_to_another_person_falls_back() -> None:
    person = _person("Elena Ruiz")
    relation = RelationshipItem(
        entity_id=uuid.uuid4(),
        other_display_name="Tomás Vidal",
        relationship_type=RelationshipType.child_of,
        status=RelationshipStatus.verified,
        epistemic=EpistemicKind.verified_fact,
    )
    fact = FactItem(
        entity_id=uuid.uuid4(),
        text="arquitecta",
        status="verified",
        epistemic=EpistemicKind.verified_fact,
    )
    context = AnswerContext(
        intent=Intent.person_query,
        term="Elena",
        resolution=ResolutionKind.resolved,
        person=person,
        verified_relations=[relation],
        facts=[fact],
    )
    payload = (
        '{"answer": "Tomás Vidal es arquitecto. Elena Ruiz es la hija de '
        'Tomás Vidal.", "sources": ['
        f'{{"entity_type": "person", "entity_id": "{person.id}"}}, '
        f'{{"entity_type": "relationship", "entity_id": "{relation.entity_id}"}}, '
        f'{{"entity_type": "fact", "entity_id": "{fact.entity_id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)

    assert response.text == "Elena Ruiz era hijo de Tomás Vidal."
    assert "arquitect" not in response.text
    assert {source.entity_type for source in response.sources} == {
        "person",
        "relationship",
    }


@pytest.mark.asyncio
async def test_reversed_relationship_claim_falls_back_to_stored_direction() -> None:
    person = _person("Elena Ruiz")
    relation = RelationshipItem(
        entity_id=uuid.uuid4(),
        other_display_name="Tomás Vidal",
        relationship_type=RelationshipType.child_of,
        status=RelationshipStatus.verified,
        epistemic=EpistemicKind.verified_fact,
    )
    context = AnswerContext(
        intent=Intent.person_query,
        term="Elena",
        resolution=ResolutionKind.resolved,
        person=person,
        verified_relations=[relation],
    )
    payload = (
        '{"answer": "Tomás Vidal era hijo de Elena Ruiz.", "sources": ['
        f'{{"entity_type": "person", "entity_id": "{person.id}"}}, '
        f'{{"entity_type": "relationship", "entity_id": "{relation.entity_id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)

    assert response.text == "Elena Ruiz era hijo de Tomás Vidal."
    assert "Tomás Vidal era hijo" not in response.text


@pytest.mark.asyncio
async def test_fact_query_without_stored_fact_rejects_invented_detail() -> None:
    person = _person("Paco")
    context = AnswerContext(
        intent=Intent.fact_query,
        term="Paco",
        resolution=ResolutionKind.resolved,
        person=person,
    )
    payload = (
        '{"answer": "Paco vivía en Denia.", "sources": ['
        f'{{"entity_type": "person", "entity_id": "{person.id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(
            context, message="¿Dónde vivió Paco?"
        )

    assert response.text == "No tengo guardado ningún dato sobre eso de Paco."
    assert "Denia" not in response.text
    assert [(source.entity_type, source.usage_type) for source in response.sources] == [
        ("person", "subject")
    ]


@pytest.mark.asyncio
async def test_fact_query_claim_not_supported_by_stored_fact_falls_back() -> None:
    person = _person("Paco")
    fact = FactItem(
        entity_id=uuid.uuid4(),
        text="panadero",
        status="verified",
        epistemic=EpistemicKind.verified_fact,
    )
    context = AnswerContext(
        intent=Intent.fact_query,
        term="Paco",
        resolution=ResolutionKind.resolved,
        person=person,
        facts=[fact],
    )
    payload = (
        '{"answer": "Paco vivía en Denia.", "sources": ['
        f'{{"entity_type": "person", "entity_id": "{person.id}"}}, '
        f'{{"entity_type": "fact", "entity_id": "{fact.entity_id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(
            context, message="¿A qué se dedicaba Paco?"
        )

    assert "Denia" not in response.text
    assert "panadero" in response.text


@pytest.mark.asyncio
async def test_event_query_without_stored_event_rejects_invented_event() -> None:
    person = _person("María")
    context = AnswerContext(
        intent=Intent.event_query,
        term="María",
        resolution=ResolutionKind.resolved,
        person=person,
    )
    payload = (
        '{"answer": "La boda de María es el sábado.", "sources": ['
        f'{{"entity_type": "person", "entity_id": "{person.id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)

    assert response.text == "No tengo eventos próximos guardados para María."
    assert "boda" not in response.text


def test_claim_audit_distinguishes_supported_and_unsupported_claims() -> None:
    person = _person("Elena Ruiz")
    fact = FactItem(
        entity_id=uuid.uuid4(),
        text="arquitecta",
        status="verified",
        epistemic=EpistemicKind.verified_fact,
    )
    context = AnswerContext(
        intent=Intent.person_query,
        term="Elena",
        resolution=ResolutionKind.resolved,
        person=person,
        facts=[fact],
    )

    audit = audit_answer_claims(
        context, "Elena Ruiz es arquitecta. Tomás Vidal es arquitecto."
    )

    assert audit.supported_claims == ("Elena Ruiz es arquitecta",)
    assert audit.unsupported_claims == ("Tomás Vidal es arquitecto",)
    assert audit.source_absent_personal_claims == ()


def test_claim_audit_flags_source_absent_personal_claims() -> None:
    context = AnswerContext(intent=Intent.memory_query)

    audit = audit_answer_claims(
        context, "Lucía estuvo ingresada dos semanas en 1999."
    )

    assert audit.unsupported_claims == (
        "Lucía estuvo ingresada dos semanas en 1999",
    )
    assert audit.source_absent_personal_claims == (
        "Lucía estuvo ingresada dos semanas en 1999",
    )


def test_claim_audit_does_not_flag_honest_no_data() -> None:
    context = AnswerContext(intent=Intent.memory_query)

    audit = audit_answer_claims(
        context,
        "No tengo guardado ese recuerdo todavía.",
        message="¿Recuerdas aquel crucero?",
    )

    assert audit.unsupported_claims == ()
    assert audit.source_absent_personal_claims == ()


def test_claim_audit_does_not_skip_a_statement_with_a_question_tag() -> None:
    context = AnswerContext(intent=Intent.memory_query)

    audit = audit_answer_claims(
        context, "Lucía estuvo ingresada dos semanas en 1999, ¿verdad?"
    )

    assert audit.source_absent_personal_claims == (
        "Lucía estuvo ingresada dos semanas en 1999, ¿verdad",
    )
