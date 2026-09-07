import json
import uuid

import httpx
import pytest

from app.domain.enums import (
    EpistemicKind,
    Intent,
    RelationshipStatus,
    RelationshipType,
)
from app.llm.openai_compatible import OpenAICompatibleResponder
from app.retrieval.context_builder import (
    AnswerContext,
    RelationshipItem,
)
from app.retrieval.entity_resolver import PersonCandidate, ResolutionKind


def _person(name: str = "Francisco Navarro") -> PersonCandidate:
    return PersonCandidate(id=uuid.uuid4(), display_name=name)


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
    person = _person()
    relation = _verified_relation()
    context = _person_context(person, relation)
    payload = (
        '{"answer": "Francisco era amigo de Antonio.", '
        f'"sources": [{{"entity_type": "person", "entity_id": "{person.id}"}}, '
        f'{{"entity_type": "relationship", "entity_id": "{relation.entity_id}"}}]}}'
    )
    async with _client_for(payload) as client:
        response = await _responder(client).respond(context)

    assert response.text == "Francisco era amigo de Antonio."
    by_type = {source.entity_type: source for source in response.sources}
    assert str(by_type["person"].entity_id) == str(person.id)
    assert by_type["person"].usage_type == "subject"
    assert str(by_type["relationship"].entity_id) == str(relation.entity_id)
    assert by_type["relationship"].usage_type == "verified_fact"


@pytest.mark.asyncio
async def test_usage_type_is_derived_from_context_not_from_model() -> None:
    person = _person()
    relation = _verified_relation()
    context = _person_context(person, relation)
    payload = (
        '{"answer": "ok", '
        f'"sources": [{{"entity_type": "relationship", "entity_id": "{relation.entity_id}", "usage_type": "hallucinated"}}]}}'
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
