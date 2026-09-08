import uuid

from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.deterministic import DeterministicResponder
from app.models import Claim, Person, PersonAlias
from app.retrieval.disambiguation import DisambiguationResolver
from app.services.conversation import ConversationService
from tests.fixtures.universe import SeedUniverse

API = "/api/v1/chat"

_GENERAL_REPLY = "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


async def _chat(
    app_client: AsyncClient,
    message: str,
    conversation_id: str | None = None,
) -> dict:
    payload = {"message": message}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = await app_client.post(API, json=payload)
    assert response.status_code == 200
    return response.json()


def _person_subjects(body: dict) -> set[str]:
    return {
        source["entity_id"]
        for source in body["sources"]
        if source["entity_type"] == "person" and source["usage_type"] == "subject"
    }


def _person(*, first: str, last: str | None = None, description: str | None = None, aliases=()):
    return Person(
        id=uuid.uuid4(),
        first_name=first,
        last_name=last,
        display_name=f"{first} {last}" if last else first,
        description=description,
        aliases=[PersonAlias(alias=alias, alias_type="nickname") for alias in aliases],
    )


class _StaticPersonRepo:
    def __init__(self, people: list[Person]) -> None:
        self._by_id = {person.id: person for person in people}

    async def get(self, person_id: uuid.UUID) -> Person | None:
        return self._by_id.get(person_id)


def _navarro() -> Person:
    return _person(
        first="Francisco",
        last="Navarro",
        description="Old friend of Antonio.",
        aliases=["Paco"],
    )


def _garcia() -> Person:
    return _person(
        first="Paco",
        last="García",
        description="Local baker, known as 'Paco el panadero'.",
        aliases=["Paco"],
    )


# --------------------------------------------------------------------------
# DisambiguationResolver unit tests
# --------------------------------------------------------------------------


async def test_ordinal_follow_up_picks_indexed_candidate():
    navarro, garcia = _navarro(), _garcia()
    resolver = DisambiguationResolver(_StaticPersonRepo([navarro, garcia]))

    assert (await resolver.resolve("el primero", [navarro.id, garcia.id])).id == navarro.id
    assert (await resolver.resolve("la primera", [navarro.id, garcia.id])).id == navarro.id
    assert (await resolver.resolve("1", [navarro.id, garcia.id])).id == navarro.id
    assert (await resolver.resolve("el segundo", [navarro.id, garcia.id])).id == garcia.id
    assert (await resolver.resolve("2", [navarro.id, garcia.id])).id == garcia.id
    assert (await resolver.resolve("el último", [navarro.id, garcia.id])).id == garcia.id
    assert (await resolver.resolve("la última", [navarro.id, garcia.id])).id == garcia.id


async def test_ordinal_out_of_range_returns_none():
    navarro, garcia = _navarro(), _garcia()
    resolver = DisambiguationResolver(_StaticPersonRepo([navarro, garcia]))
    assert await resolver.resolve("el tercero", [navarro.id, garcia.id]) is None


async def test_full_name_phrase_wins_over_shared_alias():
    navarro, garcia = _navarro(), _garcia()
    resolver = DisambiguationResolver(_StaticPersonRepo([navarro, garcia]))

    # "paco garcia" folds to "paco garcia": the García display name is the
    # longest contained phrase, beating the shared "paco" alias of Navarro.
    assert (await resolver.resolve("paco garcia", [navarro.id, garcia.id])).id == garcia.id
    assert (await resolver.resolve("Paco García", [navarro.id, garcia.id])).id == garcia.id


async def test_shared_alias_alone_returns_none():
    navarro, garcia = _navarro(), _garcia()
    resolver = DisambiguationResolver(_StaticPersonRepo([navarro, garcia]))
    assert await resolver.resolve("paco", [navarro.id, garcia.id]) is None


async def test_first_name_follow_up_picks_candidate():
    navarro, garcia = _navarro(), _garcia()
    resolver = DisambiguationResolver(_StaticPersonRepo([navarro, garcia]))
    assert (await resolver.resolve("Francisco", [navarro.id, garcia.id])).id == navarro.id


async def test_description_keyword_picks_unique_candidate():
    navarro, garcia = _navarro(), _garcia()
    resolver = DisambiguationResolver(_StaticPersonRepo([navarro, garcia]))
    assert (
        await resolver.resolve("el panadero", [navarro.id, garcia.id])
    ).id == garcia.id


async def test_description_keyword_matching_several_candidates_returns_none():
    first = _person(first="Ana", description="Vivía cerca de aquel panadero.")
    second = _person(first="Luis", description="Era el panadero del pueblo.")
    resolver = DisambiguationResolver(_StaticPersonRepo([first, second]))
    assert await resolver.resolve("el panadero", [first.id, second.id]) is None


async def test_unrelated_follow_up_returns_none():
    navarro, garcia = _navarro(), _garcia()
    resolver = DisambiguationResolver(_StaticPersonRepo([navarro, garcia]))
    assert await resolver.resolve("hola", [navarro.id, garcia.id]) is None
    assert await resolver.resolve("qué día es hoy", [navarro.id, garcia.id]) is None


async def test_empty_candidates_returns_none():
    resolver = DisambiguationResolver(_StaticPersonRepo([]))
    assert await resolver.resolve("el panadero", []) is None


# --------------------------------------------------------------------------
# API integration: multi-turn disambiguation
# --------------------------------------------------------------------------


async def test_ordinal_follow_up_resolves_second_candidate(
    app_client: AsyncClient, seed: SeedUniverse
):
    first = await _chat(app_client, "¿Quién era Paco?")
    assert first["ambiguity"] is not None
    second_candidate = first["ambiguity"]["candidates"][1]

    follow = await _chat(app_client, "el segundo", first["conversation_id"])
    assert follow["intent"] == "person_query"
    assert follow["ambiguity"] is None
    assert follow["gap"] is None
    assert _person_subjects(follow) == {second_candidate["id"]}


async def test_description_follow_up_resolves_panadero(
    app_client: AsyncClient, seed: SeedUniverse
):
    first = await _chat(app_client, "¿Quién era Paco?")
    assert first["ambiguity"] is not None

    follow = await _chat(app_client, "el panadero", first["conversation_id"])
    assert follow["intent"] == "person_query"
    assert follow["ambiguity"] is None
    assert follow["gap"] is None
    assert "Paco García" in follow["answer"]
    assert _person_subjects(follow) == {str(seed.person_ids["paco_garcia"])}


async def test_full_name_follow_up_resolves_garcia(
    app_client: AsyncClient, seed: SeedUniverse
):
    first = await _chat(app_client, "¿Quién era Paco?")
    assert first["ambiguity"] is not None

    follow = await _chat(app_client, "Paco García", first["conversation_id"])
    assert follow["intent"] == "person_query"
    assert follow["ambiguity"] is None
    assert _person_subjects(follow) == {str(seed.person_ids["paco_garcia"])}


async def test_first_name_follow_up_resolves_navarro(
    app_client: AsyncClient, seed: SeedUniverse
):
    first = await _chat(app_client, "¿Quién era Paco?")
    assert first["ambiguity"] is not None

    follow = await _chat(app_client, "Francisco", first["conversation_id"])
    assert follow["intent"] == "person_query"
    assert follow["ambiguity"] is None
    assert "amigo de Antonio" in follow["answer"]
    assert _person_subjects(follow) == {str(seed.person_ids["paco_navarro"])}


async def test_unrelated_follow_up_routes_normally(
    app_client: AsyncClient, seed: SeedUniverse
):
    first = await _chat(app_client, "¿Quién era Paco?")
    assert first["ambiguity"] is not None

    follow = await _chat(app_client, "hola", first["conversation_id"])
    assert follow["intent"] == "general_chat"
    assert follow["answer"] == _GENERAL_REPLY
    assert follow["ambiguity"] is None
    assert follow["gap"] is None
    assert follow["sources"] == []


async def test_person_query_without_pending_ambiguity_is_unchanged(
    app_client: AsyncClient, seed: SeedUniverse
):
    body = await _chat(app_client, "¿Quién era Francisco?")
    assert body["intent"] == "person_query"
    assert body["ambiguity"] is None
    assert body["gap"] is None
    assert "amigo de Antonio" in body["answer"]
    assert _person_subjects(body) == {str(seed.person_ids["paco_navarro"])}


# --------------------------------------------------------------------------
# Behavior guarantees on the disambiguation turn
# --------------------------------------------------------------------------


async def test_disambiguation_turn_skips_claim_ingestion_and_gap(
    db_session: AsyncSession, seed: SeedUniverse
):
    service = ConversationService(
        db_session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
    )
    first = await service.handle_user_message("¿Quién era Paco?")
    assert first.ambiguity is not None

    claims_before = (
        await db_session.execute(select(func.count()).select_from(Claim))
    ).scalar_one()
    assert claims_before == 0

    # "él era el panadero" carries a first-person past signal ("era"), so if the
    # disambiguation selection were routed as a normal conversational turn it
    # would ingest a candidate claim. The turn must resolve Paco García instead
    # and skip claim ingestion and knowledge-gap handling entirely.
    follow = await service.handle_user_message(
        "él era el panadero", conversation_id=first.conversation_id
    )
    assert follow.intent.value == "person_query"
    assert follow.ambiguity is None
    assert follow.gap is None
    assert not follow.gap_created
    subject_ids = {
        source.entity_id
        for source in follow.sources
        if source.entity_type == "person" and source.usage_type == "subject"
    }
    assert subject_ids == {seed.person_ids["paco_garcia"]}

    claims_after = (
        await db_session.execute(select(func.count()).select_from(Claim))
    ).scalar_one()
    assert claims_after == 0
