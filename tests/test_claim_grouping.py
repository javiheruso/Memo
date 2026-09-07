import uuid
from datetime import UTC, datetime

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.base import ClaimCandidate
from app.claims.deterministic import DeterministicClaimExtractor
from app.claims.grouping import DeterministicClaimGrouper, normalize_claim_text
from app.domain.enums import ClaimStatus
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.base import Responder, Response
from app.models import Claim, ClaimGroup, ClaimGroupMember
from app.repositories.claim import ClaimRepository
from app.repositories.claim_group import ClaimGroupRepository
from app.repositories.conversation import ConversationRepository
from app.retrieval.context_builder import AnswerContext
from app.services.conversation import ConversationService

API = "/api/v1/chat"

_GENERAL_REPLY = "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."


class StubResponder(Responder):
    """Fake responder used to prove grouping never changes the answer."""

    def __init__(self, text: str = "stub answer") -> None:
        self._text = text

    async def respond(self, context: AnswerContext) -> Response:
        return Response(text=self._text)


class BrokenGrouper:
    """Grouper that always fails, used to prove fail-safe behavior."""

    def group_key(self, text: str) -> str:
        raise RuntimeError("grouper exploded")


async def _groups(db_session: AsyncSession) -> list[ClaimGroup]:
    result = await db_session.execute(select(ClaimGroup))
    return list(result.scalars())


async def _claims(db_session: AsyncSession) -> list[Claim]:
    result = await db_session.execute(select(Claim))
    return list(result.scalars())


def test_normalize_claim_text_folds_case_accents_and_punctuation():
    assert normalize_claim_text("Vivíamos en Madrid.") == "viviamos en madrid"
    assert normalize_claim_text("viviamos en madrid") == "viviamos en madrid"
    assert normalize_claim_text("  Recuerdo  que... teníamos   UN perro? ") == (
        "recuerdo que teniamos un perro"
    )


def test_deterministic_grouper_groups_same_normalized_text():
    grouper = DeterministicClaimGrouper()
    assert grouper.group_key("Vivíamos en Madrid.") == grouper.group_key(
        "viviamos en madrid"
    )


def test_paraphrases_do_not_group():
    grouper = DeterministicClaimGrouper()
    assert grouper.group_key("Vivíamos en Madrid.") != grouper.group_key(
        "Cuando estábamos en Madrid"
    )
    assert grouper.group_key("Vivíamos en Madrid.") != grouper.group_key(
        "Teníamos un perro."
    )
    assert grouper.group_key("Recuerdo que teníamos un perro.") != grouper.group_key(
        "Teníamos un perro."
    )


async def test_ingestion_groups_duplicate_claims_across_messages(
    db_session: AsyncSession,
):
    conversations = ConversationRepository(db_session)
    conversation = await conversations.get_or_create(None)
    first = await conversations.add_message(
        conversation.id, speaker="user", text="Vivíamos en Madrid."
    )
    second = await conversations.add_message(
        conversation.id, speaker="user", text="Vivíamos en Madrid."
    )
    repo = ClaimRepository(db_session)
    group_repo = ClaimGroupRepository(db_session)

    claim_a = (
        await repo.add_many(
            [ClaimCandidate(original_text="Vivíamos en Madrid.")],
            conversation_id=conversation.id,
            message_id=first.id,
        )
    )[0]
    await group_repo.attach(claim_a, "viviamos en madrid")

    claim_b = (
        await repo.add_many(
            [ClaimCandidate(original_text="Vivíamos en Madrid.")],
            conversation_id=conversation.id,
            message_id=second.id,
        )
    )[0]
    await group_repo.attach(claim_b, "viviamos en madrid")

    groups = await _groups(db_session)
    assert len(groups) == 1
    group = groups[0]
    assert group.canonical_text == "viviamos en madrid"
    assert group.total_mentions == 2
    assert group.distinct_conversations == 1
    assert group.distinct_days == 1
    assert group.first_seen_at <= group.last_seen_at


async def test_distinct_conversations_and_days_counters(db_session: AsyncSession):
    conversations = ConversationRepository(db_session)
    conv_a = await conversations.get_or_create(None)
    conv_b = await conversations.get_or_create(None)
    message = await conversations.add_message(conv_a.id, speaker="user", text="x")
    repo = ClaimRepository(db_session)
    group_repo = ClaimGroupRepository(db_session)

    day_one = datetime(2026, 8, 1, 10, 0, tzinfo=UTC)
    day_two = datetime(2026, 8, 2, 10, 0, tzinfo=UTC)

    claims = await repo.add_many(
        [ClaimCandidate(original_text="Teníamos un perro.")],
        conversation_id=conv_a.id,
        message_id=message.id,
        first_seen_at=day_one,
    )
    await group_repo.attach(claims[0], "teniamos un perro")

    other_message = await conversations.add_message(conv_b.id, speaker="user", text="x")
    claims_b = await repo.add_many(
        [ClaimCandidate(original_text="Teníamos un perro.")],
        conversation_id=conv_b.id,
        message_id=other_message.id,
        first_seen_at=day_two,
    )
    await group_repo.attach(claims_b[0], "teniamos un perro")

    groups = await _groups(db_session)
    assert len(groups) == 1
    group = groups[0]
    assert group.total_mentions == 2
    assert group.distinct_conversations == 2
    assert group.distinct_days == 2
    assert group.first_seen_at.date() == day_one.date()
    assert group.last_seen_at.date() == day_two.date()


async def test_distinct_claims_create_distinct_groups(db_session: AsyncSession):
    conversations = ConversationRepository(db_session)
    conversation = await conversations.get_or_create(None)
    message = await conversations.add_message(
        conversation.id, speaker="user", text="Vivíamos en Madrid. Teníamos un perro."
    )
    repo = ClaimRepository(db_session)
    group_repo = ClaimGroupRepository(db_session)

    claims = await repo.add_many(
        [
            ClaimCandidate(original_text="Vivíamos en Madrid."),
            ClaimCandidate(original_text="Teníamos un perro."),
        ],
        conversation_id=conversation.id,
        message_id=message.id,
    )
    for claim in claims:
        await group_repo.attach(claim, normalize_claim_text(claim.original_text))

    groups = await _groups(db_session)
    assert len(groups) == 2
    assert {group.canonical_text for group in groups} == {
        "viviamos en madrid",
        "teniamos un perro",
    }
    assert all(group.total_mentions == 1 for group in groups)


async def test_chat_pipeline_groups_claims_and_never_promotes(
    app_client: AsyncClient, db_session: AsyncSession
):
    first = await app_client.post(
        API, json={"message": "Vivíamos en Madrid en 1972."}
    )
    assert first.status_code == 200
    assert first.json()["answer"] == _GENERAL_REPLY
    conversation_id = first.json()["conversation_id"]

    second = await app_client.post(
        API, json={"message": "Vivíamos en Madrid en 1972."}
    )
    assert second.status_code == 200
    assert second.json()["answer"] == _GENERAL_REPLY
    assert second.json()["conversation_id"] != conversation_id

    groups = await _groups(db_session)
    assert len(groups) == 1
    assert groups[0].canonical_text == "viviamos en madrid en 1972"
    assert groups[0].total_mentions == 2
    assert groups[0].distinct_conversations == 2

    stored_claims = await _claims(db_session)
    assert len(stored_claims) == 2
    assert all(claim.status == ClaimStatus.candidate for claim in stored_claims)


async def test_query_intent_creates_no_groups(
    app_client: AsyncClient, seed, db_session: AsyncSession
):
    response = await app_client.post(API, json={"message": "¿Quién era Francisco?"})
    assert response.status_code == 200
    assert response.json()["intent"] == "person_query"
    assert await _groups(db_session) == []
    assert await _claims(db_session) == []


async def test_grouping_failure_does_not_affect_response(
    db_session: AsyncSession, seed
):
    service = ConversationService(
        db_session,
        StubResponder("respuesta de prueba"),
        embedder=DeterministicEmbedder(),
        claim_extractor=DeterministicClaimExtractor(),
        claim_grouper=BrokenGrouper(),
    )
    result = await service.handle_user_message("Vivíamos en Madrid.")
    assert result.answer == "respuesta de prueba"
    assert result.intent.value == "general_chat"
    assert len(await _claims(db_session)) == 1
    assert await _groups(db_session) == []
    members = (await db_session.execute(select(ClaimGroupMember))).scalars().all()
    assert members == []


async def test_admin_claims_listing_does_not_require_groups(
    app_client: AsyncClient, db_session: AsyncSession
):
    response = await app_client.get("/api/v1/admin/claims")
    assert response.status_code == 200
    assert response.json() == []


async def test_admin_claims_listing_includes_group_id(
    app_client: AsyncClient, db_session: AsyncSession
):
    response = await app_client.post(
        API, json={"message": "Vivíamos en Madrid en 1972."}
    )
    assert response.status_code == 200

    listing = await app_client.get("/api/v1/admin/claims")
    assert listing.status_code == 200
    claims = listing.json()
    assert len(claims) == 1
    assert claims[0]["original_text"] == "Vivíamos en Madrid en 1972"
    assert claims[0]["status"] == "candidate"
    assert claims[0]["claim_group_id"] is not None


async def test_admin_claim_groups_listing_shows_counters(
    app_client: AsyncClient, db_session: AsyncSession
):
    await app_client.post(API, json={"message": "Vivíamos en Madrid en 1972."})
    await app_client.post(API, json={"message": "Vivíamos en Madrid en 1972."})

    listing = await app_client.get("/api/v1/admin/claim-groups")
    assert listing.status_code == 200
    groups = listing.json()
    assert len(groups) == 1
    assert groups[0]["canonical_text"] == "viviamos en madrid en 1972"
    assert groups[0]["total_mentions"] == 2
    assert groups[0]["distinct_conversations"] == 2
    assert groups[0]["distinct_days"] == 1


async def test_admin_claim_group_detail_returns_member_claims(
    app_client: AsyncClient, db_session: AsyncSession
):
    await app_client.post(API, json={"message": "Vivíamos en Madrid en 1972."})

    groups = await _groups(db_session)
    assert len(groups) == 1
    group_id = groups[0].id

    detail = await app_client.get(f"/api/v1/admin/claim-groups/{group_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == str(group_id)
    assert body["total_mentions"] == 1
    assert len(body["claims"]) == 1
    assert body["claims"][0]["original_text"] == "Vivíamos en Madrid en 1972"
    assert body["claims"][0]["claim_group_id"] == str(group_id)


async def test_admin_claim_group_detail_404_for_unknown_group(
    app_client: AsyncClient, db_session: AsyncSession
):
    response = await app_client.get(
        f"/api/v1/admin/claim-groups/{uuid.uuid4()}"
    )
    assert response.status_code == 404
