from datetime import UTC, datetime

from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.base import ClaimCandidate
from app.claims.contradiction import (
    ContradictionSignal,
    DeterministicContradictionDetector,
    fact_surface_text,
)
from app.domain.enums import ClaimStatus, FactStatus
from app.embeddings.deterministic import DeterministicEmbedder
from app.llm.base import Responder, Response
from app.llm.deterministic import DeterministicResponder
from app.models import Claim, ClaimGroup, ClaimGroupContradiction, Fact
from app.repositories.claim import ClaimRepository
from app.repositories.claim_group import ClaimGroupRepository
from app.repositories.claim_group_contradiction import (
    ClaimGroupContradictionRepository,
)
from app.repositories.conversation import ConversationRepository
from app.repositories.fact import FactRepository
from app.retrieval.context_builder import AnswerContext
from app.schemas.fact import FactCreate
from app.services.conversation import ConversationService
from tests.fixtures.universe import SeedUniverse

API = "/api/v1"

_GENERAL_REPLY = "Estoy aquí para conversar. Cuéntame de qué te apetece hablar."


class StubResponder(Responder):
    """Fake responder used to prove detection never changes the answer."""

    def __init__(self, text: str = "stub answer") -> None:
        self._text = text

    async def respond(self, context: AnswerContext) -> Response:
        return Response(text=self._text)


class BrokenDetector:
    """Detector that always fails, used to prove fail-safe behavior."""

    name = "broken"

    def detect(self, claim_text: str, fact_text: str) -> ContradictionSignal | None:
        raise RuntimeError("detector exploded")


async def _verified_fact(
    db_session: AsyncSession, seed: SeedUniverse, *, value_text: str, status: FactStatus
) -> Fact:
    repo = FactRepository(db_session)
    return await repo.create(
        FactCreate(
            subject_person_id=seed.person_ids["carmen"],
            fact_type="residence",
            value_text=value_text,
            status=status,
            source_id=seed.source_ids["family_records"],
        )
    )


async def _contradictions(db_session: AsyncSession) -> list[ClaimGroupContradiction]:
    result = await db_session.execute(select(ClaimGroupContradiction))
    return list(result.scalars())


async def _groups(db_session: AsyncSession) -> list[ClaimGroup]:
    result = await db_session.execute(select(ClaimGroup))
    return list(result.scalars())


# --------------------------------------------------------------------------
# Detector behavior
# --------------------------------------------------------------------------


def test_detector_flags_negated_restatement_of_fact():
    detector = DeterministicContradictionDetector()
    signal = detector.detect("No vivíamos en Madrid.", "Vivíamos en Madrid")
    assert signal is not None
    assert signal.matched_terms == ("madrid", "viviamos")


def test_detector_accepts_extra_framing_words_in_claim():
    detector = DeterministicContradictionDetector()
    signal = detector.detect("Recuerdo que no vivíamos en Madrid.", "Vivíamos en Madrid")
    assert signal is not None


def test_detector_requires_negation():
    detector = DeterministicContradictionDetector()
    assert detector.detect("Vivíamos en Madrid.", "Vivíamos en Madrid") is None


def test_detector_requires_all_fact_content_tokens():
    detector = DeterministicContradictionDetector()
    assert detector.detect("No vivíamos en Madrid.", "Vivíamos en Barcelona") is None
    assert detector.detect("No fuimos a la playa.", "Fuimos a la playa y cenamos") is None


def test_detector_supports_other_negation_markers():
    detector = DeterministicContradictionDetector()
    assert detector.detect("Nunca vivimos en Madrid.", "Vivimos en Madrid") is not None
    assert detector.detect("Ya no vivíamos en Madrid.", "Vivíamos en Madrid") is not None
    assert detector.detect("Jamás tuve un perro.", "Tuve un perro") is not None


def test_detector_ignores_mismatched_language_fact_surfaces():
    detector = DeterministicContradictionDetector()
    assert detector.detect("No pescábamos.", "Fishing") is None


def test_detector_returns_none_for_stopword_only_surface():
    detector = DeterministicContradictionDetector()
    assert detector.detect("No lo recuerdo.", "en la casa") is None


def test_detector_is_deterministic_and_named():
    detector = DeterministicContradictionDetector()
    assert detector.name == "deterministic"
    first = detector.detect("No vivíamos en Madrid.", "Vivíamos en Madrid")
    second = detector.detect("No vivíamos en Madrid.", "Vivíamos en Madrid")
    assert first == second


def test_fact_surface_text_joins_narrative_fields():
    assert fact_surface_text(value_text="Vivíamos en Madrid", place_text="Centro") == (
        "Vivíamos en Madrid Centro"
    )
    assert fact_surface_text(value_text=None, place_text=None) == ""
    assert fact_surface_text(value_text="Solo texto") == "Solo texto"


# --------------------------------------------------------------------------
# Non-destructive persistence
# --------------------------------------------------------------------------


async def test_ingestion_flags_contradiction_and_never_promotes(
    db_session: AsyncSession, seed: SeedUniverse
):
    from app.models import Memory

    fact_count_before = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()
    memory_count_before = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()
    fact = await _verified_fact(
        db_session, seed, value_text="Vivíamos en Madrid", status=FactStatus.verified
    )

    service = ConversationService(
        db_session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
    )
    result = await service.handle_user_message("No vivíamos en Madrid.")

    assert result.answer == _GENERAL_REPLY
    claims = (
        (await db_session.execute(select(Claim))).scalars().all()
    )
    assert len(claims) == 1
    assert claims[0].status == ClaimStatus.candidate
    assert claims[0].original_text == "No vivíamos en Madrid"

    groups = await _groups(db_session)
    assert len(groups) == 1
    assert groups[0].needs_review is True

    signals = await _contradictions(db_session)
    assert len(signals) == 1
    assert signals[0].claim_group_id == groups[0].id
    assert signals[0].fact_id == fact.id
    assert signals[0].detector_name == "deterministic"
    assert signals[0].detection_count == 1
    assert "madrid" in signals[0].matched_terms

    fact_count = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()
    memory_count = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()
    assert fact_count == fact_count_before + 1
    assert memory_count == memory_count_before


async def test_repeated_detection_refreshes_counter_not_evidence(
    db_session: AsyncSession, seed: SeedUniverse
):
    await _verified_fact(
        db_session, seed, value_text="Vivíamos en Madrid", status=FactStatus.verified
    )
    service = ConversationService(
        db_session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
    )
    await service.handle_user_message("No vivíamos en Madrid.")
    await service.handle_user_message("No vivíamos en Madrid.")

    signals = await _contradictions(db_session)
    assert len(signals) == 1
    assert signals[0].detection_count == 2
    groups = await _groups(db_session)
    assert len(groups) == 1
    assert groups[0].needs_review is True


async def test_only_verified_facts_are_compared(
    db_session: AsyncSession, seed: SeedUniverse
):
    await _verified_fact(
        db_session, seed, value_text="Vivíamos en Madrid", status=FactStatus.pending
    )
    service = ConversationService(
        db_session,
        DeterministicResponder(),
        embedder=DeterministicEmbedder(),
    )
    result = await service.handle_user_message("No vivíamos en Madrid.")

    assert result.answer == _GENERAL_REPLY
    assert await _contradictions(db_session) == []
    groups = await _groups(db_session)
    assert len(groups) == 1
    assert groups[0].needs_review is False


async def test_contradiction_repo_upsert_is_append_only(
    db_session: AsyncSession, seed: SeedUniverse
):
    conversations = ConversationRepository(db_session)
    conversation = await conversations.get_or_create(None)
    message = await conversations.add_message(
        conversation.id, speaker="user", text="No vivíamos en Madrid."
    )
    claim = (
        await ClaimRepository(db_session).add_many(
            [ClaimCandidate(original_text="No vivíamos en Madrid.")],
            conversation_id=conversation.id,
            message_id=message.id,
        )
    )[0]
    group = await ClaimGroupRepository(db_session).attach(claim, "no viviamos en madrid")
    fact = await _verified_fact(
        db_session, seed, value_text="Vivíamos en Madrid", status=FactStatus.verified
    )
    repo = ClaimGroupContradictionRepository(db_session)

    first = await repo.upsert(
        claim_group_id=group.id,
        fact_id=fact.id,
        detector_name="deterministic",
        matched_terms="madrid, viviamos",
        now=datetime(2026, 8, 31, 10, 0, tzinfo=UTC),
    )
    second = await repo.upsert(
        claim_group_id=group.id,
        fact_id=fact.id,
        detector_name="deterministic",
        matched_terms="madrid, viviamos",
        now=datetime(2026, 8, 31, 11, 0, tzinfo=UTC),
    )
    await db_session.commit()

    signals = await _contradictions(db_session)
    assert len(signals) == 1
    assert signals[0].id == first.id == second.id
    assert signals[0].detection_count == 2
    assert signals[0].first_detected_at == datetime(2026, 8, 31, 10, 0, tzinfo=UTC)
    assert signals[0].last_detected_at == datetime(2026, 8, 31, 11, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# Fail-safe behavior
# --------------------------------------------------------------------------


async def test_detector_failure_never_affects_ingestion_or_response(
    db_session: AsyncSession, seed: SeedUniverse
):
    await _verified_fact(
        db_session, seed, value_text="Vivíamos en Madrid", status=FactStatus.verified
    )
    service = ConversationService(
        db_session,
        StubResponder("respuesta de prueba"),
        embedder=DeterministicEmbedder(),
        claim_contradiction_detector=BrokenDetector(),
    )
    result = await service.handle_user_message("No vivíamos en Madrid.")

    assert result.answer == "respuesta de prueba"
    assert len(await _groups(db_session)) == 1
    claims = (await db_session.execute(select(Claim))).scalars().all()
    assert len(claims) == 1
    assert claims[0].status == ClaimStatus.candidate
    assert await _contradictions(db_session) == []


async def test_detector_without_signal_leaves_group_clean(
    db_session: AsyncSession, seed: SeedUniverse
):
    await _verified_fact(
        db_session, seed, value_text="Vivíamos en Madrid", status=FactStatus.verified
    )
    service = ConversationService(
        db_session,
        StubResponder("respuesta de prueba"),
        embedder=DeterministicEmbedder(),
    )
    await service.handle_user_message("Vivíamos en Madrid.")

    assert await _contradictions(db_session) == []
    groups = await _groups(db_session)
    assert len(groups) == 1
    assert groups[0].needs_review is False


# --------------------------------------------------------------------------
# Admin summaries
# --------------------------------------------------------------------------


async def test_admin_claim_contradictions_listing_summary(
    app_client: AsyncClient, seed: SeedUniverse
):
    fact = (await app_client.post(
        f"{API}/admin/facts",
        json={
            "subject_person_id": str(seed.person_ids["carmen"]),
            "fact_type": "residence",
            "value_text": "Vivíamos en Madrid",
            "status": "verified",
            "source_id": str(seed.source_ids["family_records"]),
        },
    )).json()

    chat = await app_client.post(
        f"{API}/chat", json={"message": "No vivíamos en Madrid."}
    )
    assert chat.status_code == 200
    assert chat.json()["answer"] == _GENERAL_REPLY

    listing = await app_client.get(f"{API}/admin/claim-contradictions")
    assert listing.status_code == 200
    items = listing.json()
    assert len(items) == 1
    item = items[0]
    assert item["fact_id"] == fact["id"]
    assert item["detector_name"] == "deterministic"
    assert item["detection_count"] == 1
    assert "madrid" in item["matched_terms"]
    assert item["group"]["needs_review"] is True
    assert item["group"]["canonical_text"] == "no viviamos en madrid"
    assert item["fact"]["value_text"] == "Vivíamos en Madrid"
    assert item["fact"]["status"] == "verified"


async def test_admin_claim_group_detail_includes_contradictions(
    app_client: AsyncClient, seed: SeedUniverse
):
    await app_client.post(
        f"{API}/admin/facts",
        json={
            "subject_person_id": str(seed.person_ids["carmen"]),
            "fact_type": "residence",
            "value_text": "Vivíamos en Madrid",
            "status": "verified",
            "source_id": str(seed.source_ids["family_records"]),
        },
    )
    await app_client.post(f"{API}/chat", json={"message": "No vivíamos en Madrid."})

    groups = await app_client.get(f"{API}/admin/claim-groups")
    assert groups.status_code == 200
    group_id = groups.json()[0]["id"]

    detail = await app_client.get(f"{API}/admin/claim-groups/{group_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["needs_review"] is True
    assert len(body["contradictions"]) == 1
    contradiction = body["contradictions"][0]
    assert contradiction["detector_name"] == "deterministic"
    assert contradiction["fact"]["value_text"] == "Vivíamos en Madrid"


async def test_admin_claim_groups_filter_by_needs_review(
    app_client: AsyncClient, seed: SeedUniverse
):
    await app_client.post(
        f"{API}/admin/facts",
        json={
            "subject_person_id": str(seed.person_ids["carmen"]),
            "fact_type": "residence",
            "value_text": "Vivíamos en Madrid",
            "status": "verified",
            "source_id": str(seed.source_ids["family_records"]),
        },
    )
    await app_client.post(f"{API}/chat", json={"message": "No vivíamos en Madrid."})
    await app_client.post(f"{API}/chat", json={"message": "Teníamos un perro."})

    flagged = await app_client.get(
        f"{API}/admin/claim-groups", params={"needs_review": "true"}
    )
    assert flagged.status_code == 200
    assert [g["canonical_text"] for g in flagged.json()] == ["no viviamos en madrid"]

    clean = await app_client.get(
        f"{API}/admin/claim-groups", params={"needs_review": "false"}
    )
    assert [g["canonical_text"] for g in clean.json()] == ["teniamos un perro"]
