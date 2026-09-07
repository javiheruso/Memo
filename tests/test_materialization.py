"""Explicit admin-only materialization of claim groups / gaps into facts/memories.

Covers auditability (append-only provenance rows, source-state transitions),
guardrails (unknown ids, ignored sources, duplicate attempts, unknown evidence
sources) and the no-promotion guarantees (claims and gap evidence are never
mutated).
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.domain.enums import ClaimStatus, MemoryStatus
from app.models import (
    Claim,
    ClaimGroup,
    ClaimGroupContradiction,
    Fact,
    KnowledgeGap,
    KnowledgeGapConversation,
    Materialization,
    Memory,
)
from app.repositories.conversation import ConversationRepository
from app.repositories.knowledge_gap import KnowledgeGapRepository
from tests.fixtures.universe import SeedUniverse

API = "/api/v1/admin"

_SKIP_ON_POSTGRES = pytest.mark.skipif(
    settings.test_database_url is not None,
    reason="relies on SQLite embedding behavior; covered by postgres smoke tests",
)


async def _claim_group(
    app_client: AsyncClient, *, text: str = "Vivíamos en Madrid."
) -> dict:
    response = await app_client.post("/api/v1/chat", json={"message": text})
    assert response.status_code == 200
    groups = (await app_client.get(f"{API}/claim-groups")).json()
    return groups[0]


async def _gap(
    db_session: AsyncSession, *, reference: str = "eustaquio"
) -> KnowledgeGap:
    conversation = await ConversationRepository(db_session).get_or_create(None)
    gap, _ = await KnowledgeGapRepository(db_session).upsert(
        gap_type="person_unknown",
        canonical_reference=reference,
        conversation_id=conversation.id,
        description=f"User asked about unknown person '{reference}'.",
    )
    return gap


async def _materializations(db_session: AsyncSession) -> list[Materialization]:
    return list((await db_session.execute(select(Materialization))).scalars())


def _fact_payload(
    seed: SeedUniverse,
    *,
    value_text: str | None = None,
    status: str = "pending",
    source_id: str | None = None,
) -> dict:
    return {
        "subject_person_id": str(seed.person_ids["carmen"]),
        "fact_type": "residence",
        "value_text": value_text,
        "status": status,
        "source_id": source_id or str(seed.source_ids["shared_by_carmen"]),
    }


def _memory_payload(
    seed: SeedUniverse,
    *,
    narrative: str | None = None,
    source_id: str | None = None,
    people: list[dict] | None = None,
) -> dict:
    payload = {
        "narrative": narrative,
        "source_id": source_id or str(seed.source_ids["shared_by_carmen"]),
    }
    if people is not None:
        payload["people"] = people
    return payload


# --------------------------------------------------------------------------
# Auditability: provenance rows, defaults, source-state transitions
# --------------------------------------------------------------------------


async def test_materialize_claim_group_fact_is_auditable(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    group = await _claim_group(app_client)

    response = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-fact",
        json=_fact_payload(seed),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["value_text"] == "viviamos en madrid"
    assert body["status"] == "pending"
    assert body["source_id"] == str(seed.source_ids["shared_by_carmen"])
    assert body["notes"] is None

    provenance = await _materializations(db_session)
    assert len(provenance) == 1
    row = provenance[0]
    assert str(row.source_type.value) == "claim_group"
    assert str(row.source_id) == group["id"]
    assert str(row.target_type.value) == "fact"
    assert str(row.target_id) == body["id"]
    assert row.created_at is not None

    stored_group = (
        await db_session.execute(
            select(ClaimGroup).where(ClaimGroup.id == uuid.UUID(group["id"]))
        )
    ).scalar_one()
    assert stored_group.status.value == "resolved"
    assert stored_group.needs_review is False
    assert stored_group.resolved_at is not None
    assert f"fact {body['id']}" in stored_group.resolution_note


async def test_materialize_claim_group_fact_honors_overrides(
    app_client: AsyncClient, seed: SeedUniverse
):
    group = await _claim_group(app_client)

    response = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-fact",
        json=_fact_payload(seed, value_text="Vivíamos en Madrid capital", status="verified"),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["value_text"] == "Vivíamos en Madrid capital"
    assert body["status"] == "verified"


async def test_materialize_claim_group_memory_is_auditable(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    group = await _claim_group(app_client, text="Íbamos a bailar los sábados.")

    response = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-memory",
        json=_memory_payload(seed),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["narrative"] == "ibamos a bailar los sabados"
    assert body["title"] == "ibamos a bailar los sabados"
    assert body["memory_status"] == "candidate"
    assert body["people"] == []

    provenance = await _materializations(db_session)
    assert len(provenance) == 1
    assert str(provenance[0].source_type.value) == "claim_group"
    assert str(provenance[0].target_type.value) == "memory"
    assert str(provenance[0].target_id) == body["id"]

    stored_group = (
        await db_session.execute(
            select(ClaimGroup).where(ClaimGroup.id == uuid.UUID(group["id"]))
        )
    ).scalar_one()
    assert stored_group.status.value == "resolved"
    assert f"memory {body['id']}" in stored_group.resolution_note


async def test_materialize_gap_fact_is_auditable_and_links_gap(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    gap = await _gap(db_session)

    response = await app_client.post(
        f"{API}/knowledge-gaps/{gap.id}/materialize-fact",
        json=_fact_payload(seed),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["value_text"] == "eustaquio"
    assert body["status"] == "pending"

    provenance = await _materializations(db_session)
    assert len(provenance) == 1
    assert str(provenance[0].source_type.value) == "knowledge_gap"
    assert str(provenance[0].source_id) == str(gap.id)
    assert str(provenance[0].target_type.value) == "fact"
    assert str(provenance[0].target_id) == body["id"]

    await db_session.refresh(gap)
    assert gap.status.value == "resolved"
    assert str(gap.resolved_fact_id) == body["id"]
    assert str(gap.resolved_person_id) == str(seed.person_ids["carmen"])


async def test_materialize_gap_memory_is_auditable(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    gap = await _gap(db_session, reference="eustaquio")

    response = await app_client.post(
        f"{API}/knowledge-gaps/{gap.id}/materialize-memory",
        json=_memory_payload(
            seed,
            narrative="Eustaquio was a neighbor of the family.",
            people=[
                {
                    "person_id": str(seed.person_ids["maria"]),
                    "role": "daughter",
                }
            ],
        ),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["narrative"] == "Eustaquio was a neighbor of the family."
    assert body["title"] == "eustaquio"
    assert body["memory_status"] == "candidate"
    assert body["people"] == [
        {"person_id": str(seed.person_ids["maria"]), "role": "daughter"}
    ]

    provenance = await _materializations(db_session)
    assert len(provenance) == 1
    assert str(provenance[0].source_type.value) == "knowledge_gap"
    assert str(provenance[0].target_type.value) == "memory"

    await db_session.refresh(gap)
    assert gap.status.value == "resolved"
    assert gap.resolved_fact_id is None
    assert str(gap.resolved_person_id) == str(seed.person_ids["maria"])


# --------------------------------------------------------------------------
# Guardrails: unknown ids, ignored sources, duplicates, unknown evidence source
# --------------------------------------------------------------------------


async def test_materialize_fact_404_for_unknown_claim_group(
    app_client: AsyncClient, seed: SeedUniverse
):
    response = await app_client.post(
        f"{API}/claim-groups/{uuid.uuid4()}/materialize-fact",
        json=_fact_payload(seed),
    )
    assert response.status_code == 404


async def test_materialize_memory_404_for_unknown_gap(
    app_client: AsyncClient, seed: SeedUniverse
):
    response = await app_client.post(
        f"{API}/knowledge-gaps/{uuid.uuid4()}/materialize-memory",
        json=_memory_payload(seed),
    )
    assert response.status_code == 404


async def test_materialize_ignored_claim_group_is_409(
    app_client: AsyncClient, seed: SeedUniverse
):
    group = await _claim_group(app_client)
    await app_client.patch(
        f"{API}/claim-groups/{group['id']}", json={"status": "ignored"}
    )

    response = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-fact",
        json=_fact_payload(seed),
    )
    assert response.status_code == 409
    assert "ignored" in response.json()["detail"]


async def test_materialize_ignored_gap_is_409(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    gap = await _gap(db_session)
    await app_client.patch(f"{API}/knowledge-gaps/{gap.id}", json={"status": "ignored"})

    response = await app_client.post(
        f"{API}/knowledge-gaps/{gap.id}/materialize-memory",
        json=_memory_payload(seed),
    )
    assert response.status_code == 409
    assert "ignored" in response.json()["detail"]


async def test_duplicate_materialization_is_409(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    group = await _claim_group(app_client)
    facts_before = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()

    first = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-fact",
        json=_fact_payload(seed),
    )
    assert first.status_code == 201

    second = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-fact",
        json=_fact_payload(seed),
    )
    assert second.status_code == 409
    assert "already materialized as fact" in second.json()["detail"]

    facts_after = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()
    assert facts_after == facts_before + 1
    provenance = await _materializations(db_session)
    assert len(provenance) == 1


async def test_claim_group_can_become_one_fact_and_one_memory(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    group = await _claim_group(app_client)

    fact = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-fact",
        json=_fact_payload(seed),
    )
    memory = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-memory",
        json=_memory_payload(seed),
    )
    assert fact.status_code == 201
    assert memory.status_code == 201

    provenance = await _materializations(db_session)
    assert {row.target_type.value for row in provenance} == {"fact", "memory"}


async def test_materialize_with_unknown_evidence_source_is_422(
    app_client: AsyncClient, seed: SeedUniverse
):
    group = await _claim_group(app_client)

    response = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-fact",
        json=_fact_payload(seed, source_id=str(uuid.uuid4())),
    )
    assert response.status_code == 422
    assert "not found" in response.json()["detail"]


# --------------------------------------------------------------------------
# Source-state transitions: materialized sources leave the review queue
# --------------------------------------------------------------------------


async def test_materialized_group_is_not_reflagged_by_new_contradicting_claim(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    await app_client.post(
        f"{API}/facts",
        json={
            "subject_person_id": str(seed.person_ids["carmen"]),
            "fact_type": "residence",
            "value_text": "Vivíamos en Madrid",
            "status": "verified",
            "source_id": str(seed.source_ids["family_records"]),
        },
    )
    group = await _claim_group(app_client, text="No vivíamos en Madrid.")
    assert group["needs_review"] is True

    materialized = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-fact",
        json=_fact_payload(seed),
    )
    assert materialized.status_code == 201

    await app_client.post("/api/v1/chat", json={"message": "No vivíamos en Madrid."})

    stored_group = (
        await db_session.execute(
            select(ClaimGroup).where(ClaimGroup.id == uuid.UUID(group["id"]))
        )
    ).scalar_one()
    assert stored_group.status.value == "resolved"
    assert stored_group.needs_review is False

    signals = (
        (await db_session.execute(select(ClaimGroupContradiction)))
        .scalars()
        .all()
    )
    assert len(signals) == 1
    assert signals[0].detection_count == 1


async def test_resolved_gap_can_still_be_materialized_but_not_twice(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    gap = await _gap(db_session)
    await app_client.patch(f"{API}/knowledge-gaps/{gap.id}", json={"status": "resolved"})

    first = await app_client.post(
        f"{API}/knowledge-gaps/{gap.id}/materialize-fact",
        json=_fact_payload(seed),
    )
    assert first.status_code == 201

    second = await app_client.post(
        f"{API}/knowledge-gaps/{gap.id}/materialize-fact",
        json=_fact_payload(seed),
    )
    assert second.status_code == 409


# --------------------------------------------------------------------------
# No-promotion guarantees
# --------------------------------------------------------------------------


async def test_materialization_never_mutates_source_evidence(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    group = await _claim_group(app_client, text="Vivíamos en Madrid.")
    claims_before = (
        (await db_session.execute(select(Claim))).scalars().all()
    )
    original_texts = {claim.original_text for claim in claims_before}
    assert original_texts == {"Vivíamos en Madrid"}

    response = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-fact",
        json=_fact_payload(seed),
    )
    assert response.status_code == 201

    claims_after = (await db_session.execute(select(Claim))).scalars().all()
    assert len(claims_after) == len(claims_before)
    assert all(claim.status == ClaimStatus.candidate for claim in claims_after)
    assert all(
        claim.original_text in original_texts for claim in claims_after
    )

    gap_count = (
        await db_session.execute(select(func.count()).select_from(KnowledgeGap))
    ).scalar_one()
    assert gap_count == 0


async def test_gap_materialization_keeps_gap_occurrences_append_only(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    gap = await _gap(db_session)
    conversation = (await db_session.execute(select(KnowledgeGapConversation))).scalars().all()
    assert len(conversation) == 1
    memory_count_before = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()

    response = await app_client.post(
        f"{API}/knowledge-gaps/{gap.id}/materialize-memory",
        json=_memory_payload(seed),
    )
    assert response.status_code == 201

    conversations_after = (
        (await db_session.execute(select(KnowledgeGapConversation))).scalars().all()
    )
    assert len(conversations_after) == 1

    memory_count = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()
    assert memory_count == memory_count_before + 1


@_SKIP_ON_POSTGRES
async def test_materialized_memory_on_sqlite_keeps_embedding_null(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    group = await _claim_group(app_client, text="Íbamos a la playa en verano.")

    response = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-memory",
        json=_memory_payload(seed),
    )
    assert response.status_code == 201
    memory_id = uuid.UUID(response.json()["id"])
    stored = (
        await db_session.execute(select(Memory).where(Memory.id == memory_id))
    ).scalar_one()
    assert stored.memory_status == MemoryStatus.candidate
    assert stored.embedding is None
