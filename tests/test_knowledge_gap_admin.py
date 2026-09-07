import uuid

from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import KnowledgeGapStatus
from app.models import Fact, Memory
from app.repositories.conversation import ConversationRepository
from app.repositories.fact import FactRepository
from app.repositories.knowledge_gap import KnowledgeGapRepository
from tests.fixtures.universe import SeedUniverse

API = "/api/v1/admin"


async def _gap(
    db_session: AsyncSession, *, reference: str = "eustaquio"
) -> object:
    conversation = await ConversationRepository(db_session).get_or_create(None)
    gap, _ = await KnowledgeGapRepository(db_session).upsert(
        gap_type="person_unknown",
        canonical_reference=reference,
        conversation_id=conversation.id,
        description=f"User asked about unknown person '{reference}'.",
    )
    return gap


# --------------------------------------------------------------------------
# Read surface and filtering
# --------------------------------------------------------------------------


async def test_admin_knowledge_gaps_list_and_filter_by_status(
    db_session: AsyncSession, app_client: AsyncClient
):
    await _gap(db_session, reference="eustaquio")
    await _gap(db_session, reference="rigoberto")

    listed = await app_client.get(f"{API}/knowledge-gaps")
    assert listed.status_code == 200
    body = listed.json()
    assert {g["canonical_reference"] for g in body} == {"eustaquio", "rigoberto"}
    assert all(g["status"] == "observed" for g in body)

    observed = await app_client.get(
        f"{API}/knowledge-gaps", params={"status": "observed"}
    )
    assert len(observed.json()) == 2

    eustaquio = next(
        g for g in observed.json() if g["canonical_reference"] == "eustaquio"
    )
    await app_client.patch(
        f"{API}/knowledge-gaps/{eustaquio['id']}", json={"status": "resolved"}
    )
    resolved = await app_client.get(
        f"{API}/knowledge-gaps", params={"status": "resolved"}
    )
    assert [g["canonical_reference"] for g in resolved.json()] == ["eustaquio"]

    observed = await app_client.get(
        f"{API}/knowledge-gaps", params={"status": "observed"}
    )
    assert [g["canonical_reference"] for g in observed.json()] == ["rigoberto"]


async def test_admin_knowledge_gap_listing_accepts_status_enum(
    db_session: AsyncSession, app_client: AsyncClient
):
    await _gap(db_session)
    for status in KnowledgeGapStatus:
        response = await app_client.get(
            f"{API}/knowledge-gaps", params={"status": status.value}
        )
        assert response.status_code == 200


# --------------------------------------------------------------------------
# Resolution transitions
# --------------------------------------------------------------------------


async def test_admin_resolve_knowledge_gap_links_person_and_fact(
    db_session: AsyncSession, app_client: AsyncClient, seed: SeedUniverse
):
    gap = await _gap(db_session)
    fact = (await FactRepository(db_session).list())[0]

    response = await app_client.patch(
        f"{API}/knowledge-gaps/{gap.id}",
        json={
            "status": "resolved",
            "resolved_person_id": str(seed.person_ids["maria"]),
            "resolved_fact_id": str(fact.id),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "resolved"
    assert body["resolved_person_id"] == str(seed.person_ids["maria"])
    assert body["resolved_fact_id"] == str(fact.id)
    assert body["occurrence_count"] == 1


async def test_admin_ignore_knowledge_gap(
    db_session: AsyncSession, app_client: AsyncClient
):
    gap = await _gap(db_session)

    response = await app_client.patch(
        f"{API}/knowledge-gaps/{gap.id}", json={"status": "ignored"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"

    ignored = await app_client.get(
        f"{API}/knowledge-gaps", params={"status": "ignored"}
    )
    assert [g["id"] for g in ignored.json()] == [str(gap.id)]


async def test_admin_knowledge_gap_patch_404_for_unknown(
    app_client: AsyncClient,
):
    response = await app_client.patch(
        f"{API}/knowledge-gaps/{uuid.uuid4()}", json={"status": "resolved"}
    )
    assert response.status_code == 404


async def test_admin_knowledge_gap_resolution_reference_requires_resolved_status(
    db_session: AsyncSession, app_client: AsyncClient
):
    gap = await _gap(db_session)
    response = await app_client.patch(
        f"{API}/knowledge-gaps/{gap.id}",
        json={"status": "observed", "resolved_person_id": str(uuid.uuid4())},
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------
# Auditability and no-promotion guarantees
# --------------------------------------------------------------------------


async def test_admin_knowledge_gap_resolution_is_auditable(
    db_session: AsyncSession, app_client: AsyncClient
):
    gap = await _gap(db_session)
    await app_client.patch(
        f"{API}/knowledge-gaps/{gap.id}", json={"status": "resolved"}
    )

    listed = await app_client.get(f"{API}/knowledge-gaps")
    detail = next(g for g in listed.json() if g["id"] == str(gap.id))
    assert detail["status"] == "resolved"
    assert detail["updated_at"] is not None


async def test_admin_knowledge_gap_resolution_never_promotes(
    db_session: AsyncSession, app_client: AsyncClient
):
    gap = await _gap(db_session)
    fact_before = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()
    memory_before = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()

    await app_client.patch(
        f"{API}/knowledge-gaps/{gap.id}", json={"status": "resolved"}
    )

    fact_after = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()
    memory_after = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()
    assert fact_after == fact_before
    assert memory_after == memory_before
