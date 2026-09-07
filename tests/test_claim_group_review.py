import uuid

from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import ClaimStatus
from app.models import Claim, ClaimGroupContradiction, Fact, Memory
from tests.fixtures.universe import SeedUniverse

API = "/api/v1"


async def _verified_fact(
    app_client: AsyncClient, seed: SeedUniverse, *, value_text: str = "Vivíamos en Madrid"
) -> dict:
    response = await app_client.post(
        f"{API}/admin/facts",
        json={
            "subject_person_id": str(seed.person_ids["carmen"]),
            "fact_type": "residence",
            "value_text": value_text,
            "status": "verified",
            "source_id": str(seed.source_ids["family_records"]),
        },
    )
    assert response.status_code == 201
    return response.json()


async def _flag_contradiction(
    app_client: AsyncClient, seed: SeedUniverse
) -> dict:
    """Create a verified fact and a contradicting claim through the pipeline."""
    await _verified_fact(app_client, seed)
    response = await app_client.post(
        f"{API}/chat", json={"message": "No vivíamos en Madrid."}
    )
    assert response.status_code == 200
    groups = (await app_client.get(f"{API}/admin/claim-groups")).json()
    assert [g["canonical_text"] for g in groups] == ["no viviamos en madrid"]
    return groups[0]


# --------------------------------------------------------------------------
# Resolution transitions
# --------------------------------------------------------------------------


async def test_admin_resolve_claim_group_marks_resolved(
    app_client: AsyncClient, seed: SeedUniverse
):
    group = await _flag_contradiction(app_client, seed)
    assert group["needs_review"] is True
    assert group["status"] == "active"

    response = await app_client.patch(
        f"{API}/admin/claim-groups/{group['id']}",
        json={
            "status": "resolved",
            "resolution_note": "Falso recuerdo; vivían en Guardamar.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "resolved"
    assert body["needs_review"] is False
    assert body["resolved_at"] is not None
    assert body["resolution_note"] == "Falso recuerdo; vivían en Guardamar."
    assert len(body["claims"]) == 1
    assert len(body["contradictions"]) == 1


async def test_admin_ignore_claim_group_marks_ignored(
    app_client: AsyncClient, seed: SeedUniverse
):
    group = await _flag_contradiction(app_client, seed)

    response = await app_client.patch(
        f"{API}/admin/claim-groups/{group['id']}",
        json={"status": "ignored"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ignored"
    assert body["needs_review"] is False
    assert body["resolved_at"] is not None


async def test_admin_reopen_claim_group_clears_resolution_stamp(
    app_client: AsyncClient, seed: SeedUniverse
):
    group = await _flag_contradiction(app_client, seed)
    await app_client.patch(
        f"{API}/admin/claim-groups/{group['id']}", json={"status": "resolved"}
    )

    response = await app_client.patch(
        f"{API}/admin/claim-groups/{group['id']}", json={"status": "active"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "active"
    assert body["resolved_at"] is None


async def test_admin_resolve_claim_group_404_for_unknown(
    app_client: AsyncClient,
):
    response = await app_client.patch(
        f"{API}/admin/claim-groups/{uuid.uuid4()}", json={"status": "resolved"}
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------
# No re-flagging for manually resolved/ignored groups
# --------------------------------------------------------------------------


async def test_resolved_group_is_not_reflagged_by_new_contradicting_claim(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    group = await _flag_contradiction(app_client, seed)
    await app_client.patch(
        f"{API}/admin/claim-groups/{group['id']}", json={"status": "resolved"}
    )

    response = await app_client.post(
        f"{API}/chat", json={"message": "No vivíamos en Madrid."}
    )
    assert response.status_code == 200

    groups = (await app_client.get(f"{API}/admin/claim-groups")).json()
    assert len(groups) == 1
    assert groups[0]["status"] == "resolved"
    assert groups[0]["needs_review"] is False

    signals = (
        (await db_session.execute(select(ClaimGroupContradiction)))
        .scalars()
        .all()
    )
    assert len(signals) == 1
    assert signals[0].detection_count == 1

    claims = (await db_session.execute(select(Claim))).scalars().all()
    assert len(claims) == 2
    assert all(claim.status == ClaimStatus.candidate for claim in claims)


async def test_ignored_group_is_not_reflagged_by_new_contradicting_claim(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    group = await _flag_contradiction(app_client, seed)
    await app_client.patch(
        f"{API}/admin/claim-groups/{group['id']}", json={"status": "ignored"}
    )

    await app_client.post(f"{API}/chat", json={"message": "No vivíamos en Madrid."})

    groups = (await app_client.get(f"{API}/admin/claim-groups")).json()
    assert groups[0]["status"] == "ignored"
    assert groups[0]["needs_review"] is False

    signals = (
        (await db_session.execute(select(ClaimGroupContradiction)))
        .scalars()
        .all()
    )
    assert len(signals) == 1
    assert signals[0].detection_count == 1


# --------------------------------------------------------------------------
# Filtering and admin reads
# --------------------------------------------------------------------------


async def test_admin_claim_groups_filter_by_status(
    app_client: AsyncClient, seed: SeedUniverse
):
    flagged = await _flag_contradiction(app_client, seed)
    await app_client.post(f"{API}/chat", json={"message": "Teníamos un perro."})
    await app_client.patch(
        f"{API}/admin/claim-groups/{flagged['id']}", json={"status": "resolved"}
    )

    resolved = await app_client.get(
        f"{API}/admin/claim-groups", params={"status": "resolved"}
    )
    assert resolved.status_code == 200
    assert [g["canonical_text"] for g in resolved.json()] == ["no viviamos en madrid"]

    active = await app_client.get(
        f"{API}/admin/claim-groups", params={"status": "active"}
    )
    assert [g["canonical_text"] for g in active.json()] == ["teniamos un perro"]

    ignored = await app_client.get(
        f"{API}/admin/claim-groups", params={"status": "ignored"}
    )
    assert ignored.json() == []


async def test_admin_contradictions_listing_shows_group_status(
    app_client: AsyncClient, seed: SeedUniverse
):
    group = await _flag_contradiction(app_client, seed)
    await app_client.patch(
        f"{API}/admin/claim-groups/{group['id']}", json={"status": "resolved"}
    )

    listing = await app_client.get(f"{API}/admin/claim-contradictions")
    assert listing.status_code == 200
    item = listing.json()[0]
    assert item["group"]["status"] == "resolved"
    assert item["group"]["needs_review"] is False


# --------------------------------------------------------------------------
# No-promotion guarantees
# --------------------------------------------------------------------------


async def test_resolution_never_promotes_claims(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    fact_count_before = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()
    memory_count_before = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()

    group = await _flag_contradiction(app_client, seed)
    await app_client.patch(
        f"{API}/admin/claim-groups/{group['id']}", json={"status": "resolved"}
    )

    fact_count = (
        await db_session.execute(select(func.count()).select_from(Fact))
    ).scalar_one()
    memory_count = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()
    assert fact_count == fact_count_before + 1
    assert memory_count == memory_count_before

    claims = (await db_session.execute(select(Claim))).scalars().all()
    assert all(claim.status == ClaimStatus.candidate for claim in claims)
