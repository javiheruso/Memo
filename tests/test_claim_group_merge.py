"""Explicit human canonical merge of duplicate claim groups.

Covers the merge transaction (members move unchanged, contradiction evidence
transfers/folds, remembered keys re-point, the absorbed group is removed),
remembered-key routing for future claims, deterministic suggestions, the guards
(self-merge, unknown/ignored/inactive/materialized groups, opposite-polarity
confirmation) and the review-context lift (source message in group detail).
"""

import uuid

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.base import ClaimCandidate
from app.domain.enums import ClaimStatus
from app.models import (
    Claim,
    ClaimGroup,
    ClaimGroupContradiction,
    ClaimGroupMember,
    ClaimGroupRememberedKey,
)
from app.repositories.claim import ClaimRepository
from app.repositories.claim_group import ClaimGroupRepository
from app.repositories.conversation import ConversationRepository
from tests.fixtures.universe import SeedUniverse

API = "/api/v1/admin"
CHAT = "/api/v1/chat"


async def _chat(app_client: AsyncClient, text: str) -> None:
    response = await app_client.post(CHAT, json={"message": text})
    assert response.status_code == 200


async def _groups(app_client: AsyncClient) -> list[dict]:
    response = await app_client.get(f"{API}/claim-groups")
    assert response.status_code == 200
    return response.json()


async def _group_by_text(app_client: AsyncClient, canonical: str) -> dict:
    for group in await _groups(app_client):
        if group["canonical_text"] == canonical:
            return group
    raise AssertionError(f"no claim group with canonical text {canonical!r}")


async def _verified_fact(
    app_client: AsyncClient, seed: SeedUniverse, *, value_text: str
) -> dict:
    response = await app_client.post(
        f"{API}/facts",
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


async def _materialize_fact(
    app_client: AsyncClient, group: dict, seed: SeedUniverse
) -> None:
    response = await app_client.post(
        f"{API}/claim-groups/{group['id']}/materialize-fact",
        json={
            "subject_person_id": str(seed.person_ids["carmen"]),
            "fact_type": "residence",
            "source_id": str(seed.source_ids["shared_by_carmen"]),
        },
    )
    assert response.status_code == 201


async def _merge(
    app_client: AsyncClient, source_id: str, target_id: str, *, confirm: bool = False
):
    return await app_client.post(
        f"{API}/claim-groups/{source_id}/merge",
        json={"target_group_id": target_id, "confirm_opposite_polarity": confirm},
    )


# --------------------------------------------------------------------------
# Merge transaction
# --------------------------------------------------------------------------


async def test_merge_folds_source_into_survivor_and_remembers_key(
    app_client: AsyncClient, db_session: AsyncSession
):
    await _chat(app_client, "Vivíamos en Madrid.")
    await _chat(app_client, "Recuerdo que vivíamos en Madrid.")
    survivor = await _group_by_text(app_client, "viviamos en madrid")
    source = await _group_by_text(app_client, "recuerdo que viviamos en madrid")
    assert survivor["id"] != source["id"]

    response = await _merge(app_client, source["id"], survivor["id"])
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == survivor["id"]
    assert body["total_mentions"] == 2
    assert {claim["original_text"] for claim in body["claims"]} == {
        "Vivíamos en Madrid",
        "Recuerdo que vivíamos en Madrid",
    }

    remaining = await _groups(app_client)
    assert [g["id"] for g in remaining] == [survivor["id"]]

    remembered = (
        await db_session.execute(select(ClaimGroupRememberedKey))
    ).scalars().all()
    assert [(row.canonical_text, str(row.claim_group_id)) for row in remembered] == [
        ("recuerdo que viviamos en madrid", survivor["id"])
    ]


async def test_future_claim_using_absorbed_wording_routes_to_survivor(
    app_client: AsyncClient, db_session: AsyncSession
):
    await _chat(app_client, "Vivíamos en Madrid.")
    await _chat(app_client, "Recuerdo que vivíamos en Madrid.")
    survivor = await _group_by_text(app_client, "viviamos en madrid")
    source = await _group_by_text(app_client, "recuerdo que viviamos en madrid")
    await _merge(app_client, source["id"], survivor["id"])

    await _chat(app_client, "Recuerdo que vivíamos en Madrid.")

    remaining = await _groups(app_client)
    assert [g["id"] for g in remaining] == [survivor["id"]]

    detail = await app_client.get(f"{API}/claim-groups/{survivor['id']}")
    assert detail.status_code == 200
    assert detail.json()["total_mentions"] == 3

    claims = (await db_session.execute(select(Claim))).scalars().all()
    assert len(claims) == 3
    assert all(claim.status == ClaimStatus.candidate for claim in claims)
    member_rows = (
        (await db_session.execute(select(ClaimGroupMember))).scalars().all()
    )
    assert {row.claim_group_id for row in member_rows} == {
        uuid.UUID(survivor["id"])
    }


async def test_merge_moves_claims_without_mutating_them(
    app_client: AsyncClient, db_session: AsyncSession
):
    await _chat(app_client, "Vivíamos en Madrid.")
    await _chat(app_client, "Recuerdo que vivíamos en Madrid.")
    survivor = await _group_by_text(app_client, "viviamos en madrid")
    source = await _group_by_text(app_client, "recuerdo que viviamos en madrid")

    await _merge(app_client, source["id"], survivor["id"])

    claims = (await db_session.execute(select(Claim))).scalars().all()
    assert {claim.original_text for claim in claims} == {
        "Vivíamos en Madrid",
        "Recuerdo que vivíamos en Madrid",
    }
    assert all(claim.status == ClaimStatus.candidate for claim in claims)
    assert all(claim.conversation_id is not None for claim in claims)


async def test_merge_folds_contradiction_evidence_into_survivor(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
):
    await _verified_fact(app_client, seed, value_text="Vivíamos en Madrid")
    await _chat(app_client, "No vivíamos en Madrid.")
    await _chat(app_client, "Tampoco vivíamos en Madrid.")
    survivor = await _group_by_text(app_client, "no viviamos en madrid")
    source = await _group_by_text(app_client, "tampoco viviamos en madrid")
    assert survivor["needs_review"] is True
    assert source["needs_review"] is True

    response = await _merge(app_client, source["id"], survivor["id"])
    assert response.status_code == 200
    assert response.json()["needs_review"] is True

    signals = (
        (await db_session.execute(select(ClaimGroupContradiction)))
        .scalars()
        .all()
    )
    assert len(signals) == 1
    assert str(signals[0].claim_group_id) == survivor["id"]
    assert signals[0].detection_count == 2


async def test_chain_merge_repoints_remembered_keys(
    app_client: AsyncClient, db_session: AsyncSession
):
    await _chat(app_client, "Vivíamos en Madrid.")
    await _chat(app_client, "Recuerdo que vivíamos en Madrid.")
    await _chat(app_client, "Nosotros vivíamos en Madrid.")
    group_a = await _group_by_text(app_client, "viviamos en madrid")
    group_b = await _group_by_text(app_client, "recuerdo que viviamos en madrid")
    group_c = await _group_by_text(app_client, "nosotros viviamos en madrid")

    await _merge(app_client, group_c["id"], group_b["id"])
    await _merge(app_client, group_b["id"], group_a["id"])

    remaining = await _groups(app_client)
    assert [g["id"] for g in remaining] == [group_a["id"]]

    rows = (await db_session.execute(select(ClaimGroupRememberedKey))).scalars().all()
    targets = {row.canonical_text: str(row.claim_group_id) for row in rows}
    assert targets == {
        "nosotros viviamos en madrid": group_a["id"],
        "recuerdo que viviamos en madrid": group_a["id"],
    }


# --------------------------------------------------------------------------
# Deterministic suggestions
# --------------------------------------------------------------------------


async def test_merge_suggestions_ranked_by_shared_content_tokens(
    app_client: AsyncClient,
):
    await _chat(app_client, "Vivíamos en Madrid.")
    await _chat(app_client, "Recuerdo que vivíamos en Madrid.")
    await _chat(app_client, "Teníamos un perro.")
    source = await _group_by_text(app_client, "recuerdo que viviamos en madrid")

    response = await app_client.get(
        f"{API}/claim-groups/{source['id']}/merge-suggestions"
    )
    assert response.status_code == 200
    suggestions = response.json()
    assert len(suggestions) == 2
    by_text = {item["canonical_text"]: item for item in suggestions}
    assert by_text["viviamos en madrid"]["shared_terms"] > by_text[
        "teniamos un perro"
    ]["shared_terms"]
    assert suggestions[0]["canonical_text"] == "viviamos en madrid"
    assert by_text["viviamos en madrid"]["opposite_polarity"] is False


async def test_merge_suggestions_404_for_unknown_group(app_client: AsyncClient):
    response = await app_client.get(
        f"{API}/claim-groups/{uuid.uuid4()}/merge-suggestions"
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


async def test_merge_self_merge_is_conflict(app_client: AsyncClient):
    await _chat(app_client, "Vivíamos en Madrid.")
    group = await _group_by_text(app_client, "viviamos en madrid")

    response = await _merge(app_client, group["id"], group["id"])
    assert response.status_code == 409
    assert "itself" in response.json()["detail"]


async def test_merge_unknown_groups_are_404(app_client: AsyncClient):
    await _chat(app_client, "Vivíamos en Madrid.")
    group = await _group_by_text(app_client, "viviamos en madrid")

    missing_source = await _merge(app_client, str(uuid.uuid4()), group["id"])
    assert missing_source.status_code == 404

    missing_target = await _merge(app_client, group["id"], str(uuid.uuid4()))
    assert missing_target.status_code == 404


async def test_merge_ignored_source_is_conflict(app_client: AsyncClient):
    await _chat(app_client, "Vivíamos en Madrid.")
    await _chat(app_client, "Recuerdo que vivíamos en Madrid.")
    survivor = await _group_by_text(app_client, "viviamos en madrid")
    source = await _group_by_text(app_client, "recuerdo que viviamos en madrid")
    await app_client.patch(
        f"{API}/claim-groups/{source['id']}", json={"status": "ignored"}
    )

    response = await _merge(app_client, source["id"], survivor["id"])
    assert response.status_code == 409
    assert "ignored" in response.json()["detail"]


async def test_merge_ignored_target_is_conflict(app_client: AsyncClient):
    await _chat(app_client, "Vivíamos en Madrid.")
    await _chat(app_client, "Recuerdo que vivíamos en Madrid.")
    survivor = await _group_by_text(app_client, "viviamos en madrid")
    source = await _group_by_text(app_client, "recuerdo que viviamos en madrid")
    await app_client.patch(
        f"{API}/claim-groups/{survivor['id']}", json={"status": "ignored"}
    )

    response = await _merge(app_client, source["id"], survivor["id"])
    assert response.status_code == 409
    assert "ignored" in response.json()["detail"]


async def test_merge_resolved_source_is_conflict(app_client: AsyncClient):
    await _chat(app_client, "Vivíamos en Madrid.")
    await _chat(app_client, "Recuerdo que vivíamos en Madrid.")
    survivor = await _group_by_text(app_client, "viviamos en madrid")
    source = await _group_by_text(app_client, "recuerdo que viviamos en madrid")
    await app_client.patch(
        f"{API}/claim-groups/{source['id']}", json={"status": "resolved"}
    )

    response = await _merge(app_client, source["id"], survivor["id"])
    assert response.status_code == 409
    assert "only active" in response.json()["detail"]


async def test_merge_materialized_source_is_conflict(
    app_client: AsyncClient, seed: SeedUniverse
):
    await _chat(app_client, "Vivíamos en Madrid.")
    await _chat(app_client, "Recuerdo que vivíamos en Madrid.")
    survivor = await _group_by_text(app_client, "viviamos en madrid")
    source = await _group_by_text(app_client, "recuerdo que viviamos en madrid")

    await _materialize_fact(app_client, source, seed)
    reopened = await app_client.patch(
        f"{API}/claim-groups/{source['id']}", json={"status": "active"}
    )
    assert reopened.status_code == 200

    response = await _merge(app_client, source["id"], survivor["id"])
    assert response.status_code == 409
    assert "materialization" in response.json()["detail"]


async def test_opposite_polarity_merge_requires_confirmation(
    app_client: AsyncClient,
):
    await _chat(app_client, "Vivíamos en Madrid.")
    await _chat(app_client, "No vivíamos en Madrid.")
    survivor = await _group_by_text(app_client, "viviamos en madrid")
    source = await _group_by_text(app_client, "no viviamos en madrid")

    refused = await _merge(app_client, source["id"], survivor["id"])
    assert refused.status_code == 409
    assert "confirm_opposite_polarity" in refused.json()["detail"]

    accepted = await _merge(
        app_client, source["id"], survivor["id"], confirm=True
    )
    assert accepted.status_code == 200
    assert accepted.json()["id"] == survivor["id"]
    assert accepted.json()["total_mentions"] == 2


async def test_same_polarity_negated_merge_needs_no_confirmation(
    app_client: AsyncClient,
):
    await _chat(app_client, "No vivíamos en Madrid.")
    await _chat(app_client, "Tampoco vivíamos en Madrid.")
    survivor = await _group_by_text(app_client, "no viviamos en madrid")
    source = await _group_by_text(app_client, "tampoco viviamos en madrid")

    response = await _merge(app_client, source["id"], survivor["id"])
    assert response.status_code == 200


# --------------------------------------------------------------------------
# Remembered-key resolution at the repository level
# --------------------------------------------------------------------------


async def test_attach_resolves_remembered_key_to_survivor(
    db_session: AsyncSession,
):
    conversations = ConversationRepository(db_session)
    conversation = await conversations.get_or_create(None)
    message = await conversations.add_message(
        conversation.id, speaker="user", text="Vivíamos en Madrid."
    )
    repo = ClaimRepository(db_session)
    group_repo = ClaimGroupRepository(db_session)

    claim = (
        await repo.add_many(
            [ClaimCandidate(original_text="Vivíamos en Madrid.")],
            conversation_id=conversation.id,
            message_id=message.id,
        )
    )[0]
    survivor = await group_repo.attach(claim, "viviamos en madrid")

    db_session.add(
        ClaimGroupRememberedKey(
            canonical_text="recuerdo que viviamos en madrid",
            claim_group_id=survivor.id,
        )
    )
    await db_session.commit()

    second_message = await conversations.add_message(
        conversation.id, speaker="user", text="Recuerdo que vivíamos en Madrid."
    )
    absorbed = (
        await repo.add_many(
            [ClaimCandidate(original_text="Recuerdo que vivíamos en Madrid.")],
            conversation_id=conversation.id,
            message_id=second_message.id,
        )
    )[0]
    routed = await group_repo.attach(absorbed, "recuerdo que viviamos en madrid")
    await db_session.commit()

    assert routed.id == survivor.id
    groups = (await db_session.execute(select(ClaimGroup))).scalars().all()
    assert [group.id for group in groups] == [survivor.id]
    assert groups[0].total_mentions == 2


# --------------------------------------------------------------------------
# Review-context lift (source message in group detail)
# --------------------------------------------------------------------------


async def test_claim_group_detail_exposes_member_message(
    app_client: AsyncClient,
):
    await _chat(app_client, "Vivíamos en Madrid. Teníamos un perro.")
    survivor = await _group_by_text(app_client, "viviamos en madrid")

    response = await app_client.get(f"{API}/claim-groups/{survivor['id']}")
    assert response.status_code == 200
    body = response.json()
    assert len(body["claims"]) == 1
    claim = body["claims"][0]
    assert claim["original_text"] == "Vivíamos en Madrid"
    assert claim["message"]["text"] == "Vivíamos en Madrid. Teníamos un perro."
    assert claim["message"]["speaker"] == "user"
    assert claim["message"]["id"] == claim["message_id"]
