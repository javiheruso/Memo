"""Server-rendered admin UI: pages render and form flows work end to end.

Every write path goes through the same pydantic schemas and services as the
JSON admin API, so these tests assert the HTML surface renders and that the
forms drive the existing repositories/services (status transitions,
materialization with provenance, entity create) exactly as the API would.
"""

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models import (
    ClaimGroup,
    Fact,
    KnowledgeGap,
    Materialization,
    Media,
    Memory,
    Person,
    Source,
)
from app.repositories.conversation import ConversationRepository
from app.repositories.knowledge_gap import KnowledgeGapRepository
from tests.fixtures.universe import SeedUniverse

UI = "/admin"
API = "/api/v1/admin"

_SKIP_ON_POSTGRES = pytest.mark.skipif(
    settings.test_database_url is not None,
    reason="relies on SQLite backfill behavior; covered by postgres smoke tests",
)


async def _claim_group(app_client: AsyncClient, *, text: str = "Vivíamos en Madrid.") -> dict:
    response = await app_client.post("/api/v1/chat", json={"message": text})
    assert response.status_code == 200
    groups = (await app_client.get(f"{API}/claim-groups")).json()
    assert groups
    return groups[0]


async def _gap(db_session: AsyncSession, *, reference: str = "eustaquio") -> KnowledgeGap:
    conversation = await ConversationRepository(db_session).get_or_create(None)
    gap, _ = await KnowledgeGapRepository(db_session).upsert(
        gap_type="person_unknown",
        canonical_reference=reference,
        conversation_id=conversation.id,
        description=f"User asked about unknown person '{reference}'.",
    )
    return gap


# --------------------------------------------------------------------------
# Pages render
# --------------------------------------------------------------------------


async def test_dashboard_renders(app_client: AsyncClient) -> None:
    response = await app_client.get(UI)
    assert response.status_code == 200
    assert "Memo Admin" in response.text
    assert "Claim groups" in response.text
    assert "Knowledge gaps" in response.text


async def test_claim_groups_page_lists_groups(app_client: AsyncClient) -> None:
    await _claim_group(app_client, text="Teníamos un perro.")
    response = await app_client.get(f"{UI}/claim-groups")
    assert response.status_code == 200
    assert "teniamos un perro" in response.text

    filtered = await app_client.get(f"{UI}/claim-groups", params={"status": "active"})
    assert filtered.status_code == 200
    assert "teniamos un perro" in filtered.text


async def test_claim_group_detail_renders_evidence_and_forms(
    app_client: AsyncClient,
) -> None:
    group = await _claim_group(app_client)
    response = await app_client.get(f"{UI}/claim-groups/{group['id']}")
    assert response.status_code == 200
    assert group["canonical_text"] in response.text
    assert "Vivíamos en Madrid" in response.text
    assert "Materialize as fact" in response.text
    assert "Materialize as memory" in response.text


async def test_claim_group_detail_404(app_client: AsyncClient) -> None:
    response = await app_client.get(f"{UI}/claim-groups/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_knowledge_gaps_page_lists_gaps(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    await _gap(db_session, reference="eustaquio")
    response = await app_client.get(f"{UI}/knowledge-gaps")
    assert response.status_code == 200
    assert "eustaquio" in response.text


async def test_knowledge_gap_detail_renders_forms(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    gap = await _gap(db_session)
    response = await app_client.get(f"{UI}/knowledge-gaps/{gap.id}")
    assert response.status_code == 200
    assert gap.canonical_reference in response.text
    assert "Materialize as fact" in response.text
    assert "Materialize as memory" in response.text


async def test_claim_groups_status_dropdown_has_all_options(
    app_client: AsyncClient,
) -> None:
    response = await app_client.get(f"{UI}/claim-groups")
    assert response.status_code == 200
    assert 'value="active"' in response.text
    assert 'value="resolved"' in response.text
    assert 'value="ignored"' in response.text


async def test_knowledge_gaps_status_dropdown_has_all_options(
    app_client: AsyncClient,
) -> None:
    response = await app_client.get(f"{UI}/knowledge-gaps")
    assert response.status_code == 200
    for status in ("observed", "recurrent", "family_review", "resolved", "ignored"):
        assert f'value="{status}"' in response.text


async def test_claim_group_detail_shows_message_context_and_verbatim_default(
    app_client: AsyncClient,
) -> None:
    response = await app_client.post(
        "/api/v1/chat",
        json={"message": "Vivíamos en Madrid. Teníamos un perro."},
    )
    assert response.status_code == 200
    groups = (await app_client.get(f"{API}/claim-groups")).json()
    group = next(g for g in groups if g["canonical_text"] == "viviamos en madrid")

    page = await app_client.get(f"{UI}/claim-groups/{group['id']}")
    assert page.status_code == 200
    assert "Vivíamos en Madrid. Teníamos un perro." in page.text
    assert 'value="Vivíamos en Madrid"' in page.text
    assert "Merge into another group" in page.text


async def test_claim_group_merge_form_flow(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    await app_client.post("/api/v1/chat", json={"message": "Vivíamos en Madrid."})
    await app_client.post(
        "/api/v1/chat", json={"message": "Recuerdo que vivíamos en Madrid."}
    )
    groups = (await app_client.get(f"{API}/claim-groups")).json()
    source = next(g for g in groups if g["canonical_text"] == "recuerdo que viviamos en madrid")
    survivor = next(g for g in groups if g["canonical_text"] == "viviamos en madrid")

    page = await app_client.get(f"{UI}/claim-groups/{source['id']}")
    assert page.status_code == 200
    assert "Merge into another group" in page.text
    assert f'value="{survivor["id"]}"' in page.text

    merged = await app_client.post(
        f"{UI}/claim-groups/{source['id']}/merge",
        data={"target_group_id": survivor["id"]},
        follow_redirects=True,
    )
    assert merged.status_code == 200
    assert "Merged claim group" in merged.text

    remaining = (await app_client.get(f"{API}/claim-groups")).json()
    assert [g["id"] for g in remaining] == [survivor["id"]]

    stored = (
        await db_session.execute(select(ClaimGroup).where(ClaimGroup.id == uuid.UUID(survivor["id"])))
    ).scalar_one()
    assert stored.total_mentions == 2
    assert stored.status.value == "active"


async def test_entity_pages_render(app_client: AsyncClient) -> None:
    for path in ("/sources", "/persons", "/media", "/facts", "/memories", "/embeddings"):
        response = await app_client.get(f"{UI}{path}")
        assert response.status_code == 200, path


# --------------------------------------------------------------------------
# Claim-group review flows
# --------------------------------------------------------------------------


async def test_claim_group_resolve_form_flow(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    group = await _claim_group(app_client)
    response = await app_client.post(
        f"{UI}/claim-groups/{group['id']}/status",
        data={"status": "resolved", "resolution_note": "revisado"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "revisado" in response.text

    stored = (
        await db_session.execute(
            select(ClaimGroup).where(ClaimGroup.id == uuid.UUID(group["id"]))
        )
    ).scalar_one()
    assert stored.status.value == "resolved"
    assert stored.needs_review is False
    assert stored.resolution_note == "revisado"


async def test_claim_group_ignore_form_flow(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    group = await _claim_group(app_client)
    response = await app_client.post(
        f"{UI}/claim-groups/{group['id']}/status",
        data={"status": "ignored"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "marked as ignored" in response.text
    stored = (
        await db_session.execute(
            select(ClaimGroup).where(ClaimGroup.id == uuid.UUID(group["id"]))
        )
    ).scalar_one()
    assert stored.status.value == "ignored"


async def test_claim_group_materialize_fact_form_flow(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
) -> None:
    group = await _claim_group(app_client)
    response = await app_client.post(
        f"{UI}/claim-groups/{group['id']}/materialize-fact",
        data={
            "subject_person_id": str(seed.person_ids["carmen"]),
            "fact_type": "residence",
            "source_id": str(seed.source_ids["shared_by_carmen"]),
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "Materialized as fact" in response.text

    provenance = list((await db_session.execute(select(Materialization))).scalars())
    assert len(provenance) == 1
    assert str(provenance[0].source_id) == group["id"]
    assert provenance[0].target_type.value == "fact"

    stored_group = (
        await db_session.execute(
            select(ClaimGroup).where(ClaimGroup.id == uuid.UUID(group["id"]))
        )
    ).scalar_one()
    assert stored_group.status.value == "resolved"


async def test_claim_group_materialize_memory_form_flow(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
) -> None:
    group = await _claim_group(app_client)
    response = await app_client.post(
        f"{UI}/claim-groups/{group['id']}/materialize-memory",
        data={
            "narrative": "Recuerdo que teníamos un perro.",
            "source_id": str(seed.source_ids["shared_by_carmen"]),
            "person_id": str(seed.person_ids["carmen"]),
            "person_role": "owner",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "Materialized as memory" in response.text

    memories = list(
        (
            await db_session.execute(
                select(Memory)
                .where(Memory.narrative == "Recuerdo que teníamos un perro.")
                .options(selectinload(Memory.people))
            )
        ).scalars()
    )
    assert len(memories) == 1
    assert memories[0].people[0].person_id == seed.person_ids["carmen"]


async def test_claim_group_materialize_duplicate_is_refused(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    group = await _claim_group(app_client)
    payload = {
        "subject_person_id": str(seed.person_ids["carmen"]),
        "fact_type": "residence",
        "source_id": str(seed.source_ids["shared_by_carmen"]),
    }
    first = await app_client.post(
        f"{UI}/claim-groups/{group['id']}/materialize-fact", data=payload
    )
    assert first.status_code == 303

    second = await app_client.post(
        f"{UI}/claim-groups/{group['id']}/materialize-fact",
        data=payload,
        follow_redirects=True,
    )
    assert second.status_code == 200
    assert "already materialized" in second.text


async def test_claim_group_materialize_ignored_is_refused(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    group = await _claim_group(app_client)
    await app_client.post(
        f"{UI}/claim-groups/{group['id']}/status", data={"status": "ignored"}
    )
    response = await app_client.post(
        f"{UI}/claim-groups/{group['id']}/materialize-fact",
        data={
            "subject_person_id": str(seed.person_ids["carmen"]),
            "fact_type": "residence",
            "source_id": str(seed.source_ids["shared_by_carmen"]),
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "ignored" in response.text


# --------------------------------------------------------------------------
# Knowledge-gap review flows
# --------------------------------------------------------------------------


async def test_knowledge_gap_resolve_form_flow(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
) -> None:
    gap = await _gap(db_session)
    response = await app_client.post(
        f"{UI}/knowledge-gaps/{gap.id}/status",
        data={
            "status": "resolved",
            "resolved_person_id": str(seed.person_ids["maria"]),
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "marked as resolved" in response.text

    await db_session.refresh(gap)
    assert gap.status.value == "resolved"
    assert str(gap.resolved_person_id) == str(seed.person_ids["maria"])


async def test_knowledge_gap_materialize_memory_form_flow(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
) -> None:
    gap = await _gap(db_session)
    response = await app_client.post(
        f"{UI}/knowledge-gaps/{gap.id}/materialize-memory",
        data={
            "narrative": "Eustaquio was a neighbor.",
            "source_id": str(seed.source_ids["shared_by_carmen"]),
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "Materialized as memory" in response.text

    provenance = list((await db_session.execute(select(Materialization))).scalars())
    assert len(provenance) == 1
    assert str(provenance[0].source_id) == str(gap.id)
    assert provenance[0].target_type.value == "memory"

    await db_session.refresh(gap)
    assert gap.status.value == "resolved"


# --------------------------------------------------------------------------
# Core entity create flows
# --------------------------------------------------------------------------


async def test_sources_create_form_flow(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    response = await app_client.post(
        f"{UI}/sources",
        data={"source_type": "photo", "description": "album"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "created" in response.text
    sources = list((await db_session.execute(select(Source))).scalars())
    assert len(sources) == 1
    assert sources[0].source_type.value == "photo"


async def test_persons_create_form_flow(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    response = await app_client.post(
        f"{UI}/persons",
        data={"first_name": "Carmen", "last_name": "Gómez"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "Carmen Gómez" in response.text
    persons = list((await db_session.execute(select(Person))).scalars())
    assert len(persons) == 1
    assert persons[0].display_name == "Carmen Gómez"


async def test_media_page_lists_and_create_form_flow(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    page = await app_client.get(f"{UI}/media")
    assert page.status_code == 200
    assert "Create media" in page.text

    response = await app_client.post(
        f"{UI}/media",
        data={
            "media_type": "photo",
            "storage_path": "albums/vera.jpg",
            "title": "Foto de Vera",
            "approx_date": "1962-03-08",
            "place_text": "Alicante",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "created" in response.text
    assert "Foto de Vera" in response.text

    media = list((await db_session.execute(select(Media))).scalars())
    assert len(media) == 1
    assert media[0].media_type == "photo"
    assert media[0].storage_path == "albums/vera.jpg"
    assert media[0].approx_date.isoformat() == "1962-03-08"
    assert media[0].place_text == "Alicante"


async def test_persons_page_sets_and_clears_primary_media(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
) -> None:
    page = await app_client.get(f"{UI}/persons")
    assert page.status_code == 200
    assert "Primary media" in page.text
    assert "Set primary" in page.text

    set_response = await app_client.post(
        f"{UI}/persons/{seed.person_ids['maria']}/primary-media",
        data={"media_id": str(seed.media_ids["luisa_photo"])},
        follow_redirects=True,
    )
    assert set_response.status_code == 200
    assert "Primary media updated" in set_response.text

    stored = (
        await db_session.execute(
            select(Person).where(Person.id == seed.person_ids["maria"])
        )
    ).scalar_one()
    await db_session.refresh(stored)
    assert stored.primary_media_id == seed.media_ids["luisa_photo"]

    clear_response = await app_client.post(
        f"{UI}/persons/{seed.person_ids['maria']}/primary-media",
        data={"media_id": ""},
        follow_redirects=True,
    )
    assert clear_response.status_code == 200
    assert "Primary media updated" in clear_response.text

    await db_session.refresh(stored)
    assert stored.primary_media_id is None


async def test_facts_create_form_defaults_to_pending(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
) -> None:
    response = await app_client.post(
        f"{UI}/facts",
        data={
            "subject_person_id": str(seed.person_ids["carmen"]),
            "fact_type": "hobby",
            "value_text": "Gardening",
            "source_id": str(seed.source_ids["shared_by_carmen"]),
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "created" in response.text
    facts = list(
        (
            await db_session.execute(
                select(Fact).where(Fact.value_text == "Gardening")
            )
        ).scalars()
    )
    assert len(facts) == 1
    assert facts[0].status.value == "pending"
    assert facts[0].value_text == "Gardening"


async def test_memories_create_form_flow(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
) -> None:
    response = await app_client.post(
        f"{UI}/memories",
        data={
            "title": "Gardening mornings",
            "narrative": "We gardened together every morning.",
            "source_id": str(seed.source_ids["shared_by_carmen"]),
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "created" in response.text
    memories = list(
        (
            await db_session.execute(
                select(Memory).where(Memory.title == "Gardening mornings")
            )
        ).scalars()
    )
    assert len(memories) == 1
    assert memories[0].memory_status.value == "candidate"


async def test_memories_create_form_rejects_unknown_source(
    app_client: AsyncClient, seed: SeedUniverse, db_session: AsyncSession
) -> None:
    count_before = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()
    response = await app_client.post(
        f"{UI}/memories",
        data={
            "narrative": "Orphan memory with a bogus source.",
            "source_id": str(uuid.uuid4()),
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "Database error" in response.text
    count_after = (
        await db_session.execute(select(func.count()).select_from(Memory))
    ).scalar_one()
    assert count_after == count_before


# --------------------------------------------------------------------------
# Embedding backfill
# --------------------------------------------------------------------------


@_SKIP_ON_POSTGRES
async def test_backfill_page_and_action_on_sqlite(
    app_client: AsyncClient, seed: SeedUniverse
) -> None:
    page = await app_client.get(f"{UI}/embeddings")
    assert page.status_code == 200
    assert "Backfill embeddings" in page.text

    response = await app_client.post(
        f"{UI}/embeddings/backfill", follow_redirects=True
    )
    assert response.status_code == 200
    assert "requires a PostgreSQL dialect" in response.text
