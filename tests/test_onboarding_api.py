import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import (
    ConversationProfile,
    Fact,
    Memory,
    OnboardingSession,
    Person,
    Relationship,
    Source,
)
from app.schemas.onboarding import OnboardingSource
from app.services.onboarding import (
    OnboardingDraftInvalidError,
    OnboardingService,
)

API = "/api/v1/admin"
ONBOARDING = f"{API}/onboarding"


async def test_onboarding_draft_is_saved_and_resumed(app_client: AsyncClient) -> None:
    created = await app_client.post(
        ONBOARDING,
        json={
            "identity": {
                "first_name": "Elena",
                "preferred_name": "Elenita",
                "language": "es-ES",
            },
            "reviewer": {"name": "Javi", "relationship": "hijo"},
        },
    )
    assert created.status_code == 201
    body = created.json()
    session_id = body["id"]
    assert body["status"] == "draft"
    assert body["payload"]["identity"]["first_name"] == "Elena"
    assert body["consent_confirmed"] is False

    resumed = await app_client.get(f"{ONBOARDING}/{session_id}")
    assert resumed.status_code == 200
    assert resumed.json()["payload"]["identity"]["preferred_name"] == "Elenita"

    saved = await app_client.put(
        f"{ONBOARDING}/{session_id}/draft",
        json={
            "identity": resumed.json()["payload"]["identity"],
            "reviewer": resumed.json()["payload"]["reviewer"],
            "contributors": [
                {
                    "name": "Javi",
                    "relationship": "hijo",
                    "permission_note": "Administer the profile",
                }
            ],
            "sources": [
                {
                    "key": "javi",
                    "source_type": "family_member",
                    "description": "Javi, son",
                }
            ],
            "sensitive_topics": [
                {
                    "topic": "fallecimiento de Miguel",
                    "handling": "Do not initiate; redirect gently.",
                    "triggers": ["murió"],
                }
            ],
        },
    )
    assert saved.status_code == 200
    payload = saved.json()["payload"]
    assert [item["name"] for item in payload["contributors"]] == ["Javi"]
    assert payload["sensitive_topics"][0]["triggers"] == ["murió"]

    listed = await app_client.get(ONBOARDING)
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [session_id]


async def test_onboarding_draft_rejects_duplicate_local_keys(
    app_client: AsyncClient,
) -> None:
    created = await app_client.post(ONBOARDING)
    session_id = created.json()["id"]

    response = await app_client.put(
        f"{ONBOARDING}/{session_id}/draft",
        json={
            "sources": [
                {"key": "family", "source_type": "family_member"},
                {"key": "family", "source_type": "document"},
            ]
        },
    )
    assert response.status_code == 422


async def test_save_draft_revalidates_in_place_mutations_before_persisting(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    """A parsed draft can be mutated past pydantic; the save path must recheck.

    This is the service-level guard behind the UI form flow, which appends
    entries directly to the already-validated draft lists.
    """
    created = await app_client.post(ONBOARDING)
    session_id = uuid.UUID(created.json()["id"])

    service = OnboardingService(db_session)
    row = await service.get(session_id)
    draft = service.draft_of(row)
    draft.sources.append(
        OnboardingSource(key="family", source_type="family_member")
    )
    draft.sources.append(OnboardingSource(key="family", source_type="document"))

    with pytest.raises(OnboardingDraftInvalidError) as excinfo:
        await service.save_draft(session_id, draft)
    assert any("sources keys must be unique" in item for item in excinfo.value.problems)

    stored = (await app_client.get(f"{ONBOARDING}/{session_id}")).json()
    assert stored["payload"]["sources"] == []
    assert (await app_client.get(ONBOARDING)).status_code == 200


async def test_onboarding_unknown_session_is_404(app_client: AsyncClient) -> None:
    response = await app_client.get(f"{ONBOARDING}/{uuid.uuid4()}")
    assert response.status_code == 404


def _first_day_package() -> dict:
    return {
        "identity": {
            "first_name": "Elena",
            "preferred_name": "Elenita",
            "language": "es-ES",
        },
        "reviewer": {"name": "Javi", "relationship": "hijo"},
        "contributors": [
            {
                "name": "Javi",
                "relationship": "hijo",
                "permission_note": "Administer the profile",
            }
        ],
        "sources": [
            {
                "key": "javi",
                "source_type": "family_member",
                "description": "Javi, son",
            },
            {
                "key": "notebook",
                "source_type": "document",
                "description": "Family notebook",
            },
        ],
        "people": [
            {
                "key": "miguel",
                "first_name": "Miguel",
                "relationship_type": "spouse_of",
                "certainty": "probable",
                "source_key": "javi",
                "notes": "Elena's husband",
            }
        ],
        "sensitive_topics": [
            {
                "topic": "fallecimiento de Miguel",
                "handling": "Do not initiate; redirect gently if asked.",
                "consent_review_required": True,
                "triggers": ["murió"],
            }
        ],
        "facts": [
            {
                "subject_key": "subject",
                "fact_type": "hobby",
                "value_text": "costura",
                "status": "verified",
                "source_key": "javi",
            }
        ],
        "memories": [
            {
                "title": "Tardes de costura",
                "narrative": "Elena cosía con Miguel los domingos.",
                "sensitivity_level": "low",
                "source_key": "notebook",
                "people_keys": ["subject", "miguel"],
            }
        ],
        "preferences": {
            "positive_topics": ["costura", "familia"],
            "never_do": ["No hablar de política"],
            "conversational_style": "Pausado",
        },
    }


async def _draft_session(client: AsyncClient, payload: dict) -> str:
    created = await client.post(ONBOARDING, json=payload)
    assert created.status_code == 201
    session_id = created.json()["id"]
    saved = await client.put(f"{ONBOARDING}/{session_id}/draft", json=payload)
    assert saved.status_code == 200
    return session_id


async def test_activation_requires_explicit_consent(app_client: AsyncClient) -> None:
    session_id = await _draft_session(app_client, _first_day_package())

    blocked = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert blocked.status_code == 409
    assert "consent" in blocked.json()["detail"].lower()
    assert (await app_client.get(f"{ONBOARDING}/{session_id}")).json()[
        "status"
    ] == "draft"

    consent = await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={
            "confirmed_by": "Javi",
            "scope": "Activate Elena's conversational profile",
        },
    )
    assert consent.status_code == 200
    assert consent.json()["consent_confirmed"] is True
    assert consent.json()["status"] == "draft"

    activated = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert activated.status_code == 200
    body = activated.json()
    assert body["status"] == "activated"
    assert body["activated_at"] is not None
    assert body["subject_person_id"] is not None
    assert body["profile_id"] is not None


async def test_editing_draft_after_consent_invalidates_consent(
    app_client: AsyncClient,
) -> None:
    session_id = await _draft_session(app_client, _first_day_package())
    consent = await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={"confirmed_by": "Javi", "scope": "Activate profile"},
    )
    assert consent.status_code == 200
    assert consent.json()["consent_confirmed"] is True

    changed = _first_day_package()
    changed["identity"]["preferred_name"] = "Elena"
    saved = await app_client.put(
        f"{ONBOARDING}/{session_id}/draft", json=changed
    )
    assert saved.status_code == 200
    body = saved.json()
    assert body["consent_confirmed"] is False
    assert body["consent_confirmed_by"] is None
    assert body["consent_confirmed_at"] is None
    assert body["consent_scope"] is None

    blocked = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert blocked.status_code == 409
    assert "consent" in blocked.json()["detail"].lower()

    reconsent = await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={"confirmed_by": "Javi", "scope": "Activate the revised profile"},
    )
    assert reconsent.status_code == 200
    assert (
        await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    ).status_code == 200


async def test_activation_rejects_consent_for_a_different_payload(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    session_id = await _draft_session(app_client, _first_day_package())
    await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={"confirmed_by": "Javi", "scope": "Activate profile"},
    )

    row = await db_session.get(OnboardingSession, uuid.UUID(session_id))
    assert row is not None
    mutated = dict(row.payload)
    mutated["identity"] = {**mutated["identity"], "preferred_name": "Cambiada"}
    row.payload = mutated
    await db_session.commit()

    response = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert response.status_code == 409
    detail = response.json()["detail"].lower()
    assert "consent" in detail
    assert "draft" in detail
    stored = (await app_client.get(f"{ONBOARDING}/{session_id}")).json()
    assert stored["status"] == "draft"
    assert stored["subject_person_id"] is None


async def test_activation_rejects_duplicate_sensitive_topics(
    app_client: AsyncClient,
) -> None:
    payload = _first_day_package()
    payload["sensitive_topics"] = [
        {
            "topic": "Fallecimiento de Miguel",
            "handling": "Do not initiate.",
            "triggers": ["murió"],
        },
        {
            "topic": "  fallecimiento de MIGUEL  ",
            "handling": "Redirect gently.",
            "triggers": ["funeral"],
        },
    ]
    session_id = await _draft_session(app_client, payload)
    await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={"confirmed_by": "Javi", "scope": "Activate profile"},
    )

    response = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert response.status_code == 422
    problems = response.json()["detail"]
    assert any(
        "unique" in problem and "case-insensitive" in problem
        for problem in problems
    )
    stored = (await app_client.get(f"{ONBOARDING}/{session_id}")).json()
    assert stored["status"] == "draft"
    assert stored["subject_person_id"] is None


async def test_double_activation_materializes_the_package_once(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    session_id = await _draft_session(app_client, _first_day_package())
    await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={"confirmed_by": "Javi", "scope": "Activate profile"},
    )

    first = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert first.status_code == 200
    second = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert second.status_code == 409

    persons = (await db_session.execute(select(Person))).scalars().all()
    profiles = (
        await db_session.execute(select(ConversationProfile))
    ).scalars().all()
    assert len(persons) == 2
    assert len(profiles) == 1


async def test_activation_rejects_an_incomplete_package(
    app_client: AsyncClient,
) -> None:
    created = await app_client.post(ONBOARDING)
    session_id = created.json()["id"]
    await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={"confirmed_by": "Javi", "scope": "Activate profile"},
    )

    response = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert response.status_code == 422
    problems = response.json()["detail"]
    assert "identity.first_name is required" in problems
    assert "reviewer.name is required" in problems
    assert "at least one permitted contributor is required" in problems


async def test_activation_materializes_provenance_and_active_profile_behavior(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    session_id = await _draft_session(app_client, _first_day_package())
    await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={"confirmed_by": "Javi", "scope": "Activate profile"},
    )
    response = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert response.status_code == 200
    body = response.json()

    subject = await db_session.get(Person, uuid.UUID(body["subject_person_id"]))
    assert subject is not None
    assert subject.display_name == "Elena"

    facts = (
        await db_session.execute(
            select(Fact).where(Fact.subject_person_id == subject.id)
        )
    ).scalars().all()
    assert [(fact.fact_type, fact.value_text, fact.status.value) for fact in facts] == [
        ("hobby", "costura", "verified")
    ]
    source = await db_session.get(Source, facts[0].source_id)
    assert source is not None
    assert source.source_type.value == "family_member"
    assert source.external_reference.startswith(f"onboarding/{session_id}/")

    memories = (
        await db_session.execute(select(Memory).options(selectinload(Memory.people)))
    ).scalars().all()
    assert len(memories) == 1
    assert memories[0].memory_status.value == "candidate"
    assert memories[0].sensitivity_level.value == "low"
    miguel = (
        await db_session.execute(
            select(Person).where(Person.first_name == "Miguel")
        )
    ).scalar_one()
    assert {link.person_id for link in memories[0].people} == {
        subject.id,
        miguel.id,
    }

    relationships = (
        await db_session.execute(select(Relationship))
    ).scalars().all()
    assert len(relationships) == 1
    assert relationships[0].relationship_type.value == "spouse_of"
    assert relationships[0].status.value == "probable"
    assert relationships[0].source_id is not None

    profile = (
        await db_session.execute(
            select(ConversationProfile)
            .where(ConversationProfile.id == uuid.UUID(body["profile_id"]))
            .options(selectinload(ConversationProfile.sensitive_topics))
        )
    ).scalar_one()
    assert profile is not None
    assert profile.is_active is True
    assert profile.preferred_name == "Elenita"
    assert profile.positive_topics == ["costura", "familia"]
    assert [topic.topic for topic in profile.sensitive_topics] == [
        "fallecimiento de Miguel"
    ]
    assert "Javi" in (profile.data_origin or "")
    assert "Activate profile" in (profile.data_origin or "")

    chat = await app_client.post(
        "/api/v1/chat", json={"message": "Miguel murió, ¿verdad?"}
    )
    assert chat.status_code == 200
    answer = chat.json()["answer"]
    assert "No quiero inventar detalles" in answer
    assert "costura" in answer


async def test_activated_session_cannot_be_edited_or_reactivated(
    app_client: AsyncClient,
) -> None:
    session_id = await _draft_session(app_client, _first_day_package())
    await app_client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={"confirmed_by": "Javi", "scope": "Activate profile"},
    )
    assert (
        await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    ).status_code == 200

    edited = await app_client.put(
        f"{ONBOARDING}/{session_id}/draft", json=_first_day_package()
    )
    assert edited.status_code == 409
    assert (
        await app_client.post(
            f"{ONBOARDING}/{session_id}/consent",
            json={"confirmed_by": "Javi", "scope": "Again"},
        )
    ).status_code == 409
    assert (
        await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    ).status_code == 409
