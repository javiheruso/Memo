"""Regression coverage for onboarding/materialization validation parity.

The onboarding draft must never accept a payload that the target schemas used
during activation would reject. These tests pin the shared length limits, the
full-boundary happy path, and the controlled failure path for legacy invalid
rows.
"""

import hashlib
import json
import uuid

import pytest
from annotated_types import MaxLen
from httpx import AsyncClient
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    ConversationProfile,
    Fact,
    Memory,
    OnboardingSession,
    Person,
    Relationship,
    Source,
)
from app.schemas.fact import FactCreate
from app.schemas.memory import MemoryCreate
from app.schemas.onboarding import (
    OnboardingFact,
    OnboardingIdentity,
    OnboardingMemory,
    OnboardingPerson,
)
from app.schemas.person import PersonCreate
from app.schemas.profile import ConversationProfileCreate, SensitiveTopicCreate

API = "/api/v1/admin"
ONBOARDING = f"{API}/onboarding"
UI = "/admin"


def _complete_package() -> dict:
    return {
        "identity": {
            "first_name": "Elena",
            "last_name": "Ruiz",
            "preferred_name": "Elenita",
            "language": "es-ES",
        },
        "reviewer": {"name": "Javi", "relationship": "hijo"},
        "contributors": [{"name": "Javi", "relationship": "hijo"}],
        "sources": [
            {
                "key": "family",
                "source_type": "family_member",
                "description": "Javi, son",
            }
        ],
        "people": [
            {
                "key": "miguel",
                "first_name": "Miguel",
                "last_name": "Navarro",
                "relationship_type": "spouse_of",
                "certainty": "probable",
                "source_key": "family",
                "notes": "Elena's husband",
            }
        ],
        "sensitive_topics": [
            {
                "topic": "fallecimiento de Miguel",
                "handling": "Do not initiate; redirect gently.",
                "triggers": ["murió"],
            }
        ],
        "facts": [
            {
                "fact_type": "hobby",
                "value_text": "costura",
                "status": "verified",
                "source_key": "family",
                "notes": "Confirmed by the family",
            }
        ],
        "memories": [
            {
                "title": "Tardes de costura",
                "narrative": "Elena cosía con Miguel los domingos.",
                "source_key": "family",
                "people_keys": ["subject", "miguel"],
            }
        ],
        "preferences": {
            "positive_topics": ["costura", "familia"],
            "never_do": ["No hablar de política"],
            "conversational_style": "Pausado",
        },
    }


async def _draft_and_consent(client: AsyncClient, payload: dict) -> str:
    created = await client.post(ONBOARDING, json=payload)
    assert created.status_code == 201
    session_id = created.json()["id"]
    consent = await client.post(
        f"{ONBOARDING}/{session_id}/consent",
        json={"confirmed_by": "Javi", "scope": "Activate profile"},
    )
    assert consent.status_code == 200
    return session_id


def _max_length(model: type[BaseModel], field: str) -> int | None:
    for metadata in model.model_fields[field].metadata:
        if isinstance(metadata, MaxLen):
            return metadata.max_length
    return None


# Every onboarding field that feeds a length-constrained target schema must
# declare the same maximum. This catches future drift automatically.
PARITY_FIELDS = [
    (OnboardingIdentity, "first_name", PersonCreate, "first_name"),
    (OnboardingIdentity, "last_name", PersonCreate, "last_name"),
    (
        OnboardingIdentity,
        "preferred_name",
        ConversationProfileCreate,
        "preferred_name",
    ),
    (OnboardingIdentity, "language", ConversationProfileCreate, "language"),
    (OnboardingPerson, "first_name", PersonCreate, "first_name"),
    (OnboardingPerson, "last_name", PersonCreate, "last_name"),
    (OnboardingFact, "fact_type", FactCreate, "fact_type"),
    (OnboardingMemory, "title", MemoryCreate, "title"),
    (SensitiveTopicCreate, "topic", SensitiveTopicCreate, "topic"),
]


@pytest.mark.parametrize(
    ("draft_model", "draft_field", "target_model", "target_field"),
    PARITY_FIELDS,
)
def test_onboarding_length_limits_match_materialization_targets(
    draft_model: type[BaseModel],
    draft_field: str,
    target_model: type[BaseModel],
    target_field: str,
) -> None:
    assert _max_length(draft_model, draft_field) == _max_length(
        target_model, target_field
    )


OVERLONG_PAYLOADS = [
    pytest.param(
        {"identity": {"first_name": "x" * 256}},
        "identity.first_name",
        id="identity-first-name",
    ),
    pytest.param(
        {"identity": {"last_name": "x" * 256}},
        "identity.last_name",
        id="identity-last-name",
    ),
    pytest.param(
        {"identity": {"preferred_name": "x" * 256}},
        "identity.preferred_name",
        id="identity-preferred-name",
    ),
    pytest.param(
        {"identity": {"language": "x" * 51}},
        "identity.language",
        id="identity-language",
    ),
    pytest.param(
        {
            "people": [
                {"key": "miguel", "first_name": "Miguel", "last_name": "x" * 256}
            ]
        },
        "people.0.last_name",
        id="person-last-name",
    ),
]


@pytest.mark.parametrize(("payload", "location"), OVERLONG_PAYLOADS)
async def test_onboarding_draft_rejects_overlong_identity_and_person_fields(
    app_client: AsyncClient, payload: dict, location: str
) -> None:
    response = await app_client.post(ONBOARDING, json=payload)
    assert response.status_code == 422
    problems = response.json()["detail"]
    expected = location.split(".")
    assert any(
        [str(part) for part in problem["loc"]][-len(expected) :] == expected
        for problem in problems
    ), problems


async def test_onboarding_max_length_payload_activates(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    package = _complete_package()
    source_key = "s" * 50
    person_key = "p" * 50
    package["identity"] = {
        "first_name": "f" * 255,
        "last_name": "l" * 255,
        "preferred_name": "p" * 255,
        "language": "x" * 50,
    }
    package["contributors"][0]["name"] = "c" * 255
    package["sources"][0] = {
        "key": source_key,
        "source_type": "family_member",
        "description": "d" * 1000,
    }
    package["people"][0] = {
        "key": person_key,
        "first_name": "n" * 255,
        "last_name": "l" * 255,
        "relationship_type": "spouse_of",
        "certainty": "probable",
        "source_key": source_key,
        "notes": "n" * 1000,
    }
    package["facts"][0].update(
        {
            "fact_type": "t" * 50,
            "value_text": "v" * 1000,
            "source_key": source_key,
            "notes": "n" * 1000,
        }
    )
    package["memories"][0].update(
        {
            "title": "t" * 255,
            "narrative": "n" * 1000,
            "source_key": source_key,
            "people_keys": ["subject", person_key],
        }
    )
    package["sensitive_topics"][0].update(
        {
            "topic": "s" * 255,
            "handling": "h" * 1000,
            "triggers": ["t" * 100],
        }
    )
    package["preferences"] = {
        "positive_topics": ["p" * 100],
        "never_do": ["n" * 100],
        "conversational_style": "s" * 1000,
    }

    session_id = await _draft_and_consent(app_client, package)
    activated = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert activated.status_code == 200, activated.text
    body = activated.json()
    assert body["status"] == "activated"
    assert body["profile_id"] is not None

    subject = await db_session.get(Person, uuid.UUID(body["subject_person_id"]))
    assert subject is not None
    assert subject.first_name == "f" * 255
    assert subject.last_name == "l" * 255
    # first_name + last_name can exceed the 255-char display_name column.
    assert len(subject.display_name) <= 255


async def _make_legacy_payload_invalid(
    db_session: AsyncSession, session_id: str
) -> OnboardingSession:
    row = await db_session.get(OnboardingSession, uuid.UUID(session_id))
    assert row is not None
    payload = dict(row.payload)
    payload["identity"] = {
        **payload["identity"],
        "first_name": "x" * 256,
    }
    row.payload = payload
    # Keep the consent fingerprint matching so activation reaches validation
    # instead of failing earlier on the consent mismatch guard.
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    row.consent_payload_fingerprint = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    await db_session.commit()
    return row


async def test_legacy_invalid_payload_activation_is_a_controlled_422(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    session_id = await _draft_and_consent(app_client, _complete_package())
    row = await _make_legacy_payload_invalid(db_session, session_id)

    response = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert response.status_code == 422
    problems = response.json()["detail"]
    assert any(
        "identity.first_name" in problem and "at most 255" in problem
        for problem in problems
    ), problems

    await db_session.refresh(row)
    assert row.status.value == "draft"
    assert row.subject_person_id is None
    assert (
        await db_session.execute(select(func.count()).select_from(Person))
    ).scalar_one() == 0


async def test_legacy_invalid_payload_reads_are_controlled(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    created = await app_client.post(ONBOARDING, json=_complete_package())
    session_id = created.json()["id"]
    await _make_legacy_payload_invalid(db_session, session_id)

    api_detail = await app_client.get(f"{ONBOARDING}/{session_id}")
    assert api_detail.status_code == 422
    assert any(
        "identity.first_name" in problem
        for problem in api_detail.json()["detail"]
    )

    api_list = await app_client.get(ONBOARDING)
    assert api_list.status_code == 422

    ui_list = await app_client.get(f"{UI}/onboarding")
    assert ui_list.status_code == 200
    assert "borradores guardados con datos inválidos" in ui_list.text

    ui_detail = await app_client.get(
        f"{UI}/onboarding/{session_id}", follow_redirects=True
    )
    assert ui_detail.status_code == 200
    assert "borradores guardados con datos inválidos" in ui_detail.text


async def test_saving_a_valid_draft_repairs_a_legacy_invalid_payload(
    app_client: AsyncClient, db_session: AsyncSession
) -> None:
    created = await app_client.post(ONBOARDING, json=_complete_package())
    session_id = created.json()["id"]
    await _make_legacy_payload_invalid(db_session, session_id)

    repaired = await app_client.put(
        f"{ONBOARDING}/{session_id}/draft", json=_complete_package()
    )
    assert repaired.status_code == 200
    assert repaired.json()["payload"]["identity"]["first_name"] == "Elena"
    assert (
        await app_client.get(f"{ONBOARDING}/{session_id}")
    ).status_code == 200


async def test_activation_maps_unexpected_target_validation_and_rolls_back(
    app_client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = await _draft_and_consent(app_client, _complete_package())

    class _StrictPerson(BaseModel):
        first_name: str = Field(max_length=3)

    def _strict_person_create(**kwargs):
        return _StrictPerson.model_validate(
            {"first_name": kwargs["first_name"]}
        )

    monkeypatch.setattr(
        "app.services.onboarding.PersonCreate", _strict_person_create
    )

    response = await app_client.post(f"{ONBOARDING}/{session_id}/activate")
    assert response.status_code == 422
    problems = response.json()["detail"]
    assert any("first_name" in problem and "at most 3" in problem for problem in problems)

    row = await db_session.get(OnboardingSession, uuid.UUID(session_id))
    assert row is not None
    assert row.status.value == "draft"
    assert row.activated_at is None
    for model in (
        Source,
        Person,
        Relationship,
        Fact,
        Memory,
        ConversationProfile,
    ):
        count = (
            await db_session.execute(select(func.count()).select_from(model))
        ).scalar_one()
        assert count == 0, f"{model.__name__} should have rolled back"
