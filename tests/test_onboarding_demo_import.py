import json
from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import (
    FactStatus,
    MemoryStatus,
    RelationshipStatus,
    SensitivityLevel,
)
from app.importers.onboarding_demo import (
    OnboardingDemoError,
    OnboardingDemoImporter,
    import_path,
)
from app.models import (
    Fact,
    Media,
    Memory,
    MemoryPerson,
    Person,
    PersonAlias,
    Relationship,
    Source,
)

DEMO_DOCUMENT = {
    "schema_version": "1.0",
    "fictional": True,
    "not_real_person": True,
    "notice": "Fictional demo data. Not real-world verification.",
    "profile": {
        "id": "perfil_elena",
        "person_id": "elena",
        "preferred_name": "Elena",
        "sensitive_topics": [
            {
                "topic": "fallecimiento de Miguel",
                "handling": "Do not start this topic.",
                "consent_review_required": True,
            }
        ],
        "data_origin": "Fictional for tests",
    },
    "persons": [
        {
            "id": "elena",
            "first_name": "Elena",
            "last_name": "Martínez Ruiz",
            "display_name": "Elena Martínez Ruiz",
            "birth_date": "1941-05-18",
            "birth_date_precision": "day",
            "is_deceased": False,
            "death_date": None,
            "alias": "Leni",
            "role_for_user": "usuaria",
            "description": "Fictional user persona.",
        },
        {
            "id": "miguel",
            "first_name": "Miguel",
            "last_name": "Soler Pérez",
            "display_name": "Miguel Soler Pérez",
            "birth_date": "1938-01-01",
            "birth_date_precision": "year",
            "is_deceased": True,
            "death_date": None,
            "alias": "Migue",
            "role_for_user": "esposo",
            "description": "Fictional husband.",
        },
        {
            "id": "carmen",
            "first_name": "Carmen",
            "last_name": "Soler Martínez",
            "display_name": "Carmen Soler Martínez",
            "birth_date": "1965-01-01",
            "birth_date_precision": "year",
            "is_deceased": False,
            "death_date": None,
            "alias": "Carmencita",
            "role_for_user": "hija",
            "description": "Fictional daughter.",
        },
    ],
    "sources": [
        {
            "id": "src_elena",
            "source_type": "user",
            "person_id": "elena",
            "description": "Elena's subjective recollections.",
        },
        {
            "id": "src_carmen",
            "source_type": "family_member",
            "person_id": "carmen",
            "description": "Carmen's testimony; confirm with others.",
        },
        {
            "id": "src_family",
            "source_type": "system",
            "person_id": None,
            "description": "Fictional family archive.",
        },
    ],
    "relationships": [
        {
            "person_a_id": "miguel",
            "person_b_id": "elena",
            "relationship_type": "spouse_of",
            "status": "verified",
            "source_id": "src_carmen",
        },
        {
            "person_a_id": "carmen",
            "person_b_id": "elena",
            "relationship_type": "child_of",
            "status": "probable",
            "source_id": "src_carmen",
        },
    ],
    "facts": [
        {
            "id": "fact_001",
            "subject_person_id": "elena",
            "fact_type": "parentesco",
            "value_text": "usuaria",
            "related_person_id": None,
            "status": "verified",
            "source_id": "src_carmen",
            "notes": "Fiction note from the JSON.",
        },
        {
            "id": "fact_002",
            "subject_person_id": "carmen",
            "fact_type": "aficion",
            "value_text": "costura",
            "related_person_id": None,
            "status": "probable",
            "source_id": "src_carmen",
            "notes": None,
        },
    ],
    "memories": [
        {
            "id": "mem_001",
            "title": "La casa de la infancia",
            "narrative": "Elena remembers the patio and the geraniums.",
            "approx_period_text": "1940-1950",
            "place_text": "Alcoy",
            "memory_status": "consolidated",
            "sensitivity_level": "low",
            "source_id": "src_elena",
            "people": [
                {"person_id": "elena", "role": "protagonista"},
                {"person_id": "miguel", "role": "participante"},
            ],
        },
        {
            "id": "mem_002",
            "title": "Una despedida difícil",
            "narrative": "A memory about Miguel's death, kept with high sensitivity.",
            "approx_period_text": "1998",
            "place_text": None,
            "memory_status": "recurring",
            "sensitivity_level": "high",
            "source_id": "src_carmen",
            "people": [{"person_id": "carmen", "role": "participante"}],
        },
    ],
    "testimonies_and_disputes": [
        {
            "id": "version_001",
            "memory_id": "mem_001",
            "source_id": "src_carmen",
            "claim": "The move happened before summer.",
            "status": "pending",
        }
    ],
    "unresolved_questions": ["¿Fecha exacta de boda?"],
    "media_placeholders": [
        {
            "id": "media_001",
            "type": "photo",
            "title": "Álbum familiar",
            "file_path": None,
            "status": "pending_upload",
            "persons": ["elena", "miguel"],
        }
    ],
}


async def _count(session: AsyncSession, model) -> int:
    result = await session.execute(select(func.count()).select_from(model))
    return result.scalar_one()


async def test_import_path_creates_entities_aliases_and_links(
    db_session: AsyncSession, tmp_path
) -> None:
    json_path = tmp_path / "memo_demo.json"
    json_path.write_text(json.dumps(DEMO_DOCUMENT), encoding="utf-8")

    report = await import_path(db_session, json_path)

    assert report.created == {
        "persons": 3,
        "person_aliases": 3,
        "sources": 3,
        "relationships": 2,
        "facts": 2,
        "memories": 2,
        "memory_people": 3,
    }

    persons = (await db_session.execute(select(Person))).scalars().all()
    by_name = {person.first_name: person for person in persons}
    assert set(by_name) == {"Elena", "Miguel", "Carmen"}
    assert by_name["Elena"].aliases[0].alias == "Leni"
    assert by_name["Elena"].birth_date == date(1941, 5, 18)
    assert by_name["Miguel"].is_deceased is True

    sources = (await db_session.execute(select(Source))).scalars().all()
    assert {source.external_reference for source in sources} == {
        "memo-onboarding-demo/src_elena",
        "memo-onboarding-demo/src_carmen",
        "memo-onboarding-demo/src_family",
    }

    memory = (
        await db_session.execute(
            select(Memory).where(Memory.title == "La casa de la infancia")
        )
    ).scalar_one()
    assert memory.memory_status == MemoryStatus.consolidated
    assert memory.place_text == "Alcoy"
    links = (
        (
            await db_session.execute(
                select(MemoryPerson).where(MemoryPerson.memory_id == memory.id)
            )
        )
        .scalars()
        .all()
    )
    assert {link.role for link in links} == {"protagonista", "participante"}
    assert {link.person_id for link in links} == {
        by_name["Elena"].id,
        by_name["Miguel"].id,
    }


async def test_import_is_idempotent(db_session: AsyncSession) -> None:
    importer = OnboardingDemoImporter(db_session)
    first = await importer.run(DEMO_DOCUMENT)
    counts_after_first = {
        "persons": await _count(db_session, Person),
        "aliases": await _count(db_session, PersonAlias),
        "sources": await _count(db_session, Source),
        "relationships": await _count(db_session, Relationship),
        "facts": await _count(db_session, Fact),
        "memories": await _count(db_session, Memory),
        "memory_people": await _count(db_session, MemoryPerson),
    }
    assert first.created["persons"] == 3

    second = await importer.run(DEMO_DOCUMENT)

    assert all(value == 0 for value in second.created.values())
    assert second.existing == {
        "persons": 3,
        "sources": 3,
        "relationships": 2,
        "facts": 2,
        "memories": 2,
    }
    counts_after_second = {
        "persons": await _count(db_session, Person),
        "aliases": await _count(db_session, PersonAlias),
        "sources": await _count(db_session, Source),
        "relationships": await _count(db_session, Relationship),
        "facts": await _count(db_session, Fact),
        "memories": await _count(db_session, Memory),
        "memory_people": await _count(db_session, MemoryPerson),
    }
    assert counts_after_second == counts_after_first


async def test_import_downgrades_fictional_verified_status_and_keeps_provenance(
    db_session: AsyncSession,
) -> None:
    report = await OnboardingDemoImporter(db_session).run(DEMO_DOCUMENT)

    assert report.downgraded_fact_statuses == 1
    assert report.downgraded_relationship_statuses == 1

    facts = (await db_session.execute(select(Fact))).scalars().all()
    assert {fact.status for fact in facts} == {FactStatus.probable}
    imported_fact = next(fact for fact in facts if fact.fact_type == "parentesco")
    assert "not real-world verification" in imported_fact.notes
    assert "stored as 'probable'" in imported_fact.notes
    assert "Fiction note from the JSON." in imported_fact.notes

    relationships = (await db_session.execute(select(Relationship))).scalars().all()
    assert {relationship.status for relationship in relationships} == {
        RelationshipStatus.probable
    }
    assert all(relationship.source_id is not None for relationship in relationships)


async def test_import_reports_unsupported_sections_and_date_precision(
    db_session: AsyncSession,
) -> None:
    report = await OnboardingDemoImporter(db_session).run(DEMO_DOCUMENT)

    skipped = {item.section: item for item in report.skipped_sections}
    assert set(skipped) == {
        "profile",
        "testimonies_and_disputes",
        "unresolved_questions",
        "media_placeholders",
    }
    assert skipped["testimonies_and_disputes"].count == 1
    assert skipped["media_placeholders"].count == 1
    assert report.consent_review_items == 1
    assert report.unsupported_fields == {"persons": ["role_for_user"]}
    assert sorted(report.dropped_birth_dates) == ["carmen", "miguel"]
    assert report.dropped_birth_date_precisions == {
        "carmen": "year",
        "miguel": "year",
    }

    persons = (await db_session.execute(select(Person))).scalars().all()
    by_name = {person.first_name: person for person in persons}
    assert by_name["Elena"].birth_date == date(1941, 5, 18)
    assert by_name["Miguel"].birth_date is None
    assert by_name["Carmen"].birth_date is None

    high_sensitivity = (
        await db_session.execute(
            select(Memory).where(Memory.sensitivity_level == SensitivityLevel.high)
        )
    ).scalar_one()
    assert high_sensitivity.title == "Una despedida difícil"

    assert await _count(db_session, Media) == 0


async def test_import_refuses_documents_not_marked_fictional(
    db_session: AsyncSession,
) -> None:
    document = {**DEMO_DOCUMENT, "fictional": False}

    with pytest.raises(OnboardingDemoError, match="fictional=true"):
        await OnboardingDemoImporter(db_session).run(document)
