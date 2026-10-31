"""Idempotent importer for the fictional Memo onboarding-demo JSON.

This module is an explicit operator entry point; it never runs at application
startup. Run it with the path to the demo document::

    python -m app.importers.onboarding_demo /path/to/memo_familia_ficticia.json

Design notes:

* Symbolic ids in the JSON are resolved to deterministic UUIDv5 values, so
  rerunning the same document skips rows that already exist instead of
  duplicating them.
* The document is explicitly fictional. The JSON ``verified`` status is stored
  as ``probable`` for facts and relationships: a simulated verification inside
  a story is not real-world verification.
* Sections without a repository/domain model (conversational profile, consent
  policy, attributed testimonies/versions, unresolved questions, media
  placeholders) are not silently discarded. The returned report names each
  skipped section, its item count, and the reason.
* Year-only birth dates are stored as ``NULL`` because the schema has no
  precision column; the report lists every affected person.
* The whole document is validated before any write, and every row is flushed
  inside one transaction that is committed only after the last entity. Any
  failure rolls the entire import back, so no partial demo universe persists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import (
    FactStatus,
    MemoryStatus,
    RelationshipStatus,
    RelationshipType,
    SensitivityLevel,
    SourceType,
)
from app.repositories.fact import FactRepository
from app.repositories.memory import MemoryRepository
from app.repositories.person import PersonRepository
from app.repositories.relationship import RelationshipRepository
from app.repositories.source import SourceRepository
from app.schemas.fact import FactCreate
from app.schemas.memory import MemoryCreate, MemoryPersonCreate
from app.schemas.person import PersonAliasCreate, PersonCreate
from app.schemas.relationship import RelationshipCreate
from app.schemas.source import SourceCreate

_DEMO_NAMESPACE = uuid.UUID("442d6f5f-1f88-4f1e-9d9f-6f1b4b1c9a10")
_PROVENANCE_PREFIX = "memo-onboarding-demo"
_IMPORT_NOTE = (
    "Imported from the fictional Memo onboarding demo. "
    "This is not real-world verification."
)
_DOWNGRADE_NOTE = (
    "The JSON marked this item as 'verified' inside the fiction; "
    "it was stored as 'probable'."
)

_PERSON_FIELDS = {
    "id",
    "first_name",
    "last_name",
    "display_name",
    "birth_date",
    "birth_date_precision",
    "is_deceased",
    "death_date",
    "alias",
    "description",
}
_SOURCE_FIELDS = {"id", "source_type", "person_id", "description"}
_RELATIONSHIP_FIELDS = {
    "person_a_id",
    "person_b_id",
    "relationship_type",
    "status",
    "source_id",
    "valid_from",
    "valid_to",
}
_FACT_FIELDS = {
    "id",
    "subject_person_id",
    "fact_type",
    "value_text",
    "value_date",
    "value_number",
    "related_person_id",
    "place_text",
    "valid_from",
    "valid_to",
    "status",
    "source_id",
    "notes",
}
_MEMORY_FIELDS = {
    "id",
    "title",
    "narrative",
    "approx_start_date",
    "approx_end_date",
    "approx_period_text",
    "place_text",
    "memory_status",
    "sensitivity_level",
    "source_id",
    "people",
}


class OnboardingDemoError(ValueError):
    """Raised when the demo document cannot be imported safely."""


@dataclass(frozen=True)
class SkippedSection:
    """A document section intentionally left out of the relational import."""

    section: str
    count: int
    reason: str


@dataclass
class ImportReport:
    """Outcome of one importer run, including explicit skipped content."""

    created: dict[str, int] = field(default_factory=dict)
    existing: dict[str, int] = field(default_factory=dict)
    downgraded_fact_statuses: int = 0
    downgraded_relationship_statuses: int = 0
    dropped_birth_dates: list[str] = field(default_factory=list)
    dropped_birth_date_precisions: dict[str, str] = field(default_factory=dict)
    skipped_sections: list[SkippedSection] = field(default_factory=list)
    unsupported_fields: dict[str, list[str]] = field(default_factory=dict)
    consent_review_items: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "created": dict(self.created),
            "existing": dict(self.existing),
            "downgraded_fact_statuses": self.downgraded_fact_statuses,
            "downgraded_relationship_statuses": self.downgraded_relationship_statuses,
            "dropped_birth_dates": list(self.dropped_birth_dates),
            "dropped_birth_date_precisions": dict(self.dropped_birth_date_precisions),
            "skipped_sections": [asdict(item) for item in self.skipped_sections],
            "unsupported_fields": {
                section: sorted(fields)
                for section, fields in self.unsupported_fields.items()
            },
            "consent_review_items": self.consent_review_items,
        }


def load_document(path: str | Path) -> dict[str, Any]:
    """Load a demo JSON document from an explicit filesystem path."""
    document_path = Path(path)
    try:
        raw = document_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OnboardingDemoError(
            f"Cannot read demo JSON at {document_path}: {exc}"
        ) from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OnboardingDemoError(f"Invalid JSON in {document_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise OnboardingDemoError("The demo document must be a JSON object")
    return document


def _stable_id(entity: str, symbolic_id: str) -> uuid.UUID:
    """Resolve a symbolic JSON id to a stable UUID for idempotent reruns."""
    return uuid.uuid5(_DEMO_NAMESPACE, f"{_PROVENANCE_PREFIX}:{entity}:{symbolic_id}")


def _relationship_symbolic_id(item: dict[str, Any]) -> str:
    return "|".join(
        str(item.get(key, ""))
        for key in (
            "person_a_id",
            "person_b_id",
            "relationship_type",
            "source_id",
        )
    )


def _as_list(document: dict[str, Any], key: str) -> list[Any]:
    value = document.get(key, [])
    if not isinstance(value, list):
        raise OnboardingDemoError(f"'{key}' must be a JSON array")
    return value


def _as_object(item: Any, section: str, index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise OnboardingDemoError(f"{section}[{index}] must be a JSON object")
    return item


def _required_text(
    item: dict[str, Any],
    key: str,
    context: str,
    *,
    max_length: int | None = None,
) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OnboardingDemoError(f"{context}.{key} must be a non-empty string")
    if max_length is not None and len(value) > max_length:
        raise OnboardingDemoError(
            f"{context}.{key} must be at most {max_length} characters"
        )
    return value


def _optional_text(
    item: dict[str, Any],
    key: str,
    context: str,
    *,
    max_length: int | None = None,
) -> str | None:
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise OnboardingDemoError(f"{context}.{key} must be a string or null")
    if max_length is not None and len(value) > max_length:
        raise OnboardingDemoError(
            f"{context}.{key} must be at most {max_length} characters"
        )
    return value


def _optional_date(value: Any, context: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OnboardingDemoError(f"{context} must be an ISO date string or null")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise OnboardingDemoError(
            f"{context} is not a valid ISO date: {value!r}"
        ) from exc


def _optional_decimal(value: Any, context: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise OnboardingDemoError(
            f"{context} must be a number, a numeric string, or null"
        )
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise OnboardingDemoError(
            f"{context} is not a valid number: {value!r}"
        ) from exc
    if not parsed.is_finite():
        raise OnboardingDemoError(f"{context} must be a finite number")
    return parsed


def _enum_value(enum_type, value: Any, context: str):
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise OnboardingDemoError(
            f"{context} has unsupported value {value!r}; expected one of: {allowed}"
        ) from exc


def _index_unique(
    document: dict[str, Any], section: str
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    items = [
        _as_object(item, section, index)
        for index, item in enumerate(_as_list(document, section))
    ]
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        symbolic_id = _required_text(item, "id", section)
        if symbolic_id in indexed:
            raise OnboardingDemoError(
                f"{section} contains duplicate symbolic id {symbolic_id!r}"
            )
        indexed[symbolic_id] = item
    return indexed, items


def _validate_document(document: dict[str, Any]) -> None:
    """Validate everything needed for a safe run before writing any row."""
    if (
        document.get("fictional") is not True
        or document.get("not_real_person") is not True
    ):
        raise OnboardingDemoError(
            "Refusing to import: the document must set fictional=true and "
            "not_real_person=true"
        )

    persons, _ = _index_unique(document, "persons")
    sources, _ = _index_unique(document, "sources")
    facts, _ = _index_unique(document, "facts")
    memories, _ = _index_unique(document, "memories")
    relationships = [
        _as_object(item, "relationships", index)
        for index, item in enumerate(_as_list(document, "relationships"))
    ]

    for person_id, person in persons.items():
        context = f"persons[{person_id}]"
        _required_text(person, "first_name", context, max_length=255)
        _required_text(person, "display_name", context, max_length=255)
        _optional_text(person, "last_name", context, max_length=255)
        _optional_text(person, "alias", context, max_length=255)
        _optional_text(person, "description", context)
        _optional_date(person.get("birth_date"), f"{context}.birth_date")
        _optional_date(person.get("death_date"), f"{context}.death_date")

    for source_id, source in sources.items():
        context = f"sources[{source_id}]"
        _enum_value(SourceType, source.get("source_type"), f"{context}.source_type")
        _optional_text(source, "description", context)
        if len(f"{_PROVENANCE_PREFIX}/{source_id}") > 500:
            raise OnboardingDemoError(
                f"{context}.id is too long to build an external reference"
            )
        person_id = source.get("person_id")
        if person_id is not None and person_id not in persons:
            raise OnboardingDemoError(
                f"{context}.person_id references unknown person {person_id!r}"
            )

    relationship_keys: set[str] = set()
    for index, relationship in enumerate(relationships):
        context = f"relationships[{index}]"
        for key in ("person_a_id", "person_b_id"):
            if relationship.get(key) not in persons:
                raise OnboardingDemoError(
                    f"{context}.{key} references an unknown person "
                    f"{relationship.get(key)!r}"
                )
        if relationship.get("person_a_id") == relationship.get("person_b_id"):
            raise OnboardingDemoError(
                f"{context} must connect two different persons"
            )
        source_reference = relationship.get("source_id")
        if source_reference is not None and source_reference not in sources:
            raise OnboardingDemoError(
                f"{context}.source_id references an unknown source {source_reference!r}"
            )
        _enum_value(
            RelationshipType,
            relationship.get("relationship_type"),
            f"{context}.relationship_type",
        )
        _enum_value(
            RelationshipStatus,
            relationship.get("status"),
            f"{context}.status",
        )
        _optional_date(relationship.get("valid_from"), f"{context}.valid_from")
        _optional_date(relationship.get("valid_to"), f"{context}.valid_to")
        key = _relationship_symbolic_id(relationship)
        if key in relationship_keys:
            raise OnboardingDemoError(
                f"relationships contains a duplicate identity key: {key!r}"
            )
        relationship_keys.add(key)

    for fact_id, fact in facts.items():
        context = f"facts[{fact_id}]"
        _required_text(fact, "fact_type", context, max_length=50)
        _optional_text(fact, "value_text", context)
        _optional_text(fact, "place_text", context, max_length=500)
        _optional_text(fact, "notes", context)
        _optional_decimal(fact.get("value_number"), f"{context}.value_number")
        if fact.get("subject_person_id") not in persons:
            raise OnboardingDemoError(
                f"{context}.subject_person_id references an unknown person"
            )
        related_person_id = fact.get("related_person_id")
        if related_person_id is not None and related_person_id not in persons:
            raise OnboardingDemoError(
                f"{context}.related_person_id references an unknown person"
            )
        if fact.get("source_id") not in sources:
            raise OnboardingDemoError(
                f"{context}.source_id references an unknown source"
            )
        _enum_value(FactStatus, fact.get("status"), f"{context}.status")
        _optional_date(fact.get("value_date"), f"{context}.value_date")
        _optional_date(fact.get("valid_from"), f"{context}.valid_from")
        _optional_date(fact.get("valid_to"), f"{context}.valid_to")

    for memory_id, memory in memories.items():
        context = f"memories[{memory_id}]"
        _required_text(memory, "narrative", context)
        _optional_text(memory, "title", context, max_length=255)
        _optional_text(memory, "approx_period_text", context, max_length=255)
        _optional_text(memory, "place_text", context, max_length=500)
        if memory.get("source_id") not in sources:
            raise OnboardingDemoError(
                f"{context}.source_id references an unknown source"
            )
        _enum_value(
            MemoryStatus, memory.get("memory_status"), f"{context}.memory_status"
        )
        _enum_value(
            SensitivityLevel,
            memory.get("sensitivity_level"),
            f"{context}.sensitivity_level",
        )
        _optional_date(memory.get("approx_start_date"), f"{context}.approx_start_date")
        _optional_date(memory.get("approx_end_date"), f"{context}.approx_end_date")
        people = memory.get("people", [])
        if not isinstance(people, list):
            raise OnboardingDemoError(f"{context}.people must be a JSON array")
        for person_index, person in enumerate(people):
            person = _as_object(person, f"{context}.people", person_index)
            if person.get("person_id") not in persons:
                raise OnboardingDemoError(
                    f"{context}.people[{person_index}].person_id references "
                    "an unknown person"
                )
            _optional_text(
                person,
                "role",
                f"{context}.people[{person_index}]",
                max_length=50,
            )


def _person_birth_date(
    person: dict[str, Any],
) -> tuple[date | None, str | None]:
    parsed = _optional_date(person.get("birth_date"), "birth_date")
    if parsed is None:
        return None, None
    precision = person.get("birth_date_precision", "day")
    if precision == "day":
        return parsed, None
    return None, str(precision)


def _safe_fact_status(value: Any) -> tuple[FactStatus, bool]:
    status = _enum_value(FactStatus, value, "fact status")
    if status is FactStatus.verified:
        return FactStatus.probable, True
    return status, False


def _safe_relationship_status(value: Any) -> tuple[RelationshipStatus, bool]:
    status = _enum_value(RelationshipStatus, value, "relationship status")
    if status is RelationshipStatus.verified:
        return RelationshipStatus.probable, True
    return status, False


def _extra_fields(item: dict[str, Any], supported: set[str]) -> set[str]:
    return set(item) - supported


def _notes_with_provenance(raw_notes: Any, downgraded: bool) -> str:
    parts = [_IMPORT_NOTE]
    if downgraded:
        parts.append(_DOWNGRADE_NOTE)
    if isinstance(raw_notes, str) and raw_notes.strip():
        parts.append(raw_notes.strip())
    return " ".join(parts)


class OnboardingDemoImporter:
    """Create the demo universe through the existing repositories."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._persons = PersonRepository(session)
        self._sources = SourceRepository(session)
        self._relationships = RelationshipRepository(session)
        self._facts = FactRepository(session)
        self._memories = MemoryRepository(session)

    async def run(self, document: dict[str, Any]) -> ImportReport:
        _validate_document(document)
        try:
            return await self._write_document(document)
        except Exception:
            await self._session.rollback()
            raise

    async def _write_document(self, document: dict[str, Any]) -> ImportReport:
        report = ImportReport(
            created={
                "persons": 0,
                "person_aliases": 0,
                "sources": 0,
                "relationships": 0,
                "facts": 0,
                "memories": 0,
                "memory_people": 0,
            },
            existing={
                "persons": 0,
                "sources": 0,
                "relationships": 0,
                "facts": 0,
                "memories": 0,
            },
        )

        person_items, _ = _index_unique(document, "persons")
        source_items, _ = _index_unique(document, "sources")
        fact_items, _ = _index_unique(document, "facts")
        memory_items, _ = _index_unique(document, "memories")
        relationship_items = [
            _as_object(item, "relationships", index)
            for index, item in enumerate(_as_list(document, "relationships"))
        ]

        person_ids = {
            symbolic_id: _stable_id("person", symbolic_id)
            for symbolic_id in person_items
        }
        source_ids = {
            symbolic_id: _stable_id("source", symbolic_id)
            for symbolic_id in source_items
        }
        relationship_ids = {
            _relationship_symbolic_id(item): _stable_id(
                "relationship", _relationship_symbolic_id(item)
            )
            for item in relationship_items
        }
        fact_ids = {
            symbolic_id: _stable_id("fact", symbolic_id) for symbolic_id in fact_items
        }
        memory_ids = {
            symbolic_id: _stable_id("memory", symbolic_id)
            for symbolic_id in memory_items
        }

        report.skipped_sections = self._skipped_sections(document)
        report.consent_review_items = self._consent_review_items(document)
        report.unsupported_fields = self._unsupported_fields(document)

        for symbolic_id, person in person_items.items():
            person_id = person_ids[symbolic_id]
            if await self._persons.get(person_id) is not None:
                report.existing["persons"] += 1
                continue
            birth_date, dropped_precision = _person_birth_date(person)
            if dropped_precision is not None:
                report.dropped_birth_dates.append(symbolic_id)
                report.dropped_birth_date_precisions[symbolic_id] = dropped_precision
            alias = person.get("alias")
            aliases = (
                [PersonAliasCreate(alias=alias, alias_type="nickname")]
                if isinstance(alias, str) and alias.strip()
                else []
            )
            await self._persons.create(
                PersonCreate(
                    first_name=_required_text(
                        person, "first_name", f"persons[{symbolic_id}]"
                    ),
                    last_name=person.get("last_name"),
                    display_name=person.get("display_name"),
                    birth_date=birth_date,
                    death_date=_optional_date(
                        person.get("death_date"), f"persons[{symbolic_id}].death_date"
                    ),
                    is_deceased=bool(person.get("is_deceased", False)),
                    description=person.get("description"),
                    aliases=aliases,
                ),
                entity_id=person_id,
                commit=False,
            )
            report.created["persons"] += 1
            report.created["person_aliases"] += len(aliases)

        for symbolic_id, source in source_items.items():
            source_id = source_ids[symbolic_id]
            if await self._sources.get(source_id) is not None:
                report.existing["sources"] += 1
                continue
            source_person_id = source.get("person_id")
            await self._sources.create(
                SourceCreate(
                    source_type=_enum_value(
                        SourceType,
                        source.get("source_type"),
                        f"sources[{symbolic_id}].source_type",
                    ),
                    person_id=person_ids[source_person_id]
                    if source_person_id is not None
                    else None,
                    description=source.get("description"),
                    external_reference=f"{_PROVENANCE_PREFIX}/{symbolic_id}",
                ),
                entity_id=source_id,
                commit=False,
            )
            report.created["sources"] += 1

        for relationship in relationship_items:
            key = _relationship_symbolic_id(relationship)
            relationship_id = relationship_ids[key]
            if await self._relationships.get(relationship_id) is not None:
                report.existing["relationships"] += 1
                continue
            status, downgraded = _safe_relationship_status(relationship.get("status"))
            if downgraded:
                report.downgraded_relationship_statuses += 1
            source_reference = relationship.get("source_id")
            await self._relationships.create(
                RelationshipCreate(
                    person_a_id=person_ids[relationship["person_a_id"]],
                    person_b_id=person_ids[relationship["person_b_id"]],
                    relationship_type=_enum_value(
                        RelationshipType,
                        relationship.get("relationship_type"),
                        "relationship_type",
                    ),
                    status=status,
                    source_id=source_ids[source_reference]
                    if source_reference is not None
                    else None,
                    valid_from=_optional_date(
                        relationship.get("valid_from"), "relationship.valid_from"
                    ),
                    valid_to=_optional_date(
                        relationship.get("valid_to"), "relationship.valid_to"
                    ),
                ),
                entity_id=relationship_id,
                commit=False,
            )
            report.created["relationships"] += 1

        for symbolic_id, fact in fact_items.items():
            fact_id = fact_ids[symbolic_id]
            if await self._facts.get(fact_id) is not None:
                report.existing["facts"] += 1
                continue
            status, downgraded = _safe_fact_status(fact.get("status"))
            if downgraded:
                report.downgraded_fact_statuses += 1
            related_person_id = fact.get("related_person_id")
            await self._facts.create(
                FactCreate(
                    subject_person_id=person_ids[fact["subject_person_id"]],
                    fact_type=_required_text(
                        fact, "fact_type", f"facts[{symbolic_id}]"
                    ),
                    value_text=fact.get("value_text"),
                    value_date=_optional_date(
                        fact.get("value_date"), f"facts[{symbolic_id}].value_date"
                    ),
                    value_number=fact.get("value_number"),
                    related_person_id=person_ids[related_person_id]
                    if related_person_id is not None
                    else None,
                    place_text=fact.get("place_text"),
                    valid_from=_optional_date(
                        fact.get("valid_from"), f"facts[{symbolic_id}].valid_from"
                    ),
                    valid_to=_optional_date(
                        fact.get("valid_to"), f"facts[{symbolic_id}].valid_to"
                    ),
                    status=status,
                    source_id=source_ids[fact["source_id"]],
                    notes=_notes_with_provenance(fact.get("notes"), downgraded),
                ),
                entity_id=fact_id,
                commit=False,
            )
            report.created["facts"] += 1

        for symbolic_id, memory in memory_items.items():
            memory_id = memory_ids[symbolic_id]
            if await self._memories.get(memory_id) is not None:
                report.existing["memories"] += 1
                continue
            people = [
                MemoryPersonCreate(
                    person_id=person_ids[person["person_id"]],
                    role=person.get("role"),
                )
                for person in memory.get("people", [])
            ]
            await self._memories.create(
                MemoryCreate(
                    title=memory.get("title"),
                    narrative=_required_text(
                        memory, "narrative", f"memories[{symbolic_id}]"
                    ),
                    approx_start_date=_optional_date(
                        memory.get("approx_start_date"),
                        f"memories[{symbolic_id}].approx_start_date",
                    ),
                    approx_end_date=_optional_date(
                        memory.get("approx_end_date"),
                        f"memories[{symbolic_id}].approx_end_date",
                    ),
                    approx_period_text=memory.get("approx_period_text"),
                    place_text=memory.get("place_text"),
                    memory_status=_enum_value(
                        MemoryStatus,
                        memory.get("memory_status"),
                        f"memories[{symbolic_id}].memory_status",
                    ),
                    sensitivity_level=_enum_value(
                        SensitivityLevel,
                        memory.get("sensitivity_level"),
                        f"memories[{symbolic_id}].sensitivity_level",
                    ),
                    source_id=source_ids[memory["source_id"]],
                    people=people,
                ),
                entity_id=memory_id,
                commit=False,
            )
            report.created["memories"] += 1
            report.created["memory_people"] += len(people)

        await self._session.commit()
        return report

    @staticmethod
    def _skipped_sections(document: dict[str, Any]) -> list[SkippedSection]:
        skipped: list[SkippedSection] = []
        if "profile" in document:
            skipped.append(
                SkippedSection(
                    section="profile",
                    count=1 if document.get("profile") else 0,
                    reason=(
                        "No conversational-profile model exists; importing it "
                        "would silently invent persistence."
                    ),
                )
            )
        for section, reason in (
            (
                "testimonies_and_disputes",
                (
                    "Attributed testimony/version records have no model; facts are "
                    "not a safe substitute for competing claims."
                ),
            ),
            (
                "unresolved_questions",
                (
                    "Unresolved questions are not knowledge gaps with sources; "
                    "creating gaps here would fabricate provenance."
                ),
            ),
            (
                "media_placeholders",
                (
                    "Media import is not part of this milestone; placeholders are "
                    "reported instead of fabricating storage paths."
                ),
            ),
        ):
            if section in document:
                value = document.get(section)
                count = len(value) if isinstance(value, list) else 1
                skipped.append(
                    SkippedSection(section=section, count=count, reason=reason)
                )
        return skipped

    @staticmethod
    def _consent_review_items(document: dict[str, Any]) -> int:
        profile = document.get("profile")
        if not isinstance(profile, dict):
            return 0
        topics = profile.get("sensitive_topics", [])
        if not isinstance(topics, list):
            return 0
        return sum(
            1
            for topic in topics
            if isinstance(topic, dict) and topic.get("consent_review_required") is True
        )

    @staticmethod
    def _unsupported_fields(document: dict[str, Any]) -> dict[str, list[str]]:
        definitions = (
            ("persons", _PERSON_FIELDS),
            ("sources", _SOURCE_FIELDS),
            ("relationships", _RELATIONSHIP_FIELDS),
            ("facts", _FACT_FIELDS),
            ("memories", _MEMORY_FIELDS),
        )
        result: dict[str, list[str]] = {}
        for section, supported in definitions:
            fields: set[str] = set()
            for item in _as_list(document, section):
                if isinstance(item, dict):
                    fields.update(_extra_fields(item, supported))
            if fields:
                result[section] = sorted(fields)
        return result


async def import_path(session: AsyncSession, path: str | Path) -> ImportReport:
    """Load a JSON path and import it in the caller's session."""
    return await OnboardingDemoImporter(session).run(load_document(path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import the fictional Memo onboarding-demo JSON. The importer is "
            "idempotent and never runs at application startup."
        )
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the demo JSON document (for example memo_familia_ficticia.json).",
    )
    return parser


async def _run_cli(path: Path) -> int:
    from app.db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        report = await import_path(session, path)
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: ``python -m app.importers.onboarding_demo <json_path>``."""
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(_run_cli(args.json_path))
    except OnboardingDemoError as exc:
        raise SystemExit(f"onboarding-demo import failed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ImportReport",
    "OnboardingDemoError",
    "OnboardingDemoImporter",
    "SkippedSection",
    "import_path",
    "load_document",
    "main",
]
