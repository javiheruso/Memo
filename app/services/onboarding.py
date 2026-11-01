"""Admin/family-led onboarding workflow.

The service owns the draft lifecycle and the explicit activation step. Drafts
are resumable JSON validated by the onboarding schemas; activation is gated on
a recorded operator consent and materializes the real rows through the same
repositories the admin API already uses, so trust and provenance rules are not
bypassed.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import MemoryStatus, OnboardingStatus
from app.embeddings.base import Embedder
from app.embeddings.memory_text import memory_search_text
from app.models import OnboardingSession
from app.repositories.fact import FactRepository
from app.repositories.memory import MemoryRepository
from app.repositories.onboarding import OnboardingRepository
from app.repositories.person import PersonRepository
from app.repositories.profile import ConversationProfileRepository
from app.repositories.relationship import RelationshipRepository
from app.repositories.source import SourceRepository
from app.schemas.fact import FactCreate
from app.schemas.memory import MemoryCreate, MemoryPersonCreate
from app.schemas.onboarding import OnboardingDraft
from app.schemas.person import PersonCreate
from app.schemas.profile import ConversationProfileCreate
from app.schemas.relationship import RelationshipCreate
from app.schemas.source import SourceCreate


class OnboardingError(Exception):
    """Base class for onboarding workflow failures."""


class OnboardingNotFoundError(OnboardingError):
    """The requested onboarding session does not exist."""


class OnboardingNotDraftError(OnboardingError):
    """The session is already activated and can no longer be edited."""


class OnboardingConsentRequiredError(OnboardingError):
    """Activation was requested before the operator consent was recorded."""


class OnboardingConsentMismatchError(OnboardingError):
    """The recorded consent no longer matches the draft under review."""


class OnboardingValidationError(OnboardingError):
    """The draft does not yet contain a complete first-day package."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


class OnboardingDraftInvalidError(OnboardingValidationError):
    """The draft is structurally invalid and must not be persisted."""


PROGRESS_STEPS: tuple[str, ...] = (
    "identity",
    "reviewer",
    "contributors",
    "sensitive_topics",
    "sources",
    "people",
    "content",
    "preferences",
)


def _pydantic_problems(exc: ValidationError) -> list[str]:
    """Render every pydantic error as a stable, human-readable problem."""
    problems = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "invalid input"))
        problems.append(f"{loc}: {message}" if loc else message)
    return problems or ["draft payload is invalid"]


class OnboardingService:
    def __init__(
        self, session: AsyncSession, *, embedder: Embedder | None = None
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._onboarding = OnboardingRepository(session)
        self._persons = PersonRepository(session)
        self._sources = SourceRepository(session)
        self._relationships = RelationshipRepository(session)
        self._facts = FactRepository(session)
        self._memories = MemoryRepository(session)
        self._profiles = ConversationProfileRepository(session)

    async def create(
        self, draft: OnboardingDraft | None = None
    ) -> OnboardingSession:
        validated = self._validated_for_persistence(draft or OnboardingDraft())
        row = OnboardingSession(
            status=OnboardingStatus.draft,
            payload=validated.model_dump(mode="json"),
        )
        self._session.add(row)
        await self._session.commit()
        return await self._require(row.id)

    async def get(self, session_id: uuid.UUID) -> OnboardingSession:
        return await self._require(session_id)

    async def list(self, *, limit: int = 100) -> list[OnboardingSession]:
        return await self._onboarding.list_recent(limit=limit)

    async def save_draft(
        self, session_id: uuid.UUID, draft: OnboardingDraft
    ) -> OnboardingSession:
        # Revalidate before touching the row: UI forms mutate the lists of an
        # already-parsed draft in place, which bypasses pydantic validation and
        # could otherwise persist duplicate/reserved keys that break every
        # later read.
        validated = self._validated_for_persistence(draft)
        row = await self._require_draft(session_id, for_update=True)
        row.payload = validated.model_dump(mode="json")
        if (
            row.consent_confirmed
            and row.consent_payload_fingerprint
            != self.draft_fingerprint(validated)
        ):
            self._clear_consent(row)
        await self._session.commit()
        return await self._require(session_id)

    async def record_consent(
        self, session_id: uuid.UUID, *, confirmed_by: str, scope: str
    ) -> OnboardingSession:
        """Record explicit operator consent; never activates in the same call."""
        row = await self._require_draft(session_id, for_update=True)
        row.consent_confirmed = True
        row.consent_confirmed_by = confirmed_by
        row.consent_confirmed_at = datetime.now(UTC)
        row.consent_scope = scope
        row.consent_payload_fingerprint = self.draft_fingerprint(self.draft_of(row))
        await self._session.commit()
        return await self._require(session_id)

    async def activate(self, session_id: uuid.UUID) -> OnboardingSession:
        row = await self._require_draft(session_id, for_update=True)
        if not row.consent_confirmed:
            raise OnboardingConsentRequiredError(
                "Explicit operator consent must be recorded before activation"
            )
        draft = self.draft_of(row)
        if row.consent_payload_fingerprint != self.draft_fingerprint(draft):
            raise OnboardingConsentMismatchError(
                "Recorded consent does not match the current draft; "
                "review the draft and record consent again"
            )
        problems = self.validation_problems(draft)
        if problems:
            raise OnboardingValidationError(problems)

        activated_at = datetime.now(UTC)
        try:
            claimed = await self._onboarding.claim_activation(
                session_id, activated_at=activated_at
            )
            if not claimed:
                raise OnboardingNotDraftError(
                    "Activated onboarding sessions cannot be edited or reactivated"
                )
            try:
                await self._materialize(row, draft)
            except ValidationError as exc:
                # The draft is revalidated before activation, so a target
                # schema rejection here means the onboarding schema and the
                # materialization schema drifted apart. Report it as a
                # controlled draft problem instead of letting pydantic's
                # ValidationError escape as a 500, while the outer handler
                # still rolls the whole transaction (including the activation
                # claim) back.
                raise OnboardingValidationError(_pydantic_problems(exc)) from exc
            row.status = OnboardingStatus.activated
            row.activated_at = activated_at
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        return await self._require(session_id)

    def draft_of(self, row: OnboardingSession) -> OnboardingDraft:
        """Parse a persisted payload, reporting legacy invalid rows cleanly.

        Rows written before a schema tightening (or by direct DB edits) may no
        longer validate. Raising ``OnboardingDraftInvalidError`` lets every
        caller surface a controlled validation response instead of an uncaught
        ``ValidationError``/500 during reads or activation.
        """
        try:
            return OnboardingDraft.model_validate(row.payload or {})
        except ValidationError as exc:
            raise OnboardingDraftInvalidError(_pydantic_problems(exc)) from exc

    @staticmethod
    def _validated_for_persistence(draft: OnboardingDraft) -> OnboardingDraft:
        """Return a freshly validated copy before any write.

        ``save_draft`` receives an already-typed model, so passing it around is
        not enough: in-place list mutations (the UI appends entries directly)
        skip pydantic validation. Re-parsing the dumped payload restores the
        invariant that only structurally valid drafts reach the database.
        """
        try:
            return OnboardingDraft.model_validate(draft.model_dump(mode="json"))
        except ValidationError as exc:
            raise OnboardingDraftInvalidError(_pydantic_problems(exc)) from exc

    def validation_problems_by_step(
        self, draft: OnboardingDraft
    ) -> dict[str, list[str]]:
        """Group every activation blocker under its progress step.

        The progress checklist and ``validation_problems`` share this mapping,
        so the UI can never count a step as complete while activation still
        rejects the same draft.
        """
        issues: dict[str, list[str]] = {step: [] for step in PROGRESS_STEPS}
        if not draft.identity.first_name:
            issues["identity"].append("identity.first_name is required")
        if not draft.reviewer.name:
            issues["reviewer"].append("reviewer.name is required")
        if not draft.contributors:
            issues["contributors"].append(
                "at least one permitted contributor is required"
            )
        if not draft.sources:
            issues["sources"].append("at least one source is required")
        if not draft.people:
            issues["people"].append("at least one person in the circle is required")
        if not draft.sensitive_topics:
            issues["sensitive_topics"].append(
                "at least one sensitive topic is required"
            )
        normalized_topics = [
            topic.topic.strip().casefold() for topic in draft.sensitive_topics
        ]
        if len(normalized_topics) != len(set(normalized_topics)):
            issues["sensitive_topics"].append(
                "sensitive topics must be unique (case-insensitive)"
            )
        for topic in draft.sensitive_topics:
            if not topic.handling:
                issues["sensitive_topics"].append(
                    f"sensitive topic '{topic.topic}' needs handling guidance"
                )
        if not (draft.facts or draft.memories):
            issues["content"].append("at least one fact or memory is required")
        if not (
            draft.preferences.positive_topics
            or draft.preferences.never_do
            or draft.preferences.conversational_style
        ):
            issues["preferences"].append(
                "at least one conversation preference is required"
            )

        source_keys = {source.key for source in draft.sources}
        person_keys = {person.key for person in draft.people}
        for person in draft.people:
            if (
                person.source_key is not None
                and person.source_key not in source_keys
            ):
                issues["people"].append(
                    f"person '{person.first_name}' references unknown source "
                    f"'{person.source_key}'"
                )
        for fact in draft.facts:
            if fact.source_key not in source_keys:
                issues["content"].append(
                    f"fact '{fact.fact_type}' references unknown source "
                    f"'{fact.source_key}'"
                )
            if fact.subject_key != "subject" and fact.subject_key not in person_keys:
                issues["content"].append(
                    f"fact '{fact.fact_type}' references unknown person "
                    f"'{fact.subject_key}'"
                )
        for memory in draft.memories:
            label = memory.title or memory.narrative[:30]
            if memory.source_key not in source_keys:
                issues["content"].append(
                    f"memory '{label}' references unknown source "
                    f"'{memory.source_key}'"
                )
            if len(memory.people_keys) != len(set(memory.people_keys)):
                issues["content"].append(
                    f"memory '{label}' repeats a person key"
                )
            for key in memory.people_keys:
                if key != "subject" and key not in person_keys:
                    issues["content"].append(
                        f"memory '{label}' references unknown person '{key}'"
                    )
        return issues

    def validation_problems(self, draft: OnboardingDraft) -> list[str]:
        """Return every reason the draft is not yet activatable."""
        issues = self.validation_problems_by_step(draft)
        return [
            problem for step in PROGRESS_STEPS for problem in issues[step]
        ]

    def progress(
        self, row: OnboardingSession, draft: OnboardingDraft
    ) -> dict:
        """Shared completion state for the list and detail checklists."""
        issues = self.validation_problems_by_step(draft)
        consent_matches = bool(
            row.consent_confirmed
            and row.consent_payload_fingerprint is not None
            and row.consent_payload_fingerprint
            == self.draft_fingerprint(draft)
        )
        steps = {step: not problems for step, problems in issues.items()}
        steps["consent"] = consent_matches
        steps["activated"] = row.status is OnboardingStatus.activated
        required = (*PROGRESS_STEPS, "consent")
        blockers = [
            problem for step in PROGRESS_STEPS for problem in issues[step]
        ]
        if not row.consent_confirmed:
            blockers.append("explicit operator consent is required")
        elif not consent_matches:
            blockers.append("recorded consent does not match the current draft")
        return {
            "steps": steps,
            "done": sum(1 for step in required if steps[step]),
            "total": len(required),
            "blockers": blockers,
            "ready": not blockers,
        }

    async def _materialize(
        self, row: OnboardingSession, draft: OnboardingDraft
    ) -> None:
        source_rows = {}
        for source in draft.sources:
            source_rows[source.key] = await self._sources.create(
                SourceCreate(
                    source_type=source.source_type,
                    description=source.description,
                    external_reference=f"onboarding/{row.id}/{source.key}",
                ),
                commit=False,
            )

        subject = await self._persons.create(
            PersonCreate(
                first_name=draft.identity.first_name,
                last_name=draft.identity.last_name,
            ),
            commit=False,
        )
        person_rows = {"subject": subject}

        for person in draft.people:
            created = await self._persons.create(
                PersonCreate(
                    first_name=person.first_name,
                    last_name=person.last_name,
                    description=person.notes,
                ),
                commit=False,
            )
            person_rows[person.key] = created
            source = (
                source_rows.get(person.source_key)
                if person.source_key is not None
                else None
            )
            await self._relationships.create(
                RelationshipCreate(
                    person_a_id=subject.id,
                    person_b_id=created.id,
                    relationship_type=person.relationship_type,
                    status=person.certainty,
                    source_id=source.id if source is not None else None,
                ),
                commit=False,
            )

        for fact in draft.facts:
            await self._facts.create(
                FactCreate(
                    subject_person_id=person_rows[fact.subject_key].id,
                    fact_type=fact.fact_type,
                    value_text=fact.value_text,
                    status=fact.status,
                    source_id=source_rows[fact.source_key].id,
                    notes=fact.notes,
                ),
                commit=False,
            )

        for memory in draft.memories:
            embedding = None
            if self._embedder is not None and self._memories.supports_vector_search:
                embedding = await self._embedder.embed(
                    memory_search_text(
                        title=memory.title, narrative=memory.narrative
                    )
                )
            await self._memories.create(
                MemoryCreate(
                    title=memory.title,
                    narrative=memory.narrative,
                    sensitivity_level=memory.sensitivity_level,
                    memory_status=MemoryStatus.candidate,
                    source_id=source_rows[memory.source_key].id,
                    people=[
                        MemoryPersonCreate(person_id=person_rows[key].id)
                        for key in memory.people_keys
                    ],
                ),
                embedding=embedding,
                commit=False,
            )

        profile_source = next(iter(source_rows.values()), None)
        profile = await self._profiles.create(
            ConversationProfileCreate(
                person_id=subject.id,
                preferred_name=draft.identity.preferred_name,
                language=draft.identity.language,
                conversational_style=draft.preferences.conversational_style,
                positive_topics=list(draft.preferences.positive_topics),
                sensitive_topics=list(draft.sensitive_topics),
                never_do=list(draft.preferences.never_do),
                data_origin=self._data_origin(row, draft),
                source_id=profile_source.id if profile_source is not None else None,
                is_active=True,
            ),
            commit=False,
        )
        row.subject_person_id = subject.id
        row.profile_id = profile.id

    @staticmethod
    def _clear_consent(row: OnboardingSession) -> None:
        row.consent_confirmed = False
        row.consent_confirmed_by = None
        row.consent_confirmed_at = None
        row.consent_scope = None
        row.consent_payload_fingerprint = None

    @classmethod
    def draft_fingerprint(cls, draft: OnboardingDraft) -> str:
        """Stable digest of the canonical draft payload.

        ``sort_keys`` canonicalizes object key order while list order stays
        meaningful, so any content change produces a different fingerprint.
        """
        canonical = json.dumps(
            draft.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _data_origin(row: OnboardingSession, draft: OnboardingDraft) -> str:
        reviewer = draft.reviewer.name
        if draft.reviewer.relationship:
            reviewer = f"{reviewer} ({draft.reviewer.relationship})"
        contributors = ", ".join(item.name for item in draft.contributors)
        return (
            f"Family-led onboarding {row.id} administered by {reviewer}. "
            f"Permitted contributors: {contributors}. "
            f"Consent recorded by {row.consent_confirmed_by}: {row.consent_scope}"
        )

    async def _require(
        self, session_id: uuid.UUID, *, for_update: bool = False
    ) -> OnboardingSession:
        if for_update:
            row = await self._onboarding.get_for_update(session_id)
        else:
            row = await self._onboarding.get(session_id)
        if row is None:
            raise OnboardingNotFoundError("Onboarding session not found")
        return row

    async def _require_draft(
        self, session_id: uuid.UUID, *, for_update: bool = False
    ) -> OnboardingSession:
        row = await self._require(session_id, for_update=for_update)
        if row.status is not OnboardingStatus.draft:
            raise OnboardingNotDraftError(
                "Activated onboarding sessions cannot be edited or reactivated"
            )
        return row
