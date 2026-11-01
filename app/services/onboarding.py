"""Admin/family-led onboarding workflow.

The service owns the draft lifecycle and the explicit activation step. Drafts
are resumable JSON validated by the onboarding schemas; activation is gated on
a recorded operator consent and materializes the real rows through the same
repositories the admin API already uses, so trust and provenance rules are not
bypassed.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

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


class OnboardingValidationError(OnboardingError):
    """The draft does not yet contain a complete first-day package."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


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
        row = OnboardingSession(
            status=OnboardingStatus.draft,
            payload=(draft or OnboardingDraft()).model_dump(mode="json"),
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
        row = await self._require_draft(session_id)
        row.payload = draft.model_dump(mode="json")
        await self._session.commit()
        return await self._require(session_id)

    async def record_consent(
        self, session_id: uuid.UUID, *, confirmed_by: str, scope: str
    ) -> OnboardingSession:
        """Record explicit operator consent; never activates in the same call."""
        row = await self._require_draft(session_id)
        row.consent_confirmed = True
        row.consent_confirmed_by = confirmed_by
        row.consent_confirmed_at = datetime.now(UTC)
        row.consent_scope = scope
        await self._session.commit()
        return await self._require(session_id)

    async def activate(self, session_id: uuid.UUID) -> OnboardingSession:
        row = await self._require_draft(session_id)
        if not row.consent_confirmed:
            raise OnboardingConsentRequiredError(
                "Explicit operator consent must be recorded before activation"
            )
        draft = self.draft_of(row)
        problems = self.validation_problems(draft)
        if problems:
            raise OnboardingValidationError(problems)

        try:
            await self._materialize(row, draft)
            row.status = OnboardingStatus.activated
            row.activated_at = datetime.now(UTC)
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        return await self._require(session_id)

    def draft_of(self, row: OnboardingSession) -> OnboardingDraft:
        return OnboardingDraft.model_validate(row.payload or {})

    def validation_problems(self, draft: OnboardingDraft) -> list[str]:
        """Return every reason the draft is not yet activatable."""
        problems: list[str] = []
        if not draft.identity.first_name:
            problems.append("identity.first_name is required")
        if not draft.reviewer.name:
            problems.append("reviewer.name is required")
        if not draft.contributors:
            problems.append("at least one permitted contributor is required")
        if not draft.people:
            problems.append("at least one person in the circle is required")
        if not draft.sensitive_topics:
            problems.append("at least one sensitive topic is required")
        for topic in draft.sensitive_topics:
            if not topic.handling:
                problems.append(
                    f"sensitive topic '{topic.topic}' needs handling guidance"
                )
        if not (draft.facts or draft.memories):
            problems.append("at least one fact or memory is required")
        if not (
            draft.preferences.positive_topics
            or draft.preferences.never_do
            or draft.preferences.conversational_style
        ):
            problems.append("at least one conversation preference is required")

        source_keys = {source.key for source in draft.sources}
        person_keys = {person.key for person in draft.people}
        for person in draft.people:
            if (
                person.source_key is not None
                and person.source_key not in source_keys
            ):
                problems.append(
                    f"person '{person.first_name}' references unknown source "
                    f"'{person.source_key}'"
                )
        for fact in draft.facts:
            if fact.source_key not in source_keys:
                problems.append(
                    f"fact '{fact.fact_type}' references unknown source "
                    f"'{fact.source_key}'"
                )
            if fact.subject_key != "subject" and fact.subject_key not in person_keys:
                problems.append(
                    f"fact '{fact.fact_type}' references unknown person "
                    f"'{fact.subject_key}'"
                )
        for memory in draft.memories:
            if memory.source_key not in source_keys:
                problems.append(
                    f"memory '{memory.title or memory.narrative[:30]}' references "
                    f"unknown source '{memory.source_key}'"
                )
            if len(memory.people_keys) != len(set(memory.people_keys)):
                problems.append(
                    f"memory '{memory.title or memory.narrative[:30]}' repeats "
                    "a person key"
                )
            for key in memory.people_keys:
                if key != "subject" and key not in person_keys:
                    problems.append(
                        f"memory '{memory.title or memory.narrative[:30]}' "
                        f"references unknown person '{key}'"
                    )
        return problems

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

    async def _require(self, session_id: uuid.UUID) -> OnboardingSession:
        row = await self._onboarding.get(session_id)
        if row is None:
            raise OnboardingNotFoundError("Onboarding session not found")
        return row

    async def _require_draft(self, session_id: uuid.UUID) -> OnboardingSession:
        row = await self._require(session_id)
        if row.status is not OnboardingStatus.draft:
            raise OnboardingNotDraftError(
                "Activated onboarding sessions cannot be edited or reactivated"
            )
        return row
