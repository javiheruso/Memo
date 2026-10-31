import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import EmbedderDep, get_db
from app.domain.enums import ClaimGroupStatus, KnowledgeGapStatus
from app.embeddings.memory_text import memory_search_text
from app.models import (
    Claim,
    ClaimGroup,
    ClaimGroupContradiction,
    ConversationProfile,
    Event,
    Fact,
    KnowledgeGap,
    Memory,
    Person,
    Relationship,
    Source,
)
from app.repositories.claim import ClaimRepository
from app.repositories.claim_group import ClaimGroupRepository
from app.repositories.claim_group_contradiction import (
    ClaimGroupContradictionRepository,
)
from app.repositories.event import EventRepository
from app.repositories.fact import FactRepository
from app.repositories.knowledge_gap import KnowledgeGapRepository
from app.repositories.media import MediaRepository
from app.repositories.memory import MemoryRepository
from app.repositories.person import PersonRepository
from app.repositories.profile import ConversationProfileRepository
from app.repositories.relationship import RelationshipRepository
from app.repositories.source import SourceRepository
from app.schemas.claim import (
    ClaimGroupContradictionRead,
    ClaimGroupDetail,
    ClaimGroupMergeRequest,
    ClaimGroupMergeSuggestionRead,
    ClaimGroupRead,
    ClaimGroupResolve,
    ClaimRead,
)
from app.schemas.event import EventCreate, EventRead
from app.schemas.fact import FactCreate, FactRead
from app.schemas.knowledge_gap import KnowledgeGapRead, KnowledgeGapStatusUpdate
from app.schemas.materialization import FactMaterializeRequest, MemoryMaterializeRequest
from app.schemas.media import MediaCreate, MediaRead
from app.schemas.memory import MemoryBackfillRead, MemoryCreate, MemoryRead
from app.schemas.person import PersonCreate, PersonPrimaryMediaUpdate, PersonRead
from app.schemas.profile import ConversationProfileCreate, ConversationProfileRead
from app.schemas.relationship import RelationshipCreate, RelationshipRead
from app.schemas.seed import SeedSummaryRead
from app.schemas.source import SourceCreate, SourceRead
from app.security import require_admin_api
from app.services.claim_group_merge import (
    ClaimGroupMergeError,
    ClaimGroupMergeService,
    UnknownClaimGroupMergeError,
)
from app.services.materialization import (
    MaterializationError,
    MaterializationService,
    UnknownSourceError,
    UnknownSourceRefError,
)
from app.services.memory_embeddings import MemoryEmbeddingBackfillService
from app.services.seed import AlreadySeededError, seed_if_empty

router = APIRouter(
    prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin_api)]
)

DbSession = Annotated[AsyncSession, Depends(get_db)]


def _materialization_http_error(exc: MaterializationError) -> HTTPException:
    if isinstance(exc, UnknownSourceError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, UnknownSourceRefError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


def _claim_group_merge_http_error(exc: ClaimGroupMergeError) -> HTTPException:
    if isinstance(exc, UnknownClaimGroupMergeError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


@router.post("/sources", response_model=SourceRead, status_code=201)
async def create_source(db: DbSession, payload: SourceCreate) -> Source:
    return await SourceRepository(db).create(payload)


@router.get("/sources", response_model=list[SourceRead])
async def list_sources(
    db: DbSession, limit: int = Query(default=100, ge=1, le=500)
) -> list[Source]:
    return await SourceRepository(db).list(limit=limit)


@router.post("/profiles", response_model=ConversationProfileRead, status_code=201)
async def create_profile(
    db: DbSession, payload: ConversationProfileCreate
) -> ConversationProfile:
    """Create the conversational profile for a person.

    The profile is the durable seam the conversation pipeline reads to decide
    which topics must never be initiated. Only one profile may be active at a
    time; activating a new one deactivates the previous profile in the same
    transaction.
    """
    if await PersonRepository(db).get(payload.person_id) is None:
        raise HTTPException(status_code=404, detail="Person not found")
    if (
        payload.source_id is not None
        and await SourceRepository(db).get(payload.source_id) is None
    ):
        raise HTTPException(status_code=404, detail="Source not found")
    profiles = ConversationProfileRepository(db)
    if await profiles.get_by_person(payload.person_id) is not None:
        raise HTTPException(
            status_code=409, detail="A profile already exists for this person"
        )
    return await profiles.create(payload)


@router.get("/profiles", response_model=list[ConversationProfileRead])
async def list_profiles(
    db: DbSession, limit: int = Query(default=100, ge=1, le=500)
) -> list[ConversationProfile]:
    return await ConversationProfileRepository(db).list(limit=limit)


@router.get("/profiles/active", response_model=ConversationProfileRead)
async def get_active_profile(db: DbSession) -> ConversationProfile:
    profile = await ConversationProfileRepository(db).get_active()
    if profile is None:
        raise HTTPException(status_code=404, detail="No active profile")
    return profile


@router.post(
    "/profiles/{profile_id}/activate", response_model=ConversationProfileRead
)
async def activate_profile(
    db: DbSession, profile_id: uuid.UUID
) -> ConversationProfile:
    profile = await ConversationProfileRepository(db).activate(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.post("/persons", response_model=PersonRead, status_code=201)
async def create_person(db: DbSession, payload: PersonCreate) -> Person:
    return await PersonRepository(db).create(payload)


@router.get("/persons", response_model=list[PersonRead])
async def list_persons(
    db: DbSession, limit: int = Query(default=100, ge=1, le=500)
) -> list[Person]:
    return await PersonRepository(db).list(limit=limit)


@router.get("/persons/search", response_model=list[PersonRead])
async def search_persons(
    db: DbSession, name: str = Query(min_length=1, max_length=255)
) -> list[Person]:
    return await PersonRepository(db).search_by_name(name)


@router.get("/persons/{person_id}", response_model=PersonRead)
async def get_person(db: DbSession, person_id: uuid.UUID) -> Person:
    person = await PersonRepository(db).get(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@router.patch("/persons/{person_id}/primary-media", response_model=PersonRead)
async def set_person_primary_media(
    db: DbSession, person_id: uuid.UUID, payload: PersonPrimaryMediaUpdate
) -> Person:
    """Set (or clear, with ``null``) a person's primary media row.

    The media row must exist but does not need a ``media_people`` link: the
    photo-hint retrieval tier falls back to ``persons.primary_media_id``, so a
    caregiver can attach a stored photograph without re-typing the
    person-to-media association.
    """
    if (
        payload.media_id is not None
        and await MediaRepository(db).get(payload.media_id) is None
    ):
        raise HTTPException(status_code=404, detail="Media not found")
    person = await PersonRepository(db).set_primary_media(person_id, payload.media_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@router.post("/relationships", response_model=RelationshipRead, status_code=201)
async def create_relationship(db: DbSession, payload: RelationshipCreate) -> Relationship:
    try:
        return await RelationshipRepository(db).create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/relationships", response_model=list[RelationshipRead])
async def list_relationships(
    db: DbSession, limit: int = Query(default=100, ge=1, le=500)
) -> list[Relationship]:
    return await RelationshipRepository(db).list(limit=limit)


@router.post("/facts", response_model=FactRead, status_code=201)
async def create_fact(db: DbSession, payload: FactCreate) -> Fact:
    return await FactRepository(db).create(payload)


@router.get("/facts", response_model=list[FactRead])
async def list_facts(
    db: DbSession, limit: int = Query(default=100, ge=1, le=500)
) -> list[Fact]:
    return await FactRepository(db).list(limit=limit)


@router.post("/memories", response_model=MemoryRead, status_code=201)
async def create_memory(
    db: DbSession, payload: MemoryCreate, embedder: EmbedderDep
) -> Memory:
    repo = MemoryRepository(db)
    embedding = None
    if repo.supports_vector_search:
        embedding = await embedder.embed(
            memory_search_text(title=payload.title, narrative=payload.narrative)
        )
    return await repo.create(payload, embedding=embedding)


@router.post("/memories/backfill", response_model=MemoryBackfillRead)
async def backfill_memory_embeddings(
    db: DbSession, embedder: EmbedderDep
) -> MemoryBackfillRead:
    repo = MemoryRepository(db)
    if not repo.supports_vector_search:
        raise HTTPException(
            status_code=400,
            detail="embedding backfill requires a PostgreSQL dialect",
        )
    processed = await MemoryEmbeddingBackfillService(
        embedder=embedder, memory_repo=repo
    ).backfill()
    return MemoryBackfillRead(processed=processed)


@router.get("/memories", response_model=list[MemoryRead])
async def list_memories(
    db: DbSession, limit: int = Query(default=100, ge=1, le=500)
) -> list[Memory]:
    return await MemoryRepository(db).list(limit=limit)


@router.post("/events", response_model=EventRead, status_code=201)
async def create_event(db: DbSession, payload: EventCreate) -> Event:
    return await EventRepository(db).create(payload)


@router.get("/events", response_model=list[EventRead])
async def list_events(
    db: DbSession, limit: int = Query(default=100, ge=1, le=500)
) -> list[Event]:
    return await EventRepository(db).list(limit=limit)


@router.get("/events/{event_id}", response_model=EventRead)
async def get_event(db: DbSession, event_id: uuid.UUID) -> Event:
    event = await EventRepository(db).get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.post("/media", response_model=MediaRead, status_code=201)
async def create_media(db: DbSession, payload: MediaCreate):
    """Create a metadata-only media row. No upload, no image bytes.

    ``description`` and the linked ``people`` are stored metadata provided by a
    caregiver; the row is never interpreted as image content here.
    """
    return await MediaRepository(db).create(payload)


@router.get("/media", response_model=list[MediaRead])
async def list_media(
    db: DbSession, limit: int = Query(default=100, ge=1, le=500)
):
    return await MediaRepository(db).list(limit=limit)


@router.get("/claims", response_model=list[ClaimRead])
async def list_claims(
    db: DbSession, limit: int = Query(default=100, ge=1, le=500)
) -> list[Claim]:
    return await ClaimRepository(db).list_recent(limit=limit)


@router.get("/claim-groups", response_model=list[ClaimGroupRead])
async def list_claim_groups(
    db: DbSession,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    needs_review: bool | None = Query(default=None),
    status: ClaimGroupStatus | None = Query(default=None),
) -> list[ClaimGroup]:
    return await ClaimGroupRepository(db).list_recent(
        limit=limit, offset=offset, needs_review=needs_review, status=status
    )


@router.get("/claim-groups/{group_id}", response_model=ClaimGroupDetail)
async def get_claim_group(db: DbSession, group_id: uuid.UUID) -> ClaimGroup:
    group = await ClaimGroupRepository(db).get_with_members(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Claim group not found")
    return group


@router.patch("/claim-groups/{group_id}", response_model=ClaimGroupDetail)
async def resolve_claim_group(
    db: DbSession, group_id: uuid.UUID, payload: ClaimGroupResolve
) -> ClaimGroup:
    """Apply a manual review state to a claim group.

    Resolved/ignored groups leave the review queue and are no longer re-flagged
    by contradiction detection. The group's claims and contradiction evidence
    are never modified.
    """
    repo = ClaimGroupRepository(db)
    if await repo.set_status(
        group_id, status=payload.status, resolution_note=payload.resolution_note
    ) is None:
        raise HTTPException(status_code=404, detail="Claim group not found")
    detail = await repo.get_with_members(group_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Claim group not found")
    return detail


@router.get("/claim-groups/{group_id}/merge-suggestions",
            response_model=list[ClaimGroupMergeSuggestionRead])
async def suggest_claim_group_merges(
    db: DbSession,
    group_id: uuid.UUID,
    limit: int = Query(default=200, ge=1, le=500),
) -> list[ClaimGroupMergeSuggestionRead]:
    """Deterministic merge targets for a claim group.

    Ranks the other active groups by shared content tokens so a reviewer sees
    the most plausible duplicate wording first. Pure token overlap — no
    semantics or LLM.
    """
    try:
        return await ClaimGroupMergeService(db).suggest(group_id, limit=limit)
    except ClaimGroupMergeError as exc:
        raise _claim_group_merge_http_error(exc) from exc


@router.post("/claim-groups/{group_id}/merge", response_model=ClaimGroupDetail)
async def merge_claim_group(
    db: DbSession, group_id: uuid.UUID, payload: ClaimGroupMergeRequest
) -> ClaimGroup:
    """Explicitly fold a claim group into a surviving group.

    The absorbed group's member claims move unchanged to the survivor, its
    contradiction evidence transfers/folds into the survivor, its wording is
    remembered so future claims route to the survivor, and only then is the
    group row deleted. Self-merges, ignored/inactive endpoints, groups that
    already produced a materialization, and opposite-polarity merges without
    ``confirm_opposite_polarity`` are refused. No promotion ever happens.
    """
    try:
        return await ClaimGroupMergeService(db).merge(
            group_id,
            payload.target_group_id,
            confirm_opposite_polarity=payload.confirm_opposite_polarity,
        )
    except ClaimGroupMergeError as exc:
        raise _claim_group_merge_http_error(exc) from exc


@router.get("/claim-contradictions", response_model=list[ClaimGroupContradictionRead])
async def list_claim_contradictions(
    db: DbSession,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[ClaimGroupContradiction]:
    return await ClaimGroupContradictionRepository(db).list_recent(
        limit=limit, offset=offset
    )


@router.get("/knowledge-gaps", response_model=list[KnowledgeGapRead])
async def list_knowledge_gaps(
    db: DbSession,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: KnowledgeGapStatus | None = Query(default=None),
) -> list[KnowledgeGap]:
    return await KnowledgeGapRepository(db).list_recent(
        limit=limit, offset=offset, status=status
    )


@router.patch("/knowledge-gaps/{gap_id}", response_model=KnowledgeGapRead)
async def update_knowledge_gap(
    db: DbSession, gap_id: uuid.UUID, payload: KnowledgeGapStatusUpdate
) -> KnowledgeGap:
    """Apply a manual status transition to a knowledge gap.

    Uses the existing ``knowledge_gap_status`` model. Resolving may link the
    gap to the person or fact that closed it; nothing is promoted here — the
    write only changes the gap's own status.
    """
    gap = await KnowledgeGapRepository(db).set_status(
        gap_id,
        status=payload.status,
        resolved_person_id=payload.resolved_person_id,
        resolved_fact_id=payload.resolved_fact_id,
    )
    if gap is None:
        raise HTTPException(status_code=404, detail="Knowledge gap not found")
    return gap


# --------------------------------------------------------------------------
# Explicit admin-only materialization (claim group / knowledge gap -> fact/memory)
# --------------------------------------------------------------------------


@router.post(
    "/claim-groups/{group_id}/materialize-fact",
    response_model=FactRead,
    status_code=201,
)
async def materialize_claim_group_fact(
    db: DbSession, group_id: uuid.UUID, payload: FactMaterializeRequest
) -> Fact:
    """Explicitly materialize a reviewed claim group into a verified-fact row.

    Creates the fact, records an append-only provenance row, and transitions
    the group to ``resolved`` so it leaves the review queue. Member claims and
    contradiction evidence are never modified, and ignored or already
    materialized groups are refused.
    """
    try:
        return await MaterializationService(db).materialize_claim_group_fact(
            group_id, payload
        )
    except MaterializationError as exc:
        raise _materialization_http_error(exc) from exc


@router.post(
    "/claim-groups/{group_id}/materialize-memory",
    response_model=MemoryRead,
    status_code=201,
)
async def materialize_claim_group_memory(
    db: DbSession,
    group_id: uuid.UUID,
    payload: MemoryMaterializeRequest,
    embedder: EmbedderDep,
) -> Memory:
    """Explicitly materialize a reviewed claim group into a memory.

    On PostgreSQL the memory is embedded with the same text and embedder as the
    manual memory creation path; on any other dialect the embedding stays
    ``NULL``. The group transitions to ``resolved`` and gets an append-only
    provenance row.
    """
    try:
        return await MaterializationService(
            db, embedder=embedder
        ).materialize_claim_group_memory(group_id, payload)
    except MaterializationError as exc:
        raise _materialization_http_error(exc) from exc


@router.post(
    "/knowledge-gaps/{gap_id}/materialize-fact",
    response_model=FactRead,
    status_code=201,
)
async def materialize_gap_fact(
    db: DbSession, gap_id: uuid.UUID, payload: FactMaterializeRequest
) -> Fact:
    """Explicitly materialize a knowledge gap into a verified-fact row.

    Records provenance and transitions the gap to ``resolved`` with the
    resolved-fact/person links; the gap's occurrences stay untouched.
    """
    try:
        return await MaterializationService(db).materialize_gap_fact(gap_id, payload)
    except MaterializationError as exc:
        raise _materialization_http_error(exc) from exc


@router.post(
    "/knowledge-gaps/{gap_id}/materialize-memory",
    response_model=MemoryRead,
    status_code=201,
)
async def materialize_gap_memory(
    db: DbSession,
    gap_id: uuid.UUID,
    payload: MemoryMaterializeRequest,
    embedder: EmbedderDep,
) -> Memory:
    """Explicitly materialize a knowledge gap into a memory.

    On PostgreSQL the memory is embedded like the manual creation path; on any
    other dialect the embedding stays ``NULL``. The gap transitions to
    ``resolved`` and gets an append-only provenance row.
    """
    try:
        return await MaterializationService(
            db, embedder=embedder
        ).materialize_gap_memory(gap_id, payload)
    except MaterializationError as exc:
        raise _materialization_http_error(exc) from exc


# --------------------------------------------------------------------------
# Load example data (fictional universe)
# --------------------------------------------------------------------------


@router.post("/seed", response_model=SeedSummaryRead)
async def seed_example_data(db: DbSession) -> SeedSummaryRead:
    """Load the fictional example universe into an empty database.

    Idempotent by refusal: when any person already exists the call is rejected
    with ``409`` instead of duplicating data. The response reports how many
    persons, relationships, facts, memories and media rows were created.
    """
    try:
        universe = await seed_if_empty(db)
    except AlreadySeededError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    created = universe.created
    return SeedSummaryRead(
        persons=created.persons,
        relationships=created.relationships,
        facts=created.facts,
        memories=created.memories,
        media=created.media,
    )
