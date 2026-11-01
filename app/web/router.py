"""Server-rendered admin UI.

Minimal HTML surface over the existing admin services and repositories so a
human can operate the review queues (claim groups, knowledge gaps), the
materialization actions, the embeddings backfill, and the core entity CRUD
without hand-crafting API calls.

Every write path builds the same pydantic schemas and calls the same
repositories/services as the JSON admin API (``app/api/v1/endpoints/admin.py``),
so the existing guardrails (trust-model defaults, materialization guards,
status transitions, provenance) are preserved unchanged. This module adds no
auth and no separate frontend toolchain: pages are rendered by FastAPI/Jinja2
and forms post back to this same router.
"""

import os
import uuid
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import EmbedderDep, get_db
from app.domain.enums import (
    ClaimGroupStatus,
    FactStatus,
    KnowledgeGapStatus,
    MediaType,
    MemoryStatus,
    RelationshipStatus,
    RelationshipType,
    SensitivityLevel,
    SourceType,
)
from app.embeddings.memory_text import memory_search_text
from app.models import Fact, Memory, Person, Source
from app.repositories.claim_group import ClaimGroupRepository
from app.repositories.fact import FactRepository
from app.repositories.knowledge_gap import KnowledgeGapRepository
from app.repositories.media import MediaRepository
from app.repositories.memory import MemoryRepository, UnsupportedVectorSearch
from app.repositories.person import PersonRepository
from app.repositories.source import SourceRepository
from app.schemas.claim import ClaimGroupResolve
from app.schemas.fact import FactCreate
from app.schemas.knowledge_gap import KnowledgeGapStatusUpdate
from app.schemas.materialization import (
    FactMaterializeRequest,
    MemoryMaterializeRequest,
)
from app.schemas.media import MediaCreate
from app.schemas.memory import MemoryCreate
from app.schemas.onboarding import (
    OnboardingContributor,
    OnboardingFact,
    OnboardingIdentity,
    OnboardingMemory,
    OnboardingPerson,
    OnboardingPreferences,
    OnboardingReviewer,
    OnboardingSource,
)
from app.schemas.person import PersonCreate
from app.schemas.profile import SensitiveTopicCreate
from app.schemas.source import SourceCreate
from app.security import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    admin_auth_enabled,
    cookie_secure,
    issue_session_cookie,
    require_admin_ui,
    verify_password,
)
from app.services.claim_group_merge import (
    ClaimGroupMergeError,
    ClaimGroupMergeService,
    IgnoredClaimGroupMergeError,
    InactiveClaimGroupMergeError,
    MaterializedClaimGroupMergeError,
    OppositePolarityMergeError,
    SelfClaimGroupMergeError,
    UnknownClaimGroupMergeError,
)
from app.services.materialization import (
    DuplicateMaterializationError,
    IgnoredSourceError,
    MaterializationError,
    MaterializationService,
    UnknownSourceError,
    UnknownSourceRefError,
)
from app.services.memory_embeddings import MemoryEmbeddingBackfillService
from app.services.onboarding import (
    OnboardingConsentMismatchError,
    OnboardingConsentRequiredError,
    OnboardingDraftInvalidError,
    OnboardingError,
    OnboardingNotDraftError,
    OnboardingNotFoundError,
    OnboardingService,
    OnboardingValidationError,
)
from app.services.seed import AlreadySeededError, seed_if_empty

router = APIRouter(
    prefix="/admin",
    tags=["admin-ui"],
    include_in_schema=False,
    dependencies=[Depends(require_admin_ui)],
)

auth_router = APIRouter(
    prefix="/admin",
    tags=["admin-auth"],
    include_in_schema=False,
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

DbSession = Annotated[AsyncSession, Depends(get_db)]
ClaimGroupStatusQuery = Annotated[ClaimGroupStatus | None, Query()]
NeedsReviewQuery = Annotated[bool | None, Query()]
KnowledgeGapStatusQuery = Annotated[KnowledgeGapStatus | None, Query()]


def _ctx(request: Request, **extra) -> dict:
    """Template context with the flash-message query params."""
    ctx: dict = {
        "message": request.query_params.get("message"),
        "error": request.query_params.get("error"),
        "auth_enabled": admin_auth_enabled(),
    }
    ctx.update(extra)
    return ctx


def _back(path: str, *, message: str | None = None, error: str | None = None) -> RedirectResponse:
    query: dict[str, str] = {}
    if message:
        query["message"] = message
    if error:
        query["error"] = error
    url = f"{path}?{urlencode(query)}" if query else path
    return RedirectResponse(url=url, status_code=303)


def _error_text(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        loc = ".".join(str(part) for part in first.get("loc", ()))
        msg = str(first.get("msg", "invalid input"))
        return f"{loc}: {msg}" if loc else msg
    if isinstance(exc, SQLAlchemyError):
        detail = getattr(exc, "orig", exc)
        return f"Error de base de datos: {str(detail)[:300]}"
    if isinstance(exc, DuplicateMaterializationError):
        return "La fuente ya fue materializada en este tipo de destino."
    if isinstance(exc, IgnoredSourceError):
        return "La fuente está marcada como ignorada y no se puede materializar."
    if isinstance(exc, UnknownSourceRefError):
        return "La fuente de evidencia indicada no existe."
    if isinstance(exc, UnknownSourceError):
        return "El elemento de la cola de revisión no existe."
    if isinstance(exc, SelfClaimGroupMergeError):
        return "Un grupo de afirmaciones no se puede fusionar consigo mismo."
    if isinstance(exc, UnknownClaimGroupMergeError):
        return "Uno de los grupos de afirmaciones de la fusión no existe."
    if isinstance(exc, (IgnoredClaimGroupMergeError, InactiveClaimGroupMergeError)):
        return "Solo los grupos de afirmaciones activos y no ignorados se pueden fusionar."
    if isinstance(exc, MaterializedClaimGroupMergeError):
        return (
            "Un grupo de afirmaciones que ya generó una materialización no se puede "
            "fusionar."
        )
    if isinstance(exc, OppositePolarityMergeError):
        return (
            "La fusión combina una redacción negada con una no negada; confirma la "
            "fusión de polaridad opuesta para continuar."
        )
    if isinstance(exc, UnsupportedVectorSearch):
        return "Esta acción requiere una base de datos PostgreSQL (pgvector)."
    if isinstance(exc, OnboardingDraftInvalidError):
        return "El borrador no es válido: " + "; ".join(exc.problems)
    if isinstance(exc, OnboardingValidationError):
        return "Faltan datos para activar: " + "; ".join(exc.problems)
    if isinstance(exc, OnboardingConsentRequiredError):
        return "Registra el consentimiento explícito antes de activar el perfil."
    if isinstance(exc, OnboardingConsentMismatchError):
        return (
            "El borrador cambió después del consentimiento. Revísalo y registra "
            "el consentimiento de nuevo."
        )
    if isinstance(exc, OnboardingNotDraftError):
        return "Este onboarding ya está activado y no se puede editar."
    if isinstance(exc, OnboardingNotFoundError):
        return "Onboarding no encontrado."
    return str(exc)


def _text(form, key: str) -> str | None:
    raw = form.get(key)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _checked(form, key: str) -> bool:
    return key in form


def _person_links(form, prefix: str = "person") -> list[dict]:
    person_id = _text(form, f"{prefix}_id")
    if person_id is None:
        return []
    return [{"person_id": person_id, "role": _text(form, f"{prefix}_role")}]


def _person_create(form) -> PersonCreate:
    return PersonCreate(
        first_name=_text(form, "first_name"),
        last_name=_text(form, "last_name"),
        birth_date=_text(form, "birth_date"),
        is_deceased=_checked(form, "is_deceased"),
        description=_text(form, "description"),
        aliases=[],
    )


def _source_create(form) -> SourceCreate:
    return SourceCreate(
        source_type=_text(form, "source_type"),
        person_id=_text(form, "person_id"),
        description=_text(form, "description"),
        external_reference=_text(form, "external_reference"),
    )


def _fact_create(form) -> FactCreate:
    return FactCreate(
        subject_person_id=_text(form, "subject_person_id"),
        fact_type=_text(form, "fact_type"),
        value_text=_text(form, "value_text"),
        value_date=_text(form, "value_date"),
        place_text=_text(form, "place_text"),
        status=_text(form, "status") or FactStatus.pending.value,
        source_id=_text(form, "source_id"),
        notes=_text(form, "notes"),
    )


def _memory_create(form) -> MemoryCreate:
    return MemoryCreate(
        title=_text(form, "title"),
        narrative=_text(form, "narrative"),
        approx_period_text=_text(form, "approx_period_text"),
        place_text=_text(form, "place_text"),
        memory_status=_text(form, "memory_status") or MemoryStatus.candidate.value,
        sensitivity_level=_text(form, "sensitivity_level") or SensitivityLevel.low.value,
        source_id=_text(form, "source_id"),
        people=_person_links(form),
    )


def _media_create(form) -> MediaCreate:
    return MediaCreate(
        media_type=_text(form, "media_type"),
        storage_path=_text(form, "storage_path"),
        title=_text(form, "title"),
        description=_text(form, "description"),
        approx_date=_text(form, "approx_date"),
        place_text=_text(form, "place_text"),
        source_id=_text(form, "source_id"),
        people=[],
    )


def _fact_materialize(form) -> FactMaterializeRequest:
    return FactMaterializeRequest(
        subject_person_id=_text(form, "subject_person_id"),
        fact_type=_text(form, "fact_type"),
        value_text=_text(form, "value_text"),
        value_date=_text(form, "value_date"),
        place_text=_text(form, "place_text"),
        status=_text(form, "status") or FactStatus.pending.value,
        source_id=_text(form, "source_id"),
        notes=_text(form, "notes"),
    )


def _memory_materialize(form) -> MemoryMaterializeRequest:
    return MemoryMaterializeRequest(
        title=_text(form, "title"),
        narrative=_text(form, "narrative"),
        approx_period_text=_text(form, "approx_period_text"),
        place_text=_text(form, "place_text"),
        memory_status=_text(form, "memory_status") or MemoryStatus.candidate.value,
        sensitivity_level=_text(form, "sensitivity_level") or SensitivityLevel.low.value,
        source_id=_text(form, "source_id"),
        people=_person_links(form),
    )


def _claim_group_resolve(form) -> ClaimGroupResolve:
    return ClaimGroupResolve(
        status=_text(form, "status"),
        resolution_note=_text(form, "resolution_note"),
    )


def _gap_status_update(form) -> KnowledgeGapStatusUpdate:
    return KnowledgeGapStatusUpdate(
        status=_text(form, "status"),
        resolved_person_id=_text(form, "resolved_person_id"),
        resolved_fact_id=_text(form, "resolved_fact_id"),
    )


def _csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _form_list(form, key: str) -> list[str]:
    return [
        str(value).strip()
        for value in form.getlist(key)
        if str(value).strip()
    ]


async def _onboarding_row(db: AsyncSession, session_id: uuid.UUID):
    try:
        return await OnboardingService(db).get(session_id)
    except OnboardingNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Onboarding no encontrado") from exc


ONBOARDING_ENTRY_SECTIONS = {
    "contributors",
    "sources",
    "sensitive_topics",
    "people",
    "facts",
    "memories",
}


def _onboarding_entry(section: str, form):
    if section == "contributors":
        return OnboardingContributor(
            name=_text(form, "name") or "",
            relationship=_text(form, "relationship"),
            permission_note=_text(form, "permission_note"),
        )
    if section == "sources":
        return OnboardingSource(
            key=_text(form, "key") or "",
            source_type=_text(form, "source_type") or SourceType.family_member.value,
            description=_text(form, "description"),
        )
    if section == "sensitive_topics":
        return SensitiveTopicCreate(
            topic=_text(form, "topic") or "",
            handling=_text(form, "handling"),
            consent_review_required=_checked(form, "consent_review_required"),
            triggers=_csv(_text(form, "triggers")),
        )
    if section == "people":
        return OnboardingPerson(
            key=_text(form, "key") or "",
            first_name=_text(form, "first_name") or "",
            last_name=_text(form, "last_name"),
            relationship_type=(
                _text(form, "relationship_type") or RelationshipType.other.value
            ),
            certainty=(
                _text(form, "certainty") or RelationshipStatus.probable.value
            ),
            source_key=_text(form, "source_key"),
            notes=_text(form, "notes"),
        )
    if section == "facts":
        return OnboardingFact(
            subject_key=_text(form, "subject_key") or "subject",
            fact_type=_text(form, "fact_type") or "",
            value_text=_text(form, "value_text"),
            status=_text(form, "status") or FactStatus.probable.value,
            source_key=_text(form, "source_key") or "",
            notes=_text(form, "notes"),
        )
    if section == "memories":
        return OnboardingMemory(
            title=_text(form, "title"),
            narrative=_text(form, "narrative") or "",
            sensitivity_level=(
                _text(form, "sensitivity_level") or SensitivityLevel.low.value
            ),
            source_key=_text(form, "source_key") or "",
            people_keys=_form_list(form, "people_keys"),
        )
    raise ValueError(f"Unknown onboarding section {section}")


async def _count(db: AsyncSession, model, *conditions) -> int:
    stmt = select(func.count()).select_from(model)
    if conditions:
        stmt = stmt.where(*conditions)
    return (await db.execute(stmt)).scalar_one()


async def _person_names(db: AsyncSession) -> dict[uuid.UUID, str]:
    persons = await PersonRepository(db).list(limit=500)
    return {person.id: person.display_name for person in persons}


async def _source_labels(db: AsyncSession) -> dict[uuid.UUID, str]:
    sources = await SourceRepository(db).list(limit=500)
    return {
        source.id: source.description or source.source_type.value for source in sources
    }


# --------------------------------------------------------------------------
# Browser login / logout (auth is a no-op when MEMO_ADMIN_PASSWORD is unset)
# --------------------------------------------------------------------------


@auth_router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    if not admin_auth_enabled():
        return RedirectResponse(url="/admin", status_code=303)
    return templates.TemplateResponse(
        request,
        "login.html",
        {"error": request.query_params.get("error")},
    )


@auth_router.post("/login")
async def login(request: Request) -> RedirectResponse:
    if not admin_auth_enabled():
        return RedirectResponse(url="/admin", status_code=303)
    form = await request.form()
    submitted = form.get("password")
    if submitted is None or not verify_password(str(submitted)):
        return RedirectResponse(
            url="/admin/login?error=Contrase%C3%B1a%20incorrecta", status_code=303
        )
    response = RedirectResponse(url="/admin", status_code=303)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        issue_session_cookie(),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=cookie_secure(),
        path="/",
    )
    return response


@auth_router.post("/logout")
async def logout() -> RedirectResponse:
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response


# --------------------------------------------------------------------------
# Dashboard / nav
# --------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
async def dashboard(request: Request, db: DbSession) -> HTMLResponse:
    claim_groups = await ClaimGroupRepository(db).list_recent(limit=200)
    gaps = await KnowledgeGapRepository(db).list_recent(limit=200)
    open_gap_statuses = {
        KnowledgeGapStatus.observed,
        KnowledgeGapStatus.recurrent,
        KnowledgeGapStatus.family_review,
    }
    counts = {
        "claim_groups_needs_review": sum(1 for g in claim_groups if g.needs_review),
        "claim_groups_active": sum(
            1 for g in claim_groups if g.status == ClaimGroupStatus.active
        ),
        "gaps_open": sum(1 for g in gaps if g.status in open_gap_statuses),
        "persons": await _count(db, Person),
        "sources": await _count(db, Source),
        "facts": await _count(db, Fact),
        "memories": await _count(db, Memory),
    }
    return templates.TemplateResponse(request, "dashboard.html", _ctx(request, counts=counts))


@router.post("/seed")
async def seed_example(request: Request, db: DbSession) -> RedirectResponse:
    """Load the fictional example universe from the dashboard button.

    Refuses (with a friendly notice) when any person already exists, so the
    example data is only ever loaded into an empty database.
    """
    try:
        universe = await seed_if_empty(db)
    except AlreadySeededError:
        return _back(
            "/admin",
            message="Los datos de ejemplo ya se han cargado anteriormente y no se han duplicado.",
        )
    created = universe.created
    media_label = "recurso multimedia" if created.media == 1 else "recursos multimedia"
    return _back(
        "/admin",
        message=(
            f"Se han cargado los datos de ejemplo: {created.persons} personas, "
            f"{created.relationships} relaciones, {created.facts} hechos, "
            f"{created.memories} recuerdos y {created.media} {media_label}."
        ),
    )


# --------------------------------------------------------------------------
# Admin/family-led onboarding
# --------------------------------------------------------------------------


@router.get("/onboarding", response_class=HTMLResponse)
async def onboarding_list_page(request: Request, db: DbSession) -> HTMLResponse:
    service = OnboardingService(db)
    items = []
    invalid: list[str] = []
    for row in await service.list(limit=200):
        try:
            draft = service.draft_of(row)
        except OnboardingDraftInvalidError as exc:
            invalid.append(f"{row.id}: {'; '.join(exc.problems)}")
            continue
        items.append(
            {
                "row": row,
                "draft": draft,
                "progress": service.progress(row, draft),
            }
        )
    extra: dict = {}
    if invalid:
        extra["error"] = (
            "Hay borradores guardados con datos inválidos: " + "; ".join(invalid)
        )
    return templates.TemplateResponse(
        request, "onboarding_list.html", _ctx(request, items=items, **extra)
    )


@router.post("/onboarding")
async def onboarding_create(request: Request, db: DbSession) -> RedirectResponse:
    row = await OnboardingService(db).create()
    return _back(
        f"/admin/onboarding/{row.id}",
        message="Se ha creado un borrador de onboarding.",
    )


@router.get("/onboarding/{session_id}", response_class=HTMLResponse)
async def onboarding_detail(
    request: Request, db: DbSession, session_id: uuid.UUID
) -> HTMLResponse:
    service = OnboardingService(db)
    row = await _onboarding_row(db, session_id)
    try:
        draft = service.draft_of(row)
    except OnboardingDraftInvalidError as exc:
        return _back("/admin/onboarding", error=_error_text(exc))
    return templates.TemplateResponse(
        request,
        "onboarding_detail.html",
        _ctx(
            request,
            row=row,
            draft=draft,
            progress=service.progress(row, draft),
            subject_label=(
                draft.identity.preferred_name
                or draft.identity.first_name
                or "Sin identificar"
            ),
            source_options=[source.key for source in draft.sources],
            person_options=[
                {"key": "subject", "label": draft.identity.first_name or "La persona"}
            ]
            + [
                {"key": person.key, "label": person.first_name}
                for person in draft.people
            ],
            source_type_options=SourceType,
            relationship_type_options=RelationshipType,
            relationship_status_options=RelationshipStatus,
            fact_status_options=FactStatus,
            memory_status_options=MemoryStatus,
            sensitivity_options=SensitivityLevel,
        ),
    )


@router.post("/onboarding/{session_id}/identity")
async def onboarding_identity(
    request: Request, db: DbSession, session_id: uuid.UUID
) -> RedirectResponse:
    form = await request.form()
    service = OnboardingService(db)
    row = await _onboarding_row(db, session_id)
    try:
        draft = service.draft_of(row)
        draft.identity = OnboardingIdentity(
            first_name=_text(form, "first_name") or "",
            last_name=_text(form, "last_name"),
            preferred_name=_text(form, "preferred_name"),
            language=_text(form, "language"),
        )
        draft.reviewer = OnboardingReviewer(
            name=_text(form, "reviewer_name") or "",
            relationship=_text(form, "reviewer_relationship"),
        )
        await service.save_draft(session_id, draft)
    except (ValidationError, OnboardingError) as exc:
        return _back(f"/admin/onboarding/{session_id}", error=_error_text(exc))
    return _back(
        f"/admin/onboarding/{session_id}",
        message="Identidad y revisor guardados.",
    )


@router.post("/onboarding/{session_id}/preferences")
async def onboarding_preferences(
    request: Request, db: DbSession, session_id: uuid.UUID
) -> RedirectResponse:
    form = await request.form()
    service = OnboardingService(db)
    row = await _onboarding_row(db, session_id)
    try:
        draft = service.draft_of(row)
        draft.preferences = OnboardingPreferences(
            positive_topics=_csv(_text(form, "positive_topics")),
            never_do=_csv(_text(form, "never_do")),
            conversational_style=_text(form, "conversational_style"),
        )
        await service.save_draft(session_id, draft)
    except (ValidationError, OnboardingError) as exc:
        return _back(f"/admin/onboarding/{session_id}", error=_error_text(exc))
    return _back(
        f"/admin/onboarding/{session_id}",
        message="Preferencias guardadas.",
    )


@router.post("/onboarding/{session_id}/entries/{section}")
async def onboarding_add_entry(
    request: Request, db: DbSession, session_id: uuid.UUID, section: str
) -> RedirectResponse:
    if section not in ONBOARDING_ENTRY_SECTIONS:
        raise HTTPException(status_code=404, detail="Sección de onboarding no válida")
    form = await request.form()
    service = OnboardingService(db)
    row = await _onboarding_row(db, session_id)
    try:
        draft = service.draft_of(row)
        getattr(draft, section).append(_onboarding_entry(section, form))
        await service.save_draft(session_id, draft)
    except (ValidationError, OnboardingError) as exc:
        return _back(f"/admin/onboarding/{session_id}", error=_error_text(exc))
    return _back(
        f"/admin/onboarding/{session_id}",
        message="Elemento añadido al borrador.",
    )


@router.post("/onboarding/{session_id}/entries/{section}/{index}/remove")
async def onboarding_remove_entry(
    request: Request,
    db: DbSession,
    session_id: uuid.UUID,
    section: str,
    index: int,
) -> RedirectResponse:
    if section not in ONBOARDING_ENTRY_SECTIONS:
        raise HTTPException(status_code=404, detail="Sección de onboarding no válida")
    service = OnboardingService(db)
    row = await _onboarding_row(db, session_id)
    try:
        draft = service.draft_of(row)
        items = getattr(draft, section)
        if index < 0 or index >= len(items):
            return _back(
                f"/admin/onboarding/{session_id}",
                error="El elemento ya no existe en el borrador.",
            )
        items.pop(index)
        await service.save_draft(session_id, draft)
    except (ValidationError, OnboardingError) as exc:
        return _back(f"/admin/onboarding/{session_id}", error=_error_text(exc))
    return _back(
        f"/admin/onboarding/{session_id}",
        message="Elemento eliminado del borrador.",
    )


@router.post("/onboarding/{session_id}/consent")
async def onboarding_consent(
    request: Request, db: DbSession, session_id: uuid.UUID
) -> RedirectResponse:
    form = await request.form()
    if not _checked(form, "confirmed"):
        return _back(
            f"/admin/onboarding/{session_id}",
            error="Marca la confirmación explícita antes de registrar el consentimiento.",
        )
    confirmed_by = _text(form, "confirmed_by")
    scope = _text(form, "scope")
    if confirmed_by is None or scope is None:
        return _back(
            f"/admin/onboarding/{session_id}",
            error="Indica quién confirma y el alcance del consentimiento.",
        )
    try:
        await OnboardingService(db).record_consent(
            session_id, confirmed_by=confirmed_by, scope=scope
        )
    except OnboardingError as exc:
        return _back(f"/admin/onboarding/{session_id}", error=_error_text(exc))
    return _back(
        f"/admin/onboarding/{session_id}",
        message="Consentimiento explícito registrado.",
    )


@router.post("/onboarding/{session_id}/activate")
async def onboarding_activate(
    db: DbSession, session_id: uuid.UUID, embedder: EmbedderDep
) -> RedirectResponse:
    try:
        await OnboardingService(db, embedder=embedder).activate(session_id)
    except OnboardingError as exc:
        return _back(f"/admin/onboarding/{session_id}", error=_error_text(exc))
    return _back(
        f"/admin/onboarding/{session_id}",
        message="Perfil conversacional activado.",
    )


# --------------------------------------------------------------------------
# Claim-group review queue
# --------------------------------------------------------------------------


@router.get("/claim-groups", response_class=HTMLResponse)
async def claim_groups_page(
    request: Request,
    db: DbSession,
    status: ClaimGroupStatusQuery = None,
    needs_review: NeedsReviewQuery = None,
) -> HTMLResponse:
    groups = await ClaimGroupRepository(db).list_recent(
        limit=200, needs_review=needs_review, status=status
    )
    return templates.TemplateResponse(
        request,
        "claim_groups.html",
        _ctx(
            request,
            groups=groups,
            filter_status=status,
            filter_needs_review=needs_review,
            status_options=ClaimGroupStatus,
        ),
    )


@router.get("/claim-groups/{group_id}", response_class=HTMLResponse)
async def claim_group_detail(request: Request, db: DbSession, group_id: uuid.UUID) -> HTMLResponse:
    group = await ClaimGroupRepository(db).get_with_members(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="Grupo de afirmaciones no encontrado")
    persons = await PersonRepository(db).list(limit=500)
    sources = await SourceRepository(db).list(limit=500)
    claims = group.claims
    if claims:
        earliest = min(claims, key=lambda claim: (claim.first_seen_at, claim.created_at, claim.id))
        default_text = earliest.original_text
    else:
        default_text = group.canonical_text
    mergeable = group.status is ClaimGroupStatus.active
    merge_candidates = (
        await ClaimGroupMergeService(db).suggest(group.id) if mergeable else []
    )
    return templates.TemplateResponse(
        request,
        "claim_group_detail.html",
        _ctx(
            request,
            group=group,
            persons=persons,
            sources=sources,
            default_text=default_text,
            mergeable=mergeable,
            merge_candidates=merge_candidates,
            status_options=ClaimGroupStatus,
            fact_status_options=FactStatus,
            memory_status_options=MemoryStatus,
            sensitivity_options=SensitivityLevel,
        ),
    )


@router.post("/claim-groups/{group_id}/status")
async def claim_group_set_status(
    request: Request, db: DbSession, group_id: uuid.UUID
) -> RedirectResponse:
    detail_url = f"/admin/claim-groups/{group_id}"
    form = await request.form()
    try:
        payload = _claim_group_resolve(form)
    except ValidationError as exc:
        return _back(detail_url, error=_error_text(exc))
    updated = await ClaimGroupRepository(db).set_status(
        group_id, status=payload.status, resolution_note=payload.resolution_note
    )
    if updated is None:
        return _back(detail_url, error="Grupo de afirmaciones no encontrado")
    return _back(
        detail_url,
        message=f"El grupo de afirmaciones se ha marcado como {payload.status.value}",
    )


@router.post("/claim-groups/{group_id}/merge")
async def claim_group_merge(
    request: Request, db: DbSession, group_id: uuid.UUID
) -> RedirectResponse:
    detail_url = f"/admin/claim-groups/{group_id}"
    form = await request.form()
    raw_target = _text(form, "target_group_id")
    if raw_target is None:
        return _back(detail_url, error="Selecciona un grupo de destino superviviente")
    try:
        target_group_id = uuid.UUID(raw_target)
    except ValueError:
        return _back(detail_url, error="El identificador del grupo de destino no es válido")
    try:
        survivor = await ClaimGroupMergeService(db).merge(
            group_id,
            target_group_id,
            confirm_opposite_polarity=_checked(form, "confirm_opposite_polarity"),
        )
    except (ClaimGroupMergeError, SQLAlchemyError) as exc:
        return _back(detail_url, error=_error_text(exc))
    return _back(
        f"/admin/claim-groups/{survivor.id}",
        message=f"El grupo de afirmaciones {group_id} se ha fusionado en {survivor.id}",
    )


@router.post("/claim-groups/{group_id}/materialize-fact")
async def claim_group_materialize_fact(
    request: Request, db: DbSession, group_id: uuid.UUID
) -> RedirectResponse:
    detail_url = f"/admin/claim-groups/{group_id}"
    form = await request.form()
    try:
        payload = _fact_materialize(form)
        fact = await MaterializationService(db).materialize_claim_group_fact(
            group_id, payload
        )
    except (ValidationError, SQLAlchemyError, MaterializationError) as exc:
        return _back(detail_url, error=_error_text(exc))
    return _back(detail_url, message=f"Se ha materializado como hecho {fact.id}")


@router.post("/claim-groups/{group_id}/materialize-memory")
async def claim_group_materialize_memory(
    request: Request,
    db: DbSession,
    group_id: uuid.UUID,
    embedder: EmbedderDep,
) -> RedirectResponse:
    detail_url = f"/admin/claim-groups/{group_id}"
    form = await request.form()
    try:
        payload = _memory_materialize(form)
        memory = await MaterializationService(
            db, embedder=embedder
        ).materialize_claim_group_memory(group_id, payload)
    except (ValidationError, SQLAlchemyError, MaterializationError) as exc:
        return _back(detail_url, error=_error_text(exc))
    return _back(detail_url, message=f"Se ha materializado como recuerdo {memory.id}")


# --------------------------------------------------------------------------
# Knowledge-gap review queue
# --------------------------------------------------------------------------


@router.get("/knowledge-gaps", response_class=HTMLResponse)
async def knowledge_gaps_page(
    request: Request,
    db: DbSession,
    status: KnowledgeGapStatusQuery = None,
) -> HTMLResponse:
    gaps = await KnowledgeGapRepository(db).list_recent(limit=200, status=status)
    return templates.TemplateResponse(
        request,
        "knowledge_gaps.html",
        _ctx(request, gaps=gaps, filter_status=status, status_options=KnowledgeGapStatus),
    )


@router.get("/knowledge-gaps/{gap_id}", response_class=HTMLResponse)
async def knowledge_gap_detail(request: Request, db: DbSession, gap_id: uuid.UUID) -> HTMLResponse:
    gap = await KnowledgeGapRepository(db).get(gap_id)
    if gap is None:
        raise HTTPException(status_code=404, detail="Brecha de conocimiento no encontrada")
    persons = await PersonRepository(db).list(limit=500)
    facts = await FactRepository(db).list(limit=500)
    sources = await SourceRepository(db).list(limit=500)
    resolved_person = (
        await PersonRepository(db).get(gap.resolved_person_id)
        if gap.resolved_person_id is not None
        else None
    )
    resolved_fact = (
        await FactRepository(db).get(gap.resolved_fact_id)
        if gap.resolved_fact_id is not None
        else None
    )
    return templates.TemplateResponse(
        request,
        "knowledge_gap_detail.html",
        _ctx(
            request,
            gap=gap,
            persons=persons,
            facts=facts,
            sources=sources,
            resolved_person=resolved_person,
            resolved_fact=resolved_fact,
            status_options=KnowledgeGapStatus,
            fact_status_options=FactStatus,
            memory_status_options=MemoryStatus,
            sensitivity_options=SensitivityLevel,
        ),
    )


@router.post("/knowledge-gaps/{gap_id}/status")
async def knowledge_gap_set_status(
    request: Request, db: DbSession, gap_id: uuid.UUID
) -> RedirectResponse:
    detail_url = f"/admin/knowledge-gaps/{gap_id}"
    form = await request.form()
    try:
        payload = _gap_status_update(form)
    except ValidationError as exc:
        return _back(detail_url, error=_error_text(exc))
    updated = await KnowledgeGapRepository(db).set_status(
        gap_id,
        status=payload.status,
        resolved_person_id=payload.resolved_person_id,
        resolved_fact_id=payload.resolved_fact_id,
    )
    if updated is None:
        return _back(detail_url, error="Brecha de conocimiento no encontrada")
    return _back(
        detail_url,
        message=f"La brecha de conocimiento se ha marcado como {payload.status.value}",
    )


@router.post("/knowledge-gaps/{gap_id}/materialize-fact")
async def knowledge_gap_materialize_fact(
    request: Request, db: DbSession, gap_id: uuid.UUID
) -> RedirectResponse:
    detail_url = f"/admin/knowledge-gaps/{gap_id}"
    form = await request.form()
    try:
        payload = _fact_materialize(form)
        fact = await MaterializationService(db).materialize_gap_fact(gap_id, payload)
    except (ValidationError, SQLAlchemyError, MaterializationError) as exc:
        return _back(detail_url, error=_error_text(exc))
    return _back(detail_url, message=f"Se ha materializado como hecho {fact.id}")


@router.post("/knowledge-gaps/{gap_id}/materialize-memory")
async def knowledge_gap_materialize_memory(
    request: Request,
    db: DbSession,
    gap_id: uuid.UUID,
    embedder: EmbedderDep,
) -> RedirectResponse:
    detail_url = f"/admin/knowledge-gaps/{gap_id}"
    form = await request.form()
    try:
        payload = _memory_materialize(form)
        memory = await MaterializationService(
            db, embedder=embedder
        ).materialize_gap_memory(gap_id, payload)
    except (ValidationError, SQLAlchemyError, MaterializationError) as exc:
        return _back(detail_url, error=_error_text(exc))
    return _back(detail_url, message=f"Se ha materializado como recuerdo {memory.id}")


# --------------------------------------------------------------------------
# Core entity create/list pages
# --------------------------------------------------------------------------


@router.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request, db: DbSession) -> HTMLResponse:
    sources = await SourceRepository(db).list(limit=200)
    persons = await PersonRepository(db).list(limit=500)
    person_names = {p.id: p.display_name for p in persons}
    return templates.TemplateResponse(
        request,
        "sources.html",
        _ctx(
            request,
            sources=sources,
            persons=persons,
            person_names=person_names,
            source_type_options=SourceType,
        ),
    )


@router.post("/sources")
async def sources_create(request: Request, db: DbSession) -> RedirectResponse:
    form = await request.form()
    try:
        payload = _source_create(form)
        source = await SourceRepository(db).create(payload)
    except (ValidationError, SQLAlchemyError) as exc:
        return _back("/admin/sources", error=_error_text(exc))
    return _back("/admin/sources", message=f"Se ha creado la fuente {source.id}")


@router.get("/persons", response_class=HTMLResponse)
async def persons_page(request: Request, db: DbSession) -> HTMLResponse:
    persons = await PersonRepository(db).list(limit=200)
    media = await MediaRepository(db).list(limit=200)
    media_options = sorted(
        (
            {
                "id": item.id,
                "label": f"{item.media_type}: {item.title or item.storage_path}",
            }
            for item in media
        ),
        key=lambda option: option["label"].lower(),
    )
    media_labels = {option["id"]: option["label"] for option in media_options}
    return templates.TemplateResponse(
        request,
        "persons.html",
        _ctx(
            request,
            persons=persons,
            media_options=media_options,
            media_labels=media_labels,
        ),
    )


@router.post("/persons")
async def persons_create(request: Request, db: DbSession) -> RedirectResponse:
    form = await request.form()
    try:
        payload = _person_create(form)
        person = await PersonRepository(db).create(payload)
    except (ValidationError, SQLAlchemyError) as exc:
        return _back("/admin/persons", error=_error_text(exc))
    return _back("/admin/persons", message=f"Se ha creado la persona {person.display_name}")


@router.post("/persons/{person_id}/primary-media")
async def persons_set_primary_media(
    request: Request, db: DbSession, person_id: uuid.UUID
) -> RedirectResponse:
    """Set or clear a person's primary media from the persons page.

    Reuses the same repository and media-existence check as the JSON admin
    ``PATCH`` so both surfaces behave identically.
    """
    form = await request.form()
    raw = _text(form, "media_id")
    media_id = None
    if raw is not None:
        try:
            media_id = uuid.UUID(raw)
        except ValueError:
            return _back("/admin/persons", error="El identificador del recurso multimedia no es válido")
        if await MediaRepository(db).get(media_id) is None:
            return _back("/admin/persons", error="Recurso multimedia no encontrado")
    person = await PersonRepository(db).set_primary_media(person_id, media_id)
    if person is None:
        return _back("/admin/persons", error="Persona no encontrada")
    return _back(
        "/admin/persons",
        message=f"Se ha actualizado la multimedia principal de {person.display_name}",
    )


@router.get("/media", response_class=HTMLResponse)
async def media_page(request: Request, db: DbSession) -> HTMLResponse:
    media = await MediaRepository(db).list(limit=200)
    persons = await PersonRepository(db).list(limit=500)
    person_names = {p.id: p.display_name for p in persons}
    sources = await SourceRepository(db).list(limit=500)
    source_labels = {
        source.id: source.description or source.source_type.value for source in sources
    }
    return templates.TemplateResponse(
        request,
        "media.html",
        _ctx(
            request,
            media=media,
            person_names=person_names,
            source_labels=source_labels,
            media_type_options=MediaType,
        ),
    )


@router.post("/media")
async def media_create(request: Request, db: DbSession) -> RedirectResponse:
    form = await request.form()
    try:
        payload = _media_create(form)
        media = await MediaRepository(db).create(payload)
    except (ValidationError, SQLAlchemyError) as exc:
        return _back("/admin/media", error=_error_text(exc))
    return _back(
        "/admin/media", message=f"Se ha creado el recurso multimedia {media.id}"
    )


@router.get("/facts", response_class=HTMLResponse)
async def facts_page(request: Request, db: DbSession) -> HTMLResponse:
    facts = await FactRepository(db).list(limit=200)
    return templates.TemplateResponse(
        request,
        "facts.html",
        _ctx(
            request,
            facts=facts,
            persons=await PersonRepository(db).list(limit=500),
            sources=await SourceRepository(db).list(limit=500),
            person_names=await _person_names(db),
            source_labels=await _source_labels(db),
            fact_status_options=FactStatus,
        ),
    )


@router.post("/facts")
async def facts_create(request: Request, db: DbSession) -> RedirectResponse:
    form = await request.form()
    try:
        payload = _fact_create(form)
        fact = await FactRepository(db).create(payload)
    except (ValidationError, SQLAlchemyError) as exc:
        return _back("/admin/facts", error=_error_text(exc))
    return _back("/admin/facts", message=f"Se ha creado el hecho {fact.id}")


@router.get("/memories", response_class=HTMLResponse)
async def memories_page(request: Request, db: DbSession) -> HTMLResponse:
    memories = await MemoryRepository(db).list(limit=200)
    return templates.TemplateResponse(
        request,
        "memories.html",
        _ctx(
            request,
            memories=memories,
            persons=await PersonRepository(db).list(limit=500),
            sources=await SourceRepository(db).list(limit=500),
            person_names=await _person_names(db),
            source_labels=await _source_labels(db),
            memory_status_options=MemoryStatus,
            sensitivity_options=SensitivityLevel,
        ),
    )


@router.post("/memories")
async def memories_create(
    request: Request, db: DbSession, embedder: EmbedderDep
) -> RedirectResponse:
    form = await request.form()
    try:
        payload = _memory_create(form)
    except ValidationError as exc:
        return _back("/admin/memories", error=_error_text(exc))
    repo = MemoryRepository(db)
    embedding = None
    if repo.supports_vector_search:
        embedding = await embedder.embed(
            memory_search_text(title=payload.title, narrative=payload.narrative)
        )
    try:
        memory = await repo.create(payload, embedding=embedding)
    except (SQLAlchemyError, UnsupportedVectorSearch) as exc:
        return _back("/admin/memories", error=_error_text(exc))
    return _back("/admin/memories", message=f"Se ha creado el recuerdo {memory.id}")


# --------------------------------------------------------------------------
# Embedding backfill
# --------------------------------------------------------------------------


@router.get("/embeddings", response_class=HTMLResponse)
async def embeddings_page(request: Request, db: DbSession) -> HTMLResponse:
    repo = MemoryRepository(db)
    pending = len(await repo.list_without_embeddings(limit=500))
    total = await _count(db, Memory)
    return templates.TemplateResponse(
        request,
        "embeddings.html",
        _ctx(
            request,
            total_memories=total,
            pending_embeddings=pending,
            supports_vector_search=repo.supports_vector_search,
        ),
    )


@router.post("/embeddings/backfill")
async def embeddings_backfill(request: Request, db: DbSession, embedder: EmbedderDep) -> RedirectResponse:
    try:
        processed = await MemoryEmbeddingBackfillService(
            embedder=embedder, memory_repo=MemoryRepository(db)
        ).backfill()
    except UnsupportedVectorSearch as exc:
        return _back("/admin/embeddings", error=_error_text(exc))
    return _back(
        "/admin/embeddings",
        message=f"Se han generado los embeddings de {processed} recuerdos",
    )
