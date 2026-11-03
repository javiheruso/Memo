# Memo — Project Context

Memo is a Spanish-first conversational companion for older adults. It works with curated facts, memories, events, and preferences, prioritizing provenance, uncertainty, and careful handling of sensitive topics.

> **Scope note:** This document summarizes the recorded repository state and source-level observations. It is not a security assessment or a guarantee about any deployed environment.

## Current status

**Confirmed**
- The modular-monolith MVP is implemented with public chat, administration with configurable authentication, retrieval, trust policies, claims, embeddings, optional voice/LLM providers, and admin-led onboarding.
- The onboarding workflow is integrated into `main` through commit `114e303`.
- The local checkout inspected for this document remains on `feat/admin-onboarding-workflow` at `4853ac5`; do not infer the active branch from the documented integration state.

**Pending verification before changes**
- Confirm the checked-out branch, working-tree status, and the exact `main` revision in the current clone.
- README implementation claims are partially stale; code and migrations take precedence.

## Implemented

- Public chat and voice endpoints, with conversational retrieval over facts, memories, events, and preferences.
- Intent routing, entity resolution, ambiguity handling, semantic memory retrieval, and one-turn fact-query follow-ups.
- Trust controls for provenance, uncertainty, event status, no invention, and sensitive topics.
- Protected admin UI/API for curated data, claims, knowledge gaps, profiles, and onboarding.
- Explicit, consent-gated onboarding activation with transactional materialization and provenance.
- Claim ingestion, review signals, manual materialization, and knowledge-gap tracking.
- Deterministic defaults plus optional OpenAI-compatible provider seams.
- Alembic migrations, Docker/Compose support, and SQLite/PostgreSQL CI paths.

## Planned or deferred

The following are documented as deferred, not current functionality:

- Semantic/paraphrase claim grouping, automated promotion, and family-review notifications.
- Semantic fact matching and multi-fact temporal reasoning.
- Public accounts, RBAC, fine-grained permissions, legal consent workflow, and consent revocation.
- Voice/media capture in onboarding, tablet client, browser audio workflow, media upload/serving, and media analysis.

## Product decisions and constraints

- Onboarding is currently **admin/family-led**; there is no public self-service flow.
- Activation requires explicit consent and keeps provenance.
- The system must not initiate sensitive topics. Direct questions receive a careful response and redirection.
- Memories are not presented as verified facts; uncertainty and provenance must survive retrieval.
- Ambiguity must not be silently resolved.
- Promotion from claims and knowledge gaps remains explicit and human-controlled.
- `/chat` is public by design; administrative surfaces require secure deployment configuration.

## Technical conventions

- Keep business logic in services and persistence in repositories.
- Treat migrations and current code as authoritative over historical memories and stale README statements.
- Preserve safe failure behavior for provider integrations: no invented sources and no broken text response.
- Keep claim-ingestion failures isolated from the user-facing response.
- Code and technical documentation are English; user-facing UI copy is Spanish.
- Run the appropriate SQLite/PostgreSQL and migration checks when changing persistence or retrieval behavior.

## Known risks and pending remediation

The following are source-level observations of the inspected revision. They are not claims about a live deployment.

- **Administrative authentication can fail open:** when `MEMO_ADMIN_PASSWORD` is empty, admin UI and API authentication are disabled. The application only warns in production. Default Compose configuration sets production mode while allowing an empty password.
- **PostgreSQL development defaults are unsafe for exposed hosts:** Compose publishes port `5432` and defaults to `memo` / `memo` credentials unless overridden. This does not establish that any live database is internet-accessible.
- **Chat and voice lack session-level authorization:** `conversation_id` functions as a bearer capability without ownership checks. A party that obtains it may access or continue that conversation.
- The active-profile invariant is enforced in application logic rather than by a database uniqueness constraint.
- Health checks are liveness-only and do not verify database availability.
- README claims about migration state, semantic retrieval, voice support, and onboarding UI do not fully match the current code.
- CI covers tests and migrations, but no lint/type-check job was observed.

## References

| Topic | References |
|---|---|
| Overview and roadmap | `README.md` |
| Application wiring and settings | `app/main.py`, `app/core/config.py` |
| Chat and voice | `app/api/v1/endpoints/chat.py`, `app/api/v1/endpoints/voice.py`, `app/web/chat.py` |
| Administration and onboarding | `app/api/v1/endpoints/admin.py`, `app/web/router.py`, `app/services/onboarding.py`, `app/repositories/onboarding.py` |
| Retrieval and trust policies | `app/retrieval/`, `app/policies/` |
| Claims and materialization | `app/claims/`, `app/services/claim_ingestion.py`, `app/services/materialization.py` |
| Authentication | `app/security.py` |
| Schema and migrations | `app/models/`, `migrations/versions/` |
| Deployment and CI | `compose.yaml`, `Dockerfile`, `.github/workflows/ci.yml` |

Relevant commits: `6ea3509` (fact-query follow-ups), `c8cce52` (follow-up implementation), `4853ac5` (onboarding validation hardening), and `114e303` (onboarding integrated into `main`).
