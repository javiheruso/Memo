# memo

Conversational companion with personal memory for older adults. Modular monolith
backend (Python / FastAPI / PostgreSQL), aligned with the project dossier.

## Quick path

1. Create the environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

2. Configure the environment:

   ```bash
   cp .env.example .env
   ```

3. Run the API:

   ```bash
   uvicorn app.main:app --reload
   ```

4. Verify: open <http://localhost:8000/api/v1/health> or run the tests:

   ```bash
   pytest
   ```

## Layout

| Path | Purpose |
|------|---------|
| `app/main.py` | FastAPI entrypoint, router wiring |
| `app/core/` | Settings loaded from `MEMO_*` env vars |
| `app/api/` | HTTP layer: v1 router, health/admin/chat endpoints, dependencies |
| `app/web/` | Server-rendered UI: the public large-text chat page (`/chat`) and the admin UI (`/admin`), HTML templates + form handlers over the same services/repositories |
| `app/db/` | SQLAlchemy `Base`, async engine/session, migration target |
| `app/domain/` | Domain layer: enums for the Phase 1 status/type vocabulary |
| `app/models/` | SQLAlchemy ORM models for the Phase 1 tables |
| `app/services/` | Business logic: the conversation/query pipeline orchestrator |
| `app/claims/` | Claim ingestion seam: conservative deterministic extractor, grouper and contradiction detector, plus the provider registry |
| `app/repositories/` | Async data access for sources, persons, relationships, facts, memories, conversations, knowledge gaps, claims |
| `app/schemas/` | Pydantic request/response schemas for the admin and chat APIs |
| `app/retrieval/` | Query pipeline: intent router, entity resolver, structured retriever, semantic retriever, context builder |
| `app/policies/` | Trust-model rules enforced in code (no invention, no fact promotion, ambiguity handling, sensitive-memory filtering) |
| `app/embeddings/` | Replaceable embedder seam: deterministic placeholder sized to the schema's `vector(1536)` dimension, provider registry behind `MEMO_EMBEDDER`, and an OpenAI-compatible real embedder with dimension validation and fail-closed fallback |
| `app/llm/` | Replaceable responder interface, deterministic text responder, prompt builder and OpenAI-compatible real-LLM responder with fail-closed fallback |
| `app/voice/` | Replaceable optional STT/TTS seams: `SpeechToText` / `TextToSpeech` protocols, an OpenAI-compatible implementation for `/audio/transcriptions` and `/audio/speech`, a provider registry that resolves to `None` when unconfigured, and `VoiceService` running optional STT -> the conversation pipeline -> optional TTS |
| `migrations/` | Alembic setup, Phase 1 schema and follow-up revisions |

## Database

The app targets PostgreSQL and is prepared for pgvector usage (`pgvector`
is already a dependency). The default URL expects a local `memo` database:

```bash
createdb memo
alembic upgrade head
```

The initial migration enables the `vector` extension on the target database;
the server must have the `postgresql-vector` extension package installed.
`memory.embedding` and `media.embedding` are declared as `vector(1536)` and are
left unused until semantic retrieval is implemented.

The health endpoint and tests do not require a running database; connections
are created lazily when the first DB-backed endpoint is exercised. Tests use an
in-memory SQLite database (via `aiosqlite`) and a deterministic fictional
universe in `tests/fixtures/universe.py` (Carmen, Antonio, María, Javi,
Paco/Francisco, and two people sharing the alias "Paco" for ambiguity testing).

### Local PostgreSQL with pgvector (Docker Compose)

The recommended local setup is a Compose service that runs PostgreSQL with
pgvector pre-installed and provisions the two databases that match the app
defaults: `memo` (dev) and `memo_test` (postgres test profile).

Quick path:

1. `make db-up` — start the server (first start provisions `memo` + `memo_test`)
2. `make migrate` — apply Alembic migrations to `memo`
3. `make test-postgres` — run the whole suite, including the `postgres` smoke tests, against `memo_test`
4. `make db-down` — stop the server; data survives in a named volume

| Command | What it does |
|---------|--------------|
| `make db-up` | Start the PostgreSQL + pgvector container |
| `make db-down` | Stop the container; the data volume is kept |
| `make db-logs` | Tail the postgres container logs |
| `make db-shell` | Open a `psql` shell against `memo` |
| `make db-reset` | DESTRUCTIVE — drops the data volume (`memo` + `memo_test`) and restarts fresh |
| `make migrate` | Run `alembic upgrade head` against `memo` |
| `make test` | Default suite on in-memory SQLite (no database required) |
| `make test-postgres` | Full suite including the `postgres` smoke tests against `memo_test` |

The compose service uses the image `pgvector/pgvector:pg16` and defaults that
match the app settings (`memo` / `memo` on port `5432`). Values can be
overridden through environment variables (`POSTGRES_USER`, `POSTGRES_PASSWORD`,
`POSTGRES_DB`, `POSTGRES_PORT`), for example in `.env`. The credentials must
match what the app connects with, so when you override them also update
`MEMO_DATABASE_URL` and the `make test-postgres` connection string.

> DANGER: the postgres test profile is destructive by design. It drops and
> recreates the schema in `memo_test` on every run and truncates all tables
> between tests. Never point `MEMO_TEST_DATABASE_URL` at your main dev database
> (`memo`) or at any database you care about — it will be wiped.

### PostgreSQL-backed test run

The whole suite can run against a real PostgreSQL server by pointing
`MEMO_TEST_DATABASE_URL` at it; the in-memory SQLite default stays untouched
when the variable is unset. The conftest creates the schema once, truncates all
tables between tests, and the `postgres`-marked smoke tests in
`tests/test_postgres_profile.py` run only under this profile.

The Compose setup creates `memo_test` for you; just run `make db-up` and then
`make test-postgres`. For a manual server, create the database and extension
yourself before running the profile:

```bash
createdb memo_test
psql -d memo_test -c 'CREATE EXTENSION IF NOT EXISTS vector'
MEMO_TEST_DATABASE_URL=postgresql+psycopg://memo:memo@localhost:5432/memo_test pytest
```

The `vector` extension must exist in `memo_test` before the profile runs: the
test conftest creates the schema with SQLAlchemy directly, which does not
enable extensions.

### Responder selection

The chat endpoint receives its responder from `app/llm/provider.py`, wired
through `app/api/deps.py` from the `MEMO_RESPONDER` setting. Two responders
are registered today:

- `deterministic` (default) — the conservative Spanish text responder. No
  network, no configuration.
- `openai_compatible` — a real LLM responder that calls any OpenAI-compatible
  `/chat/completions` endpoint.

A responder may expose a `from_settings` classmethod to receive its
configuration; the provider calls it with the loaded settings.

### Real LLM responder

Setting `MEMO_RESPONDER=openai_compatible` switches the chat pipeline to the
LLM-backed responder without touching routing, retrieval, or policies.

| Setting | Default | Purpose |
|---------|---------|---------|
| `MEMO_RESPONDER` | `deterministic` | `deterministic` or `openai_compatible` |
| `MEMO_EMBEDDER` | `deterministic` | Embedder used for semantic memory retrieval: `deterministic` or `openai_compatible` |
| `MEMO_EMBEDDER_BASE_URL` | *(empty)* | Base URL of the OpenAI-compatible embeddings server (e.g. `https://api.openai.com/v1`, `http://localhost:11434/v1`). Required when `openai_compatible` is selected. |
| `MEMO_EMBEDDER_API_KEY` | *(unset)* | Bearer token sent as `Authorization`. Leave unset for local backends that do not need one (Ollama, LM Studio). |
| `MEMO_EMBEDDER_MODEL` | `text-embedding-3-small` | Model name sent in the request body. Its output must match the schema dimension (1536). |
| `MEMO_EMBEDDER_TIMEOUT_SECONDS` | `30.0` | HTTP timeout for the embeddings call. |
| `MEMO_LLM_BASE_URL` | *(empty)* | Base URL of the OpenAI-compatible server (e.g. `https://api.openai.com/v1`, `http://localhost:11434/v1`). Required when `openai_compatible` is selected. |
| `MEMO_LLM_API_KEY` | *(unset)* | Bearer token sent as `Authorization`. Leave unset for local backends that do not need one (Ollama, LM Studio). |
| `MEMO_LLM_MODEL` | `gpt-4o-mini` | Model name sent in the request body. |
| `MEMO_LLM_TIMEOUT_SECONDS` | `30.0` | HTTP timeout for the chat completion call. |

How it stays within the trust model:

- The model only sees the `AnswerContext`: a flat, epistemically-tagged prompt
  built by `app/llm/prompt_builder.py`. No database handle, repository, or
  tool is ever exposed, so the model cannot query storage directly.
- Retrieval, policy decisions and ambiguity resolution stay in code upstream;
  the prompt only repeats them as instructions.
- The model must answer with a JSON object `{"answer": ..., "sources": [...]}`
  where every source references an id listed in the context. The responder
  validates each id against the allowed set and derives the `usage_type` from
  the context item itself (subject, candidate, epistemic kind, event status),
  never from the model.
- Fail closed: on any invalid output (non-JSON, empty answer, invented or
  mismatched source id) or on any provider/network problem (HTTP error,
  timeout, connection failure), the turn falls back to the deterministic
  responder, so a model mistake can degrade a turn but can never inject an
  invented source into `response_sources`.

The test suite exercises the provider through `httpx.MockTransport` and never
makes real network calls.

### Real embeddings provider

Setting `MEMO_EMBEDDER=openai_compatible` switches the semantic memory
retrieval path to an OpenAI-compatible `/embeddings` endpoint, wired the same
way as the responder (`app/embeddings/provider.py`, `MEMO_EMBEDDER` settings
table above). Fail closed:

- The provider validates the returned vector against the schema dimension
  (`vector(1536)`). A wrong-dimension response is rejected.
- On any invalid output or provider/network problem (HTTP error, timeout,
  connection failure) the call falls back to the deterministic embedder, so an
  endpoint mistake can degrade one embedding but can never inject a
  wrong-dimension vector into a pgvector column or crash retrieval.

The test suite exercises the provider through `httpx.MockTransport` and never
makes real network calls.

### Optional voice layer (STT/TTS)

Speech in and out is an optional, replaceable layer over the same text
pipeline. `POST /api/v1/chat/voice` accepts multipart `audio` and/or `text`
and runs: optional speech-to-text (when audio and an STT provider are
configured) -> the exact `ConversationService` behind `POST /chat` -> optional
text-to-speech (when a TTS provider is configured). The JSON response carries
the recognized `transcript`, the explicit `text`, the `answer`, the response
`sources`/`ambiguity`/`gap` fields and, when synthesized, the audio as base64
in `audio` (`audio_content_type` is `audio/mpeg`).

Both features default to **disabled**, so text mode stays the zero-config
default and no voice provider is required to run or test the companion. The
providers resolve to `None` when unconfigured and the endpoint degrades to
text-only (see `app/voice/provider.py`).

| Setting | Default | Purpose |
|---------|---------|---------|
| `MEMO_STT` | *(empty)* | STT provider: `openai_compatible`, or unset to disable |
| `MEMO_STT_BASE_URL` | *(empty)* | Base URL of the OpenAI-compatible server (e.g. `https://api.openai.com/v1`). Required when `openai_compatible` is selected. |
| `MEMO_STT_API_KEY` | *(unset)* | Bearer token sent as `Authorization`. Leave unset for local backends that do not need one. |
| `MEMO_STT_MODEL` | `whisper-1` | Model name sent to `/audio/transcriptions`. |
| `MEMO_STT_TIMEOUT_SECONDS` | `30.0` | HTTP timeout for the transcription call. |
| `MEMO_TTS` | *(empty)* | TTS provider: `openai_compatible`, or unset to disable |
| `MEMO_TTS_BASE_URL` | *(empty)* | Base URL of the OpenAI-compatible server. Required when `openai_compatible` is selected. |
| `MEMO_TTS_API_KEY` | *(unset)* | Bearer token sent as `Authorization`. Leave unset for local backends that do not need one. |
| `MEMO_TTS_MODEL` | `tts-1` | Model name sent to `/audio/speech`. |
| `MEMO_TTS_VOICE` | `alloy` | Voice name sent to `/audio/speech`. |
| `MEMO_TTS_TIMEOUT_SECONDS` | `30.0` | HTTP timeout for the speech call. |

Fail closed:

- STT (`app/voice/openai_compatible.py`, `/audio/transcriptions`) never invents
  or returns an empty transcript: a malformed/empty provider response raises,
  and `VoiceService` catches provider/network errors and degrades the turn to
  the explicit `text` when one is present. With no usable text the endpoint
  answers `400` instead of feeding silence into the conversation pipeline.
- TTS (`/audio/speech`) never fabricates audio: an empty audio body raises, and
  `VoiceService` catches provider/network errors and returns the text reply
  with no `audio` field. A voice provider failure can therefore never crash or
  block a text answer.

The test suite exercises the providers through `httpx.MockTransport` and the
service/endpoint through injected fakes; no test makes real network calls.

## Admin data-loading API

A minimal admin surface under `/api/v1/admin` creates and lists the Phase 1 core
entities: `sources`, `persons` (with nested aliases), `relationships`, `facts`,
`memories` (with linked people), and `media` (metadata only — no upload, no
image serving). `GET /persons/search?name=...` resolves
persons by name or alias, returning all candidates when a name is ambiguous.
A person's primary photograph (the one the photo hint tier cites) is set or
cleared with `PATCH /persons/{id}/primary-media` (`{"media_id": "<uuid>"}` or
`null` to clear); the media row must exist but needs no `media_people` link.
Trust-model defaults are conservative: new facts default to `pending` and new
memories to `candidate`; relationships default to `verified`.

## Minimal admin authentication

Both the JSON admin API and the admin UI share one optional gate: a single
admin password (`MEMO_ADMIN_PASSWORD`). With it unset the surface is
**fail-open** — no credentials are required, which keeps local development and
the default test suite unchanged. Set it and both surfaces require the same
password:

| Surface | How you authenticate |
|---------|----------------------|
| `/api/v1/admin/*` | `Authorization: Bearer <password>` header |
| `/admin` UI | Browser login at `/admin/login`; HMAC-signed session cookie for 12 hours |

How it stays small:

- **One shared password** — no user accounts, no JWT, no database sessions.
- The cookie is an HMAC-SHA256-signed expiry timestamp keyed with the admin
  password (stdlib `hmac`); no new dependencies.
- The bearer token *is* the password, compared in constant time.
- Auth is enforced through shared dependencies (`app/security.py`) mounted on
  both routers, so the two surfaces can never drift apart.
- Fail-open is the only special case: unset password ⇒ every admin route passes.

Quick path:

1. `cp .env.example .env`, set `MEMO_ADMIN_PASSWORD` to a strong value.
2. Browser: open `/admin`, log in, then operate the UI.
3. API: send the password as the bearer token, e.g.
   `curl -H "Authorization: Bearer $MEMO_ADMIN_PASSWORD" http://localhost:8000/api/v1/admin/persons`
4. Log out from the UI button (or `POST /admin/logout`).

Accepted limitations for this slice:

- Cookies are HTTP-only and SameSite=Lax but not `Secure` (so the login works
  over plain `http` locally); use TLS in any real deployment.
- Login/logout and admin form posts have no CSRF token — fine for a
  single-operator local tool, not for multi-user deployment.
- The bearer token is the raw password, so it is only as safe as the transport
  and wherever you paste it.

## Server-rendered admin UI

Open <http://localhost:8000/admin> to operate the backend workflows without
hand-calling endpoints. It is a plain FastAPI + Jinja2 surface (no frontend
toolchain) that reuses the same repositories, pydantic schemas and services as
the JSON admin API, so the existing guardrails are preserved. When
`MEMO_ADMIN_PASSWORD` is set the surface is gated by the browser login
(`/admin/login`); otherwise it is fail-open like the API.

| Page | Path | Purpose |
|------|------|---------|
| Dashboard | `/admin` | Counts and navigation |
| Claim-group queue | `/admin/claim-groups`, `/admin/claim-groups/{id}` | Review, resolve/ignore, merge duplicates, materialize to fact/memory |
| Knowledge-gap queue | `/admin/knowledge-gaps`, `/admin/knowledge-gaps/{id}` | Review, resolve/ignore, materialize to fact/memory |
| Entities | `/admin/sources`, `/admin/persons`, `/admin/media`, `/admin/facts`, `/admin/memories` | Minimal create + list; the persons page also sets/clears each person's primary media |
| Embeddings | `/admin/embeddings` | One-click memory embedding backfill (PostgreSQL/pgvector only) |

Quick path:

1. `uvicorn app.main:app --reload`
2. Open <http://localhost:8000/admin>
3. Create a source and a person, then a fact/memory; review the claim-group and
   knowledge-gap queues; materialize a reviewed source into a fact or memory.

Every form posts back to the same router with `application/x-www-form-urlencoded`
and rebuilds the API's pydantic schemas, so the trust defaults (`pending` facts,
`candidate` memories), materialization guards (unknown/ignored/duplicate sources)
and append-only provenance behave identically to the JSON API. Failures flash an
error message on the target page; nothing is silently swallowed.

## Public chat page

Open <http://localhost:8000/chat> to talk to the companion in text mode with no
auth. It is a plain FastAPI + Jinja2 surface (no frontend toolchain, no JS, no
streaming) with a large-text layout sized for the older-adult audience. Every
message is processed by the same `ConversationService` behind
`POST /api/v1/chat`, so intent routing, retrieval, trust policies, claim
ingestion and knowledge-gap persistence behave identically to the API.

Quick path:

1. `uvicorn app.main:app --reload`
2. Open <http://localhost:8000/chat>
3. Write a message and press **Enviar**; the reply renders below, and the
   transcript keeps growing across turns.

Details:

| Concern | Decision |
|---------|----------|
| Multi-turn continuity | A session cookie (`memo_chat_conversation`) holds the `conversation_id`; history itself lives in the database |
| New conversation | The *Empezar otra conversación* button clears the cookie (never deletes stored history) |
| Pipeline signals | A created gap ("unknown person") and an ambiguous person render as an inline notice above the input; the assistant reply text is still shown as the turn |
| Errors | Blank messages and persistence failures flash on the page; nothing is silently swallowed |

Accepted limitations for this slice: the transcript shows both sides of every
stored turn for the current conversation but no per-message timestamps, the
cookie is HTTP-only and SameSite=Lax but not `Secure` (use TLS in any real
deployment), and the whole surface speaks to whatever person is at the device
(a conversation can be shared; the reset button starts a clean thread).

## Query pipeline ("Who was Paco?" / "¿Viene María hoy?")

`POST /api/v1/chat` runs the baseline query pipeline end to end in text mode:

1. intent routing (`general_chat`, `person_query`, `memory_query`, `event_query`,
   `orientation_query`, `unknown`);
2. entity resolution by name/alias with explicit ambiguity handling;
3. retrieval — structured data for person/event queries, semantic memory
   retrieval for `memory_query`;
4. trust-model context building (no invention, no memory-as-fact, no silent
   ambiguity collapse, sensitive memories filtered, event status preserved);
5. a replaceable responder — today a deterministic Spanish text responder,
   selected through the `MEMO_RESPONDER` setting;
6. persistence of the conversation, messages, `response_sources`, and a
   `knowledge_gap` when the person is unknown.

`event_query` resolves the person from phrasings like "¿Viene María hoy?" or
"¿Qué eventos tiene Antonio?" and renders upcoming events phrased by their own
status (`planned` / `confirmed`), never presenting them as already completed.

`person_query` answers with the first progressive hint that works (dossier
doc 01): verified relation, then probable relation, then stored facts, then a
stored photograph, then a shared memory, and only then an honest "no details"
reply. Facts are quoted exactly as stored and probable facts are marked
"probable"; the photo tier cites stored metadata (approximate year/place) and
never describes what the image shows — media rows in this slice carry metadata
only (`/api/v1/admin/media`), with no upload, serving, or content analysis.
A photo can be attached to a person either through the media row's `people`
link or directly through `persons.primary_media_id`; retrieval honors the
latter even when no `media_people` row exists, surfacing it first in the photo
tier.

`orientation_query` catches day/date/time phrasings ("¿Qué día es hoy?", "¿Qué
hora es?", "What's the date?"). The router precedence is `person_query` >
`memory_query` > `event_query` > `orientation_query`, and the orientation
patterns are end-anchored so "¿Qué día viene María?" stays an event query. The
answer is built from a code-provided clock carried in `AnswerContext.now`
(UTC): the deterministic responder renders the weekday, date and time from it,
and the prompt builder hands that same trusted clock to the LLM path, so a
date/time statement is grounded in the application's clock, never the model's
own knowledge.

### Memory queries and the semantic retrieval seam

`memory_query` catches recollection phrasings ("¿Te acuerdas de cuando íbamos a
bailar?", "Do you remember when we went dancing?"). The pipeline embeds the
full user text and ranks memories by similarity:

- `app/embeddings/deterministic.py` — the default deterministic embedder. It
  produces stable L2-normalized bag-of-token vectors sized to the schema's
  `vector(1536)` dimension, so shared words rank memories deterministically.
- `app/embeddings/openai_compatible.py` — a real embeddings endpoint, selected
  via `MEMO_EMBEDDER=openai_compatible`, with dimension validation and
  fail-closed fallback to the deterministic embedder (see "Real embeddings
  provider" above).
- `MemoryRepository.similarity_search` — the PostgreSQL/pgvector path
  (`embedding <-> query`, L2 distance). It only matches memories that carry a
  stored embedding.
- SQLite fallback — the default test path. When the dialect is not PostgreSQL,
  `SemanticRetriever` scores the corpus in Python, embedding memories whose
  `embedding` is `NULL` on the fly from their title and narrative, so no
  vector-capable database is needed to run the suite.

Trust stays strict on this path: hits above the sensitivity ceiling are
excluded in code before they reach the context, every hit is tagged with the
`memory` epistemic kind (never a fact), and a query with no meaningful match
gets an honest "I don't have that memory stored" answer with no sources and no
invention.

### Embedding persistence and backfill

Narrative memories are the vectorizable content (dossier doc 03); relationships,
facts, events and trust state stay structured and are never embedded. The admin
API persists embeddings when the database can store pgvector values (PostgreSQL
only):

- `POST /api/v1/admin/memories` embeds the new memory (title + narrative) when
  the bound dialect is PostgreSQL and stores it in `memory.embedding`.
- `POST /api/v1/admin/memories/backfill` embeds memories whose `embedding` is
  `NULL` and returns the number processed. It is idempotent: a second call
  with nothing pending returns `0`.

Both paths assemble the embedded text through the same helper as the retrieval
fallback (`app/embeddings/memory_text.py`), so a stored vector and an
on-the-fly vector for the same memory always agree. On a non-PostgreSQL dialect
(the default SQLite test path) no vector is ever written: create leaves
`embedding` `NULL`, and backfill fails safely with a `400` "requires a
PostgreSQL dialect" response. `MemoryRepository` refuses embedding writes on
non-PostgreSQL dialects, and the PostgreSQL-profile smoke tests prove the
create/list/update/backfill paths against a real server.

The response includes the intent, the answer, the exact entities used as
sources, ambiguity candidates when applicable, and the created/updated gap.
Each retrieved item carries an explicit epistemic kind
(`verified_fact`, `probable_fact`, `memory`), so a memory can never be
presented as a verified fact. The deterministic responder is the default; an
optional real-LLM responder (`MEMO_RESPONDER=openai_compatible`) can render
the same context with fail-closed fallback, still reporting truthful sources.
Speech is intentionally not part of this foundation slice.

## Candidate claims from conversation

On conversational turns (`general_chat`, `unknown`), the pipeline extracts
candidate claims from the user message and persists them append-only into the
`claims` table through a replaceable extractor seam:

- `app/claims/deterministic.py` — the conservative deterministic default. It
  only accepts declarative statements with a first-person past-tense signal
  (imperfect/preterite verb forms and strong narrative phrases like "recuerdo
  que" / "me acuerdo"), and skips questions, greetings, requests/imperatives and
  generic chat. It prefers false negatives over noise.
- `app/claims/grouping.py` — the conservative deterministic default grouper. Two
  candidates share a group only when their normalized claim text (case, accents,
  punctuation and whitespace folded) is identical; paraphrases do NOT merge in
  this baseline. A semantic grouper can be swapped in later behind the same seam.
- `app/claims/provider.py` — registered as `deterministic` and selected via
  `MEMO_CLAIM_EXTRACTOR` / `MEMO_CLAIM_GROUPER`, mirroring the responder/embedder
  wiring, so a real LLM-backed extractor or semantic grouper can be swapped in
  later without touching the pipeline.
- `app/repositories/claim.py` — `ClaimRepository.add_many` persists verbatim
  `original_text` tied to its `conversation_id` and `message_id`, with
  `first_seen_at` and a `candidate` status. The canonical
  subject/predicate/object triplet stays `NULL`: canonicalization belongs to a
  later phase and is out of scope.
- `app/repositories/claim_group.py` and `app/services/claim_ingestion.py` —
  `ClaimIngestionService` routes each persisted claim into a `claim_groups` row
  (via the `claim_group_members` link) and refreshes the denormalized recurrence
  counters: `total_mentions`, `distinct_conversations`, `distinct_days`,
  `first_seen_at` and `last_seen_at`. No promotion or family-review logic lives
  here.
- `app/claims/contradiction.py` and `app/repositories/claim_group_contradiction.py`
  — the conservative contradiction-signal baseline. The deterministic detector
  flags a claim group against a verified fact only when the claim carries an
  explicit negation marker and repeats every content token of the fact surface
  (`value_text`/`place_text`); a semantic detector can be swapped in later via
  `MEMO_CLAIM_CONTRADICTION_DETECTOR`. Detection runs after claims and groups are
  committed, records `claim_group_contradictions` evidence append-only
  (repeated detections refresh `detection_count` instead of duplicating rows),
  and lifts the `needs_review` flag on the group. It is a pure signal: it never
  promotes, rejects, or edits claims or facts.

Trust boundaries are preserved: claims are never promoted to facts or memories,
retrieval intents (`person_query`, `memory_query`, `event_query`,
`orientation_query`) never produce claims, and ingestion is fail-safe — any
extractor, persistence, grouping or contradiction-detection error is caught and
logged so it can never change the response behavior. A grouping failure leaves
the claims persisted but ungrouped; a detection failure leaves the claims and
groups persisted but without contradiction signals.

### Inspecting claims, groups and contradictions

The admin surface exposes listing for the claim baseline: `GET /api/v1/admin/claims`
lists recent claims (each with its `claim_group_id` when grouped),
`GET /api/v1/admin/claim-groups` lists groups with their recurrence counters,
`needs_review` flag and manual `status` (`active` / `resolved` / `ignored`,
filter with `?needs_review=true|false` or `?status=active`),
`GET /api/v1/admin/claim-groups/{id}` returns a group with its member claims and
contradiction signals — each member claim also carries its originating
`message` (`text` and `speaker`), so a reviewer sees the full utterance a claim
was extracted from without opening the conversation, and
`GET /api/v1/admin/claim-contradictions` lists recent
signals with the affected group (`canonical_text`, `needs_review`, `status`) and
the referenced verified fact.

Manual resolution is a write path:
`PATCH /api/v1/admin/claim-groups/{id}` with `{"status": "resolved"}` (or
`"ignored"`) and an optional `resolution_note` takes a group out of the review
queue: `needs_review` is cleared and `resolved_at` is stamped. Resolved/ignored
groups are skipped by contradiction detection, so a later contradicting claim
never re-flags them immediately; reopening with `"status": "active"` puts them
back. This never promotes, rejects or edits member claims, and contradiction
evidence rows stay append-only. There is no other admin write path for claims,
groups or contradictions; claims only enter through conversation ingestion.

### Knowledge gaps admin surface

`GET /api/v1/admin/knowledge-gaps` lists knowledge gaps (filter with
`?status=observed|recurrent|family_review|resolved|ignored`), and
`PATCH /api/v1/admin/knowledge-gaps/{id}` applies a manual status transition
using the existing `knowledge_gap_status` model. Resolving may link the gap to
the person or fact that closed it via `resolved_person_id` / `resolved_fact_id`
(only valid with `status: "resolved"`). The write only changes the gap's own
status — it never promotes anything.

### Explicit materialization (reviewed evidence -> facts / memories)

The only promotion-like write in the system is explicit, human-initiated and
audited. Four admin endpoints materialize a reviewed claim group or knowledge
gap into a fact or a memory:

- `POST /api/v1/admin/claim-groups/{id}/materialize-fact`
- `POST /api/v1/admin/claim-groups/{id}/materialize-memory`
- `POST /api/v1/admin/knowledge-gaps/{id}/materialize-fact`
- `POST /api/v1/admin/knowledge-gaps/{id}/materialize-memory`

Each call creates the target row, appends one provenance row to the new
append-only `materializations` table, and transitions the source out of the
review queue (claim group or gap to `resolved`, with the created fact/memory
linked in the resolution audit). Guards:

- unknown claim group / gap id -> `404`;
- `ignored` sources are refused -> `409`;
- a source materialized into the same target kind twice is refused -> `409`
  (a claim group can become one fact and one memory, but never two of the same);
- unknown `source_id` in the payload -> `422`.

Defaults stay trust-safe: facts default to `pending` and memories to `candidate`
(overridable per call), and `value_text` / `narrative` / `title` default to the
source's canonical text. Memory materialization reuses the embedding path of the
manual memory creation (`app/embeddings/memory_text.py` + pgvector on
PostgreSQL), so on SQLite the embedding stays `NULL`. Member claims, gap
occurrences and contradiction evidence are never modified, and the target
row + provenance row + source transition commit together.

The admin UI forms pre-fill `value_text` / `title` / `narrative` with the
group's first-seen member wording (verbatim `original_text`, falling back to the
canonical text when there are no members), so the human default is the real
sentence the person said instead of the lowercased grouping key. The JSON API
defaults are intentionally unchanged: service-side materialization still falls
back to the canonical text when the field is omitted.

### Human canonical merge of duplicate claim groups

Conservative deterministic grouping keeps paraphrases in separate groups, so
the review queue can accumulate duplicate claim groups that only differ in
wording. The human canonical merge folds one active group into another active
survivor explicitly — never automatically and never with a semantic/LLM engine:

- `POST /api/v1/admin/claim-groups/{group_id}/merge` with
  `{"target_group_id": ..., "confirm_opposite_polarity": bool}` absorbs
  `group_id` into the surviving `target_group_id` and returns the survivor.
- `GET /api/v1/admin/claim-groups/{group_id}/merge-suggestions` ranks the other
  active groups by shared content tokens (deterministic, stopword-filtered), so
  the UI presents the most plausible duplicates first.

How the merge stays evidence-preserving:

- Member claims move to the survivor **unchanged** (never mutated); the
  survivor's recurrence counters are recomputed from its new member set.
- Contradiction evidence rows transfer to the survivor. When the survivor
  already holds a signal for the same `(fact, detector)`, the rows **fold**
  (min `first_detected_at`, max `last_detected_at`, summed `detection_count`)
  instead of duplicating or dropping evidence.
- A new `claim_group_remembered_keys` row records the absorbed group's
  canonical text pointing at the survivor, and any remembered keys that pointed
  at the absorbed group re-point to the survivor. Future claims whose wording
  normalizes to an absorbed text therefore attach to the survivor (resolved in
  `ClaimGroupRepository.attach`) instead of recreating the deleted group.
- Only after those moves does the service delete the absorbed group row, and
  the whole unit commits once. If the absorbed group was flagged
  (`needs_review`), the survivor inherits the flag.

Guards: unknown group ids `404`; self-merge, `ignored` or otherwise non-`active`
endpoints, groups that already produced a materialization (their provenance row
would dangle), and opposite-polarity merges without an explicit
`confirm_opposite_polarity=true` are refused with `409`. Opposite-polarity means
exactly one of the two canonical texts carries a negation marker; folding a
negated wording into a non-negated survivor (or vice versa) must always be a
conscious, confirmed human choice.

The UI offers the same workflow on the claim-group detail page (`/admin/claim-groups/{id}`):
an "active" group shows a *Merge into another group* form with the deterministic
suggestions ranked by shared words, an opposite-polarity confirmation checkbox,
and error flash messages identical to the API guards.

## Status

The MVP is feature-complete against the dossier. Implemented and verified:

- **Query pipeline** — intent routing (`person_query`, `memory_query`,
  `event_query`, `orientation_query`, `general_chat`), entity resolution with
  explicit ambiguity handling, structured retrieval (relationships, facts,
  events), semantic memory retrieval (pgvector on PostgreSQL, in-memory
  fallback on SQLite), and trust-model context building.
- **Trust model** — rules enforced in code: no invention, memories never
  presented as verified facts, planned/confirmed events never phrased as
  completed, sensitive memories filtered, ambiguity never collapsed.
- **Claims lifecycle** — append-only ingestion, deterministic grouping with
  remembered keys, recurrence counters, contradiction signals, manual review,
  human merge, and explicit admin-only materialization into facts/memories with
  append-only provenance.
- **Knowledge gaps** — auto-created on unknown persons, manual resolution.
- **Progressive hints** — person answers follow relation → facts → photo →
  memory, plus a metadata-only media slice with primary-media management.
- **Voice (STT/TTS)** — replaceable OpenAI-compatible seams, text-first by
  default, graceful text-only degradation.
- **Interfaces** — JSON API, server-rendered admin UI (`/admin`) with minimal
  shared-password auth, and a public conversational chat page (`/chat`).

Verification:

- 391 tests pass on the default in-memory SQLite suite.
- 386 tests pass against real PostgreSQL 18 + pgvector (`make test-postgres`).
- Alembic migrations apply cleanly (head `f2a1b3c4d5e6`), 24 tables.

## Checklist

- [x] `alembic upgrade head` applies cleanly against PostgreSQL
- [x] `pytest` passes (SQLite and PostgreSQL profiles)
- [ ] `cp .env.example .env` and set secrets before deploying

## Roadmap

Deferred by design (dossier phases) and not required for the MVP:

- Semantic/paraphrase claim grouping and `claim_metrics` (consolidation phase).
- Automated promotion and full family-review notification flows — manual
  materialization is the current authority.
- Real STT/TTS/embedder/LLM providers — configure `MEMO_STT_*`, `MEMO_TTS_*`,
  `MEMO_EMBEDDER_*`, `MEMO_LLM_*` against your OpenAI-compatible backend.
- Dedicated tablet/touch client (the `/chat` page is the current text surface).
- Client-side audio capture/playback and streaming.
- Media image serving/upload (currently metadata-only).

## Deploying

### Docker Compose (whole stack)

`compose.yaml` builds the app (`Dockerfile`) and runs it next to the
`pgvector/pgvector:pg16` database:

```bash
docker compose up -d --build
docker compose run --rm app alembic upgrade head   # apply migrations once
```

Then open <http://localhost:8000> (chat at `/chat`, admin at `/admin`, docs at
`/docs`).

Set the admin password before exposing the service, otherwise the admin
surface stays unauthenticated (the app logs a startup warning in
non-development environments):

```bash
MEMO_ADMIN_PASSWORD='a-strong-password' docker compose up -d
```

The database host inside the compose network is the service name (`postgres`);
`compose.yaml` sets `MEMO_DATABASE_URL` accordingly. Data persists in the named
`postgres_data` volume.

### Minimal deployment checklist

1. Set `MEMO_ENVIRONMENT=production` and a strong `MEMO_ADMIN_PASSWORD`.
2. Put the app behind TLS (HTTPS); session cookies are `Secure`-flagged in
   non-development environments.
3. Configure the optional providers you want to use — `MEMO_LLM_*`,
   `MEMO_EMBEDDER_*`, `MEMO_STT_*`, `MEMO_TTS_*` (OpenAI-compatible). Without
   them the deterministic defaults keep text mode fully functional.
4. Run `alembic upgrade head` against the target database.
5. Keep `/chat` public (the companion) and `/admin` protected by the shared
   password; do not expose the admin password in plaintext or commit it.

### Production safety notes

- Admin auth is **fail-open only when `MEMO_ADMIN_PASSWORD` is unset**; a
  production deployment must set it.
- The admin credential is a single shared password with no per-user audit,
  no CSRF tokens, and no login rate limiting — adequate for a local/single
  operator, not a multi-user public service.
- Secret values (`MEMO_ADMIN_PASSWORD`, API keys) belong in the environment or
  a secrets manager, never in the repository.