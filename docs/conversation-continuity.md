# Conversation continuity policy (V2 port)

This document describes the minimal V2 continuity policy ported into the
production `app/` pipeline from the experimental branch
`experiment/conversation-continuity-bench` at commit `afa29f3`. The benchmark
tree under `benchmarks/continuity/` is intentionally left unchanged and remains
the audit reference.

## Scope

The port changes only deterministic routing and verification seams. It adds no
dependency, migration, external call, live LLM use, or answer-generation path.
All answers and sources still come from the existing retriever and responder.
Continuity is derived exclusively from persisted response-source metadata
(`person` rows with `usage_type` `subject`/`candidate`), never from transcript
text.

## Policy

### 1. Explicit pronoun questions are not person names

`¿Quién es él/ella/ellos/ellas/ese/esa?` (optionally prefixed with `¿Y ...?`)
is recognized by `is_explicit_pronoun_question()` and routed as general chat.
The existing pronoun follow-up path then supplies the immediately previous
verified resolved subject:

- if a verified subject exists, the turn is answered from that person;
- if the previous answer was ambiguous, unrelated, or absent, the turn stays
  conversational with no sources and no arbitrary pick;
- the pronoun itself is never resolved against the database and never creates a
  knowledge gap.

The optional interpreter is skipped for these pronoun-only turns. They contain
no name for the interpreter to ground, so a call cannot help and could re-read
`él`/`ella` as a name.

### 2. Fact continuations reuse verified subjects from person and fact answers

An elliptical fact follow-up such as `¿Y de comer?` or `¿Y dónde vivió?` may
reuse the single verified `subject` of the latest assistant turn when that turn
was either:

- a resolved `person_query`, or
- a resolved `fact_query`.

`ConversationWindow` and `ConversationRepository` share the same intent set
(`FACT_SUBJECT_INTENTS`) so their contracts cannot drift. The continuation is
still refused when the latest assistant turn:

- has any other intent (for example `general_chat`),
- exposes any `candidate` source (unresolved ambiguity), or
- does not carry exactly one `subject` person source.

This keeps unrelated turns and ambiguity as hard continuity breaks.

### 3. Interpreter terms must be grounded in the current message

Before any database lookup, `InterpreterVerifier.verify()` requires the
suggested term to appear as a whole normalized token run in the current user
message. The check:

- folds case and accents (`José` matches `jose`);
- treats punctuation as a token boundary;
- accepts multi-word terms (`Paco García`);
- rejects absent terms, history-only terms, and substrings (`Antonio` does not
  match `Antoniola`).

Rejected suggestions produce no sources and no knowledge gap. The rejection log
contains no message text, term, or entity id.

## Contract changes

| Surface | Before | After |
|---|---|---|
| `IntentRouter.route("¿Quién es él?")` | `person_query` with term `él` | `general_chat`; pronoun follow-up path owns the turn |
| `ConversationWindow.last_fact_subject()` | `fact_query` only | resolved `person_query` or `fact_query` |
| `ConversationRepository.last_fact_subject()` | `fact_query` only | same set as the window |
| `InterpreterVerifier.verify()` | checked database existence only | requires current-message grounding before resolution |

## Regressions guarded by tests

The production suite adds distinct multi-turn/integration coverage in
`tests/test_conversation_continuity_port.py`:

1. explicit pronoun after a person answer;
2. explicit pronoun after a fact answer;
3. explicit pronoun with no active subject;
4. explicit pronoun after ambiguity (no candidate pick);
5. fact continuation after `person_query`;
6. fact continuation after `fact_query`;
7. unrelated turn breaking a fact continuation;
8. topic change and explicit return re-establishing continuity;
9. family role answer followed by a pronoun;
10. family role with no active person (no guess);
11. interpreter term absent from the current message;
12. interpreter term present only in history;
13. interpreter substring term;
14. interpreter case/accent grounding;
15. grounded interpreter name still verified and used;
16. conversation isolation;
17. sensitive blocked turn breaking pronoun carry.

Direct unit tests cover the router guard, the window/repository contract, token
grounding, the no-log-text guarantee, and verification-before-resolution.

## Limits

- Pronoun gender is not validated: `él` and `ella` both reuse the verified
  active subject.
- Continuity is metadata-only and deliberately narrow; semantic reference
  resolution beyond the single verified subject is out of scope.
- A relationship answer exposes the active person as the subject, not the
  related person named in the answer. Follow-ups continue from the active
  subject.
- The interpreter remains optional and is still only a routing hint; it cannot
  answer, create a source, or open a gap.
- The port was validated on in-memory SQLite with the deterministic embedder
  and responder. PostgreSQL/pgvector behavior is covered by the existing test
  profile but not by this policy change.
- The experimental benchmark's V2 subclass is now redundant for these three
  concerns. It is left intact as audit history and can be removed in a later
  cleanup.

## Verification evidence

Commands run on `experiment/conversation-continuity-port`:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest tests/test_conversation_continuity_port.py \
  tests/test_conversation_window.py tests/test_intent_router.py \
  tests/test_interpreter.py tests/test_multiturn_conversation.py -q
.venv/bin/python -m benchmarks.continuity.run \
  --json /tmp/continuity-port-report.json \
  --markdown /tmp/continuity-port-report.md
```

The full default suite passed with the local `.env` security variables
neutralized (`MEMO_ADMIN_PASSWORD=`, `MEMO_CHAT_SESSION_SECRET=`) and
`MEMO_TEST_DATABASE_URL` unset, matching the CI SQLite profile. The PostgreSQL
tests remained skipped.

Benchmark result with output written outside the tracked benchmark tree:

| Variant | Passed | False carries | Hallucination | Safety | Interpreter calls |
|---|---:|---:|---:|---:|---:|
| V0 deterministic baseline | 23/24 | 0 | 0 | 0 | 0 |
| V1 scripted interpreter | 24/24 | 0 | 0 | 0 | 7 |
| V2 experimental policy | 24/24 | 0 | 0 | 0 | 7 |

The port fixes the V0 baseline gaps the benchmark recorded, so three benchmark
assertions that deliberately require V0/V1 to fail are now incompatible:

- `test_known_concern_quien_es_el_after_person`
- `test_known_concern_interpreter_term_absent_from_message`
- `test_known_concern_carry_after_person_query`

Those assertions were not modified. They document the pre-port baseline and
should be updated only when the benchmark itself is revisioned.

## Integration recommendation

Merge the port as a separate conventional commit/PR from the benchmark branch.
Keep `benchmarks/continuity/` untouched for traceability. After merge, the next
benchmark revision can drop the redundant V2 subclass and replace the
baseline-failure assertions with regression assertions against the production
pipeline.
