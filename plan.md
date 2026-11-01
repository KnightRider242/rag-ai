# Phase 5 semantic retrieval plan

Branch `feat/semantic-retrieval` starts from merged `main` at `4521f18`. This implementation scope is already approved; commit, push, PR, merge, and non-test database changes are not.

1. Add a small typed retrieval service and LangChain Core adapter. Validate query length, `top_k`, one finite nonzero embedding, and provider model metadata without coupling the service to FastAPI or psycopg.
2. Extend `PostgresIndexStore` with read-only, collection-scoped exact cosine search. Use one stable transaction snapshot for compatibility checks and ranking; materialize only same-dimension, same-model, nonzero candidates before `<=>`. Preserve source metadata and controlled errors, with no schema migration.
3. Register a separately gated development-only JSON search route using the existing local Ollama provider and threadpool pattern. Add unit, API, isolated PostgreSQL, and reproducible evaluation tests with synthetic data; run targeted checks before the full default suite and opt-in live checks only against disposable schemas.
4. Update README, architecture map/decision, developer guide, roadmap, security plan, and append the session log. Explain retrieval and metrics for learning; review dependencies, secrets, all tracked/untracked changes, and `git diff --check`. Report actual test and live-service results without starting Phase 6.

## Phase 5 final evaluation and documentation (approved)

1. Expand only the synthetic evaluation fixture and tests: 9 one-chunk source documents, 4 independently labeled questions, and Recall/Hit at K=1 and K=3 plus MRR@3. Set quality expectations before observing scores; inspect failures rather than changing thresholds to pass.
2. Replace the roadmap's indexing-only flow with one readable Mermaid diagram for indexing and retrieval, sharing the existing Ollama and PostgreSQL boundaries and showing the LangChain adapter as a separate entry point.
3. Run unit, isolated PostgreSQL, live Ollama evaluation, and full regression checks against the dedicated local test database only. Verify connection destination without displaying credentials; preserve the existing orphan schema and application database.
4. Update only relevant README/architecture/learning documentation and append the session log with measured results, security limits, diff review, and remaining human review work. No new dependency, migration, endpoint, or Phase 6 code.

## Phase 6 local text RAG (approved)

1. Add typed generation and RAG modules: a loopback-only Ollama `/api/chat` provider with validated JSON-schema output, deterministic bounded evidence context, a focused prompt, and application-owned citation resolution. Reuse Phase 5 `RetrievalService`; leave PostgreSQL schema and retrieval SQL unchanged.
2. Add a separate development-only answer route and Pydantic settings for model, output, and context bounds. Use the threadpool for synchronous clients, close providers on every path, and return controlled errors without logging question or evidence text.
3. Add synthetic unit/API/security tests plus opt-in disposable-schema PostgreSQL/Ollama tests. Verify answer, abstention, invalid citations, provider failures, collection scoping, and source provenance; keep semantic quality checks separate from structural guarantees.
4. Update README, roadmap flow, codebase map, developer guide, architecture decisions, security plan, environment example, and append the session log. Run targeted then full tests, inspect every tracked/untracked change, check secrets and `git diff --check`, and report measured results and limitations. No Phase 7 work or Git publication.

## Local RAG end-to-end validation (approved)

1. Verify the clean `test/local-rag-e2e` Phase 6 baseline and trace the existing PDF upload, chunk, index, search, and answer routes plus the disposable-schema fixture. Reuse those interfaces without changing application behavior.
2. Add one opt-in ASGI test with a two-page synthetic in-memory PDF and a second-collection TXT distractor. Check database destination before fixture writes, migrate only the new schema, exercise every route, validate provenance and abstention, and record informational timings without logging content.
3. Run the new test first, investigate any genuine failure, then run existing local integration and complete regression suites against only `rag_ai_retrieval_test` and installed loopback Ollama models. Keep the default suite service-free and make no cloud calls.
4. Document test setup, citation and isolation limits, and actual results in the developer/security guides and session log. Review tracked/untracked changes, secrets, `git diff --check`, and remaining risks; leave everything uncommitted.

## Local RAG testing documentation finalization (approved)

1. Verify the existing `test/local-rag-e2e` changes, actual FastAPI routes, schema CLI, and test fixtures. Separate the user's manual Swagger observations from the previous Codex automated results; make no runtime or test edits.
2. Add one practical `docs/LOCAL_RAG_TESTING.md` covering safe localhost startup, a synthetic two-page PDF, the manual Swagger/API sequence, citation mechanics, opt-in automated tests, troubleshooting, and limitations.
3. Link the guide from README and the developer/codebase/security documents; correct stale Phase 6 status in README/roadmap while keeping later release work planned. Append a dated documentation-only session-log entry without replacing history.
4. Check commands and examples against code, relative links, changed-file secrets, Git status, and `git diff --check`. Report documentation verification separately from historical test results; leave all changes uncommitted.

## Dedicated PostgreSQL E2E destination safety (approved)

1. Verify libpq URL, environment, and service-file precedence against PostgreSQL and psycopg documentation. Keep the user's `.env` and application settings unchanged.
2. In the E2E fixture, reject service/routing overrides and ambiguous URL destinations before any connection; pin an explicit loopback address in the exact DSN passed to the existing disposable-schema fixture. Add connection-free negative and positive tests.
3. Run focused preflight tests, then the default suite with temporary development-route overrides. Run local integration only after verifying the dedicated database and Ollama; inspect fixture cleanup, changed files, secrets, and `git diff --check`.
4. Update only the local testing guide, security plan, and append-only session log with actual results and remaining limits. Leave all work uncommitted.

## Codebase clarity and maintainability pass (approved)

1. Create `docs/codebase-clarity` from clean merged `main`; trace the implemented document, embedding, storage, retrieval, answer, API, and test boundaries against code.
2. Add only concise comments for non-obvious provenance, resource, transaction, compatibility, and citation invariants. Do not change executable statements or interfaces.
3. Correct stale merged-E2E status and strengthen `docs/CODEBASE_MAP.md` with a verified PDF-to-answer path, test entry points, and development security boundaries; preserve historical results.
4. Run source compilation, the default service-free suite if practical, Markdown/link and secret checks, `git diff --check`, and a full diff review. Append actual results to the session log; leave all changes uncommitted.
