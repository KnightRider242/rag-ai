# Phase 5 semantic retrieval plan

## Phase 9B.2 combined retrieval (approved, 2026-11-02)

1. Add strict combined result models and a service that embeds one query, validates its vector, fuses bounded text and visual candidates with deterministic RRF, and deduplicates visual elements.
2. Add one PostgreSQL read path using a single repeatable-read, read-only transaction for both collection-scoped compatibility checks and exact cosine searches. Preserve existing search endpoints and schema.
3. Add a separately gated development API route and setting with sanitized failures; cover service ranking, API gates, and disposable PostgreSQL snapshot, isolation, and provenance behavior.
4. Update concise retrieval, roadmap, architecture, and security documentation; run focused and full tests against a verified disposable database, a local embedding smoke check if available, compilation, diff, and secret checks. Leave changes uncommitted. Risks: unauthenticated localhost routes and unverified visual interpretation content.

## Phase 9A.1 visual-element persistence (approved, 2026-11-01)

1. Add `0002_visual_elements.sql` with composite document ownership, constrained displayed-page geometry, provenance, and cascade deletion. Keep `0001` unchanged; update migration tests for fresh and version-1 disposable schemas.
2. Adapt the bounded Phase 8 PDF inspection result into strict, deterministic raster, table, and vector-path candidates. Normalize coordinates once for both SHA-256 identity and persistence; keep existing Phase 8 responses unchanged.
3. Add a focused visual store with source-hash checks, document-row locking, atomic extractor-set replacement, and collection-scoped listing. Add separately gated development POST/GET routes and synthetic API/storage/security tests.
4. Update current-facing README and developer, architecture, migration, roadmap, testing, and security docs; append the session log. Phase 9A.2 interpretation and Phase 9B retrieval remain future work.
5. Verify compilation, focused and full default tests, and PostgreSQL tests only after verifying the exact disposable destination. Review whitespace, links, secrets, Git status, and the full diff; leave all changes uncommitted. Risks: PostgreSQL numeric special values, expensive native PDF parsing, concurrent replacement, and unauthenticated development routes.

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

## Phase 8A.1 PDF image detection (approved)

1. Add a small PyMuPDF inspector and typed page/image metadata, reusing existing filename, upload-size, and page-count rules while accepting image-only PDFs.
2. Register a separate, disabled-by-default development route and reject production opt-in; do not alter the text parser or later RAG stages.
3. Add synthetic API and domain tests for text, images, page metadata, malformed files, resource limits, and route gating.
4. Document local Swagger verification, run focused and default tests plus diff/security checks, and append actual results to the session log. Leave the branch uncommitted.

## Phase 8A.1 displayed-coordinate correction (approved)

1. Transform PyMuPDF's unrotated image boxes with each page's rotation matrix, intersect them with the displayed page rectangle, and retain an occurrence count even for fully off-page images.
2. Add synthetic 0/90/180-degree and clipped/off-page tests, preserving invalid-geometry and image-limit rejection; clarify the response convention in the local guide and security plan.
3. Run focused/default tests, compilation, diff and Git checks; append actual results to the session log. No database, cloud, Git publication, or Phase 8A.2 work.

## Phase 8 local multimodal understanding (approved)

1. Extend PDF inspection with bounded vector-path and table metadata, preserving displayed-page coordinates and existing image fields; verify synthetic PDFs before moving on.
2. Add bounded in-memory PDF page/region PNG rendering, then a separate loopback-only Ollama vision provider with typed structured output and mocked transport tests.
3. Add development-only PDF and PNG/JPEG analysis routes that attach application-owned SHA-256, page, and region provenance to model observations; verify source and resource boundaries in synthetic API tests.
4. Run a five-page synthetic API workflow and applicable focused/default tests; use an opt-in live model check only if a compatible model is already installed, without downloads or database access.
5. Update only relevant project documentation, review security and all tracked/untracked changes, compile source, run `git diff --check`, and append actual results to the session log. Keep Phase 9 storage/retrieval separate.

## Phase 8 reliability and security hardening (approved)

1. Keep strict Ollama completion and the 512/4096/0 generation settings; add small internal failure categories for limits, transport/timeouts, envelope errors, and structured-output errors while retaining sanitized public 502 responses.
2. Extend existing Phase 8 provider, API, PDF, and image tests with bounded negative cases, a generation-budget assertion, and cleanup/no-disclosure checks. Change input handling only for a reproduced defect.
3. Run focused and default regression suites, then explicitly opt in to local Qwen3-VL evaluation of the identified five-page synthetic PDF. Record structural reliability and independently checked visual accuracy separately; make no database or cloud request.
4. Update security/testing documentation and append actual results to the session log. Review the complete diff, secrets, compilation, Git status, and remaining native-processing and access-control limitations; leave all work uncommitted.

## Phase 8 independent-review remediation (approved)

1. Make the vision response's `done` field accept only the boolean `true`, and classify HTTP status failures separately from timeouts and transport errors. Keep the public 502 contract and provider settings unchanged.
2. Extend mocked provider and API tests for strict completion, HTTP status, redirect, connection, timeout, and sanitized diagnostics; leave PostgreSQL and text-RAG behavior unchanged.
3. Reconcile current Phase 8 documentation with the recorded 15-request local evaluation while preserving historical log entries; append this session's actual verification results.
4. Run focused Phase 8 and full default suites, inspect the diff and worktree, check for secrets and whitespace errors, and report remaining local-development limits. Do not publish Git changes.

## Repository-wide documentation refinement (approved)

1. Inventory project Markdown, agent instructions, setup configuration, and the uncommitted Phase 8 files; verify current-facing claims against routes, settings, providers, storage, and tests.
2. Correct confirmed status and behavior errors in the README, architecture decisions, codebase map, roadmap, developer and testing guides, and security plan. Keep operational instructions and historical results intact.
3. Review commands, relative links, code fences, and the roadmap diagram; check consistency and avoid changing application source, tests, dependencies, or runtime configuration.
4. Run documentation and Git checks, inspect the complete working tree, and append only a dated verification note to the session log. Leave all changes unstaged and unpublished; remaining risks are documentation drift and unverified live-service behavior.

# Phase 9A.2 plan

1. Add migration 0003 with bounded, immutable interpretation rows, composite source identity, cascading deletion, and request-key uniqueness. Keep earlier migrations intact.
2. Add strict local Ollama digest verification and a service that loads a stored element/PDF, checks geometry, renders and hashes the region, runs a versioned task, then rechecks source state under a short database lock before insertion.
3. Add separately gated development routes and Pydantic response contracts for analyze, list, and retrieve; preserve Phase 8 and 9A.1 routes.
4. Add focused mocked and disposable PostgreSQL tests, update architecture/roadmap/security docs, and verify focused/full tests, compilation, diff, credentials, and Git state. Risks: Ollama cannot atomically bind a chat response to a digest; native PDF processing still lacks process isolation.

## Phase 9B.1 local visual semantic retrieval (approved, 2026-11-01)

1. Add `0004_visual_retrieval.sql` with exact interpretation and embedding-space foreign keys, model revision/dimensions, vector checks, text-format provenance, and cascade deletion. Keep earlier migrations unchanged.
2. Add an explicit one-interpretation indexing service that textualizes only bounded description/observations, embeds through the existing local provider, and atomically rechecks and persists the derived vector. Add collection-scoped compatible cosine search with deterministic source-provenance results.
3. Add separately gated development indexing/search routes, strict Pydantic requests, and sanitized errors. Do not change text search or run vision inference.
4. Add synthetic unit/API/PostgreSQL tests, update architecture/roadmap/database/testing/security docs, verify focused and full suites against a verified disposable database, perform one small local Ollama check if practical, and review compilation, diff, secrets, and Git state. Risks: unverified model interpretations remain searchable; development routes lack caller authorization.
