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
