---
name: rag-codebase-navigation
description: Navigate RAG-AI architecture and data flow. Use for unfamiliar subsystems, architecture investigations, or changes spanning components.
---

# Codebase navigation

1. **Establish context.** Run `git status --short --branch` and `git rev-parse HEAD`; identify the repository. When Graphify is available, use `list_repositories` or `graph_stats` to check its repository identity, indexed snapshot, and availability. Consult `docs/CODEBASE_MAP.md` and relevant architecture decisions as needed.
2. **Locate relevant code.** Use `graphify_rank_files` for feature-level questions, `graphify_find` for known symbols, or `query_graph` with `skeleton=true` and a small budget for initial architecture questions. Use targeted local path and symbol searches when they are cheaper or Graphify is unavailable.
3. **Trace dependencies.** Use `graphify_node` for a targeted definition, `graphify_impact` for possible change impact, and `graphify_tests_for` for linked tests. Use `graphify_trace` only when a call path needs investigation. Trace relevant configuration, FastAPI, document processing, provider, and PostgreSQL boundaries; graph links are not complete coverage.
4. **Verify locally.** Read only the relevant current source files before changing code. Cross-check graph findings against the working tree and uncommitted modifications, then apply changes using the actual local codebase. Verify useful documentation claims against current code.
5. **Test and report.** Run focused tests and required regressions. Explain the code relationships in plain language, cite relevant files, separate implemented behavior from roadmap plans, and note documentation drift. Follow the nine-section DEVELOPMENT SESSION REPORT; briefly state whether Graphify was used, which files it identified, and any index-freshness limitations.

## Token efficiency

- Prefer one focused graph query over overlapping searches; limit results to needed files, symbols, and dependencies.
- Avoid fetching definitions already in context or dumping entire graph results without a concrete reason.
- Prefer direct local inspection when it uses fewer tokens. If Graphify is stale or insufficient, continue with targeted local searches.
- Never send credentials, private document contents, or local environment secrets to Graphify.
