---
name: rag-change-verification
description: Verify RAG-AI runtime and test changes. Use when changing behavior, configuration, API contracts, providers, storage, or tests.
---

# RAG-AI change verification

1. Identify the changed behavior and its callers, API contracts, Pydantic models, dependencies, storage operations, and provider effects; mark irrelevant layers.
2. Check Windows, macOS, and Linux assumptions. Review tracked diffs and untracked files for accidental behavior changes.
3. Run relevant existing tests, then `git diff --check`. Expand testing when a runtime change or concrete risk warrants it; skip unnecessary full suites for documentation-only work.
4. Report actual results, skipped checks, and unresolved effects under the repository's reporting rules.
