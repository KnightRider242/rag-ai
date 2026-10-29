---
name: rag-migration-review
description: Review PostgreSQL schema and data preservation. Use only for migration, schema, or data-preservation changes.
---

# RAG-AI migration review

1. Inspect affected versioned SQL, migration runner, schema constraints, storage queries, and isolated-database tests.
2. Check ordering, compatibility, model name/digest/dimensions, collection keys, data-preservation assumptions, and rollback or recovery paths.
3. Require a verified backup strategy before potentially destructive operations. Never execute migrations or reset/delete Docker volumes without explicit authorization.
4. Prefer synthetic fixtures in disposable schemas when authorized. Report tested paths, unverified assumptions, and recovery risks under `AGENTS.md`.
