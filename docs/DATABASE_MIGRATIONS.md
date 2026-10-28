# Database migrations and document lifecycle

Phase 4 uses ordered SQL files in `src/rag_ai/storage/migrations/`. The current baseline is `0001_initial.sql`. The explicit migration runner stores each applied version and the SHA-256 checksum of its SQL in `rag_ai_schema_migrations`. Indexing and deletion reject databases without the current supported version. The API does not apply migrations during startup or requests.

## New local database

Set `DATABASE_URL` in your local Git-ignored `.env`, start the PostgreSQL/pgvector Compose service, then run:

```bash
python -m rag_ai.storage status
python -m rag_ai.storage upgrade
python -m rag_ai.storage status
```

`upgrade` applies pending migrations in order and is safe to rerun. An unknown newer version, a missing version, or a checksum mismatch stops the command. Migration SQL is applied transactionally; migration files must not be edited after use. Add a new numbered file for a future schema change.

## Existing unversioned Phase 4 database

The earlier local Phase 4 setup created tables without a migration history. **Back up the database before adoption.** Run `python -m rag_ai.storage status`. If it reports `unversioned Phase 4 schema`, inspect the backup and run:

```bash
python -m rag_ai.storage adopt-legacy
python -m rag_ai.storage upgrade
python -m rag_ai.storage status
```

`adopt-legacy` checks the known Phase 4 table columns, essential constraints, and collection/document index before recording version 1. It does not rewrite or delete existing documents, chunks, vectors, or model metadata. A partial or changed unversioned schema is rejected; inspect and repair it manually from a backup rather than forcing adoption. Do not point automated tests at a database containing real documents. Live tests create and remove their own randomly named schemas.

## Retention and deletion

Indexed original bytes, chunks, and vectors remain in PostgreSQL until explicitly deleted; **there is no automatic retention expiry**. `PostgresIndexStore.delete_document(collection_id=..., document_id=...)` deletes a document only in the specified collection. Foreign-key cascades remove its chunks and embeddings in the same transaction. Other documents and shared embedding-model metadata remain. The method returns no existence signal, and there is no HTTP deletion endpoint or collection authorization yet. Use only synthetic or otherwise authorized documents in this development environment. Backups and database snapshots may retain older copies and require their own retention policy.
