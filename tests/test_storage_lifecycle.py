"""Versioned migration and deletion checks in disposable PostgreSQL schemas."""

from importlib.resources import files
from uuid import uuid4

import psycopg
import pytest

from rag_ai.documents.chunker import chunk_document
from rag_ai.documents.parser import parse_document
from rag_ai.embeddings.base import EmbeddingBatch
from rag_ai.storage.migrate import MigrationError, MigrationRunner
from rag_ai.storage.postgres import PostgresIndexStore, StorageError


def _document(text: str):
    content = text.encode("utf-8")
    parsed = parse_document(
        filename="synthetic.txt", content_type="text/plain", content=content,
        max_size_bytes=100_000, max_pdf_pages=10, max_extracted_characters=100_000,
    )
    return parsed, content, chunk_document(parsed)


def _save(store: PostgresIndexStore, collection_id, text: str):
    document, content, chunked = _document(text)
    store.save(
        collection_id=collection_id, document=document, original_content=content,
        chunked=chunked,
        first_batch=EmbeddingBatch(
            model_name="synthetic-model", model_revision="synthetic-revision",
            dimensions=2, vectors=[[0.1, 0.2] for _ in chunked.chunks],
        ),
    )
    return document


def test_fresh_migration_and_reexecution(isolated_database_url: str) -> None:
    runner = MigrationRunner(isolated_database_url)
    assert runner.status() == "empty"
    assert runner.upgrade() == 1
    assert runner.status() == "version 1"
    first = _save(PostgresIndexStore(isolated_database_url), uuid4(), "Synthetic persisted text")
    assert runner.upgrade() == 1
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute(
            "SELECT count(*) FROM indexed_documents WHERE document_id = %s", (first.document_id,),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM rag_ai_schema_migrations"
        ).fetchone() == (1,)


def test_unknown_schema_version_and_checksum_are_rejected(isolated_database_url: str) -> None:
    runner = MigrationRunner(isolated_database_url)
    runner.upgrade()
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute(
            "INSERT INTO rag_ai_schema_migrations(version, checksum) VALUES (%s, %s)",
            (2, "synthetic-future-version"),
        )
    with pytest.raises(MigrationError, match="newer"):
        runner.upgrade()
    with pytest.raises(StorageError):
        PostgresIndexStore(isolated_database_url).delete_document(
            collection_id=uuid4(), document_id="synthetic",
        )
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute("DELETE FROM rag_ai_schema_migrations WHERE version = 2")
        connection.execute("UPDATE rag_ai_schema_migrations SET checksum = 'synthetic-mismatch'")
    with pytest.raises(MigrationError, match="checksum"):
        runner.upgrade()


def test_unversioned_schema_requires_explicit_verified_adoption(isolated_database_url: str) -> None:
    runner = MigrationRunner(isolated_database_url)
    script = files("rag_ai.storage.migrations").joinpath("0001_initial.sql").read_text(encoding="utf-8")
    collection_id = uuid4()
    document, content, chunked = _document("Synthetic legacy source bytes")
    chunk = chunked.chunks[0]
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute(script)
        connection.execute("INSERT INTO collections(id) VALUES (%s)", (collection_id,))
        connection.execute(
            "INSERT INTO indexed_documents(collection_id, document_id, filename, file_type, original_content) "
            "VALUES (%s, %s, %s, %s, %s)",
            (collection_id, document.document_id, document.filename, document.file_type, content),
        )
        connection.execute(
            "INSERT INTO embedding_spaces(collection_id, model_name, model_revision, dimensions) "
            "VALUES (%s, %s, %s, %s)",
            (collection_id, "synthetic-model", "synthetic-revision", 2),
        )
        connection.execute(
            "INSERT INTO indexed_chunks(collection_id, chunk_id, document_id, chunk_index, content, "
            "chunk_size, chunk_overlap) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (collection_id, chunk.chunk_id, document.document_id, chunk.chunk_index, chunk.text,
             chunked.chunk_size, chunked.chunk_overlap),
        )
        connection.execute(
            "INSERT INTO chunk_embeddings(collection_id, chunk_id, model_name, dimensions, embedding) "
            "VALUES (%s, %s, %s, %s, %s::vector)",
            (collection_id, chunk.chunk_id, "synthetic-model", 2, "[0.1,0.2]"),
        )
    assert runner.status() == "unversioned Phase 4 schema"
    with pytest.raises(MigrationError, match="adopt-legacy"):
        runner.upgrade()
    with pytest.raises(StorageError):
        PostgresIndexStore(isolated_database_url).delete_document(
            collection_id=collection_id, document_id=document.document_id,
        )
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT count(*) FROM collections").fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM pg_catalog.pg_tables "
            "WHERE schemaname = current_schema() AND tablename = 'rag_ai_schema_migrations'"
        ).fetchone() == (0,)
    assert runner.adopt_legacy() == 1
    assert runner.upgrade() == 1
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute(
            "SELECT d.original_content, c.content, vector_dims(e.embedding) "
            "FROM indexed_documents d JOIN indexed_chunks c "
            "ON d.collection_id = c.collection_id AND d.document_id = c.document_id "
            "JOIN chunk_embeddings e ON e.collection_id = c.collection_id AND e.chunk_id = c.chunk_id "
            "WHERE d.collection_id = %s", (collection_id,),
        ).fetchone() == (content, chunk.text, 2)


def test_partial_unversioned_schema_is_not_modified(isolated_database_url: str) -> None:
    runner = MigrationRunner(isolated_database_url)
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute("CREATE TABLE collections(id UUID PRIMARY KEY)")
    assert runner.status() == "partial unversioned schema"
    with pytest.raises(MigrationError):
        runner.upgrade()
    with pytest.raises(MigrationError):
        runner.adopt_legacy()
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute(
            "SELECT count(*) FROM pg_catalog.pg_tables "
            "WHERE schemaname = current_schema() AND tablename = 'rag_ai_schema_migrations'"
        ).fetchone() == (0,)


def test_changed_unversioned_schema_is_not_adopted(isolated_database_url: str) -> None:
    script = files("rag_ai.storage.migrations").joinpath("0001_initial.sql").read_text(encoding="utf-8")
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute(script)
        connection.execute("ALTER TABLE embedding_spaces DROP COLUMN model_revision")
    with pytest.raises(MigrationError, match="columns differ"):
        MigrationRunner(isolated_database_url).adopt_legacy()
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute(
            "SELECT count(*) FROM pg_catalog.pg_tables "
            "WHERE schemaname = current_schema() AND tablename = 'rag_ai_schema_migrations'"
        ).fetchone() == (0,)


def test_delete_is_collection_scoped_and_cascades_without_deleting_model_metadata(
    isolated_database_url: str,
) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    store = PostgresIndexStore(isolated_database_url)
    first_collection, second_collection = uuid4(), uuid4()
    target = _save(store, first_collection, "Target synthetic document")
    unrelated = _save(store, first_collection, "Unrelated synthetic document")
    _save(store, second_collection, "Target synthetic document")

    assert store.delete_document(collection_id=first_collection, document_id=target.document_id) is None
    assert store.delete_document(collection_id=first_collection, document_id=target.document_id) is None
    with psycopg.connect(isolated_database_url) as connection:
        for table in ("indexed_documents", "indexed_chunks"):
            assert connection.execute(
                f"SELECT count(*) FROM {table} WHERE collection_id = %s AND document_id = %s",
                (first_collection, target.document_id),
            ).fetchone() == (0,)
        assert connection.execute(
            "SELECT count(*) FROM chunk_embeddings WHERE collection_id = %s", (first_collection,),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM indexed_documents WHERE collection_id = %s AND document_id = %s",
            (first_collection, unrelated.document_id),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM indexed_documents WHERE collection_id = %s AND document_id = %s",
            (second_collection, target.document_id),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM embedding_spaces WHERE collection_id = %s", (first_collection,),
        ).fetchone() == (1,)


def test_forged_cross_collection_foreign_key_is_rejected(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    store = PostgresIndexStore(isolated_database_url)
    first_collection, second_collection = uuid4(), uuid4()
    first = _save(store, first_collection, "First synthetic collection")
    _save(store, second_collection, "Second synthetic collection")
    first_chunk = chunk_document(first).chunks[0].chunk_id
    with psycopg.connect(isolated_database_url) as connection:
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            connection.execute(
                "INSERT INTO chunk_embeddings(collection_id, chunk_id, model_name, dimensions, embedding) "
                "VALUES (%s, %s, %s, %s, %s::vector)",
                (second_collection, first_chunk, "synthetic-model", 2, "[0.1,0.2]"),
            )
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute(
            "SELECT count(*) FROM chunk_embeddings WHERE collection_id = %s", (second_collection,),
        ).fetchone() == (1,)


def test_delete_failure_rolls_back_document_and_derived_rows(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    store = PostgresIndexStore(isolated_database_url)
    collection_id = uuid4()
    document = _save(store, collection_id, "Synthetic deletion rollback text")
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute(
            "CREATE TABLE delete_blocker (collection_id UUID NOT NULL, document_id TEXT NOT NULL, "
            "FOREIGN KEY (collection_id, document_id) "
            "REFERENCES indexed_documents(collection_id, document_id) ON DELETE RESTRICT)"
        )
        connection.execute(
            "INSERT INTO delete_blocker(collection_id, document_id) VALUES (%s, %s)",
            (collection_id, document.document_id),
        )
    with pytest.raises(StorageError):
        store.delete_document(collection_id=collection_id, document_id=document.document_id)
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute(
            "SELECT count(*) FROM indexed_documents WHERE collection_id = %s", (collection_id,),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM indexed_chunks WHERE collection_id = %s", (collection_id,),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM chunk_embeddings WHERE collection_id = %s", (collection_id,),
        ).fetchone() == (1,)
