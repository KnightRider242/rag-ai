"""Explicit, ordered SQL migrations for the Phase 4 PostgreSQL schema."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import re

import psycopg
from psycopg import sql


class MigrationError(RuntimeError):
    """The database schema cannot be safely used or upgraded."""


@dataclass(frozen=True)
class Migration:
    version: int
    script: str
    checksum: str


def _migrations() -> tuple[Migration, ...]:
    scripts = files("rag_ai.storage.migrations")
    migrations: list[Migration] = []
    for resource in scripts.iterdir():
        match = re.fullmatch(r"(\d{4})_[a-z0-9_]+\.sql", resource.name)
        if match:
            script = resource.read_text(encoding="utf-8")
            migrations.append(Migration(int(match.group(1)), script, sha256(script.encode("utf-8")).hexdigest()))
    migrations.sort(key=lambda migration: migration.version)
    if [migration.version for migration in migrations] != list(range(1, len(migrations) + 1)):
        raise MigrationError("Migration files must have consecutive versions starting at 0001.")
    return tuple(migrations)


def _schema_name(connection: psycopg.Connection) -> str:
    row = connection.execute("SELECT current_schema()").fetchone()
    if row is None or row[0] is None:
        raise MigrationError("Database connection has no active schema.")
    return row[0]


def _tables(connection: psycopg.Connection, schema: str) -> set[str]:
    return {row[0] for row in connection.execute(
        "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = %s", (schema,),
    )}


_CORE_TABLES = {
    "collections", "indexed_documents", "indexed_chunks", "embedding_spaces", "chunk_embeddings",
}

_COLUMNS: dict[str, dict[str, tuple[str, bool]]] = {
    "collections": {"id": ("uuid", True), "created_at": ("timestamp with time zone", True)},
    "indexed_documents": {
        "collection_id": ("uuid", True), "document_id": ("text", True),
        "filename": ("text", True), "file_type": ("text", True),
        "page_count": ("integer", False), "original_content": ("bytea", True),
        "created_at": ("timestamp with time zone", True),
    },
    "indexed_chunks": {
        "collection_id": ("uuid", True), "chunk_id": ("text", True),
        "document_id": ("text", True), "chunk_index": ("integer", True),
        "page_number": ("integer", False), "content": ("text", True),
        "chunk_size": ("integer", True), "chunk_overlap": ("integer", True),
    },
    "embedding_spaces": {
        "collection_id": ("uuid", True), "model_name": ("text", True),
        "model_revision": ("text", True), "dimensions": ("integer", True),
    },
    "chunk_embeddings": {
        "collection_id": ("uuid", True), "chunk_id": ("text", True),
        "model_name": ("text", True), "dimensions": ("integer", True),
        "embedding": ("vector", True),
    },
}

_REQUIRED_CONSTRAINTS: dict[str, tuple[str, ...]] = {
    "collections": ("PRIMARY KEY (id)",),
    "indexed_documents": (
        "PRIMARY KEY (collection_id, document_id)",
        "FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE",
    ),
    "indexed_chunks": (
        "PRIMARY KEY (collection_id, chunk_id)",
        "UNIQUE (collection_id, document_id, chunk_index)",
        "FOREIGN KEY (collection_id, document_id) REFERENCES indexed_documents(collection_id, document_id) ON DELETE CASCADE",
    ),
    "embedding_spaces": (
        "PRIMARY KEY (collection_id, model_name)",
        "UNIQUE (collection_id, model_name, dimensions)",
        "FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE",
    ),
    "chunk_embeddings": (
        "PRIMARY KEY (collection_id, chunk_id, model_name)",
        "FOREIGN KEY (collection_id, chunk_id) REFERENCES indexed_chunks(collection_id, chunk_id) ON DELETE CASCADE",
        "FOREIGN KEY (collection_id, model_name, dimensions) REFERENCES embedding_spaces(collection_id, model_name, dimensions) ON DELETE CASCADE",
    ),
}


def _verify_phase4_shape(connection: psycopg.Connection, schema: str) -> None:
    """Accept only the known unversioned Phase 4 shape before adoption."""
    if not _CORE_TABLES <= _tables(connection, schema):
        raise MigrationError("Phase 4 schema is incomplete; inspect it before migration.")
    for table, expected in _COLUMNS.items():
        actual = {row[0]: (row[1], row[2]) for row in connection.execute(
            "SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod), a.attnotnull "
            "FROM pg_catalog.pg_attribute a "
            "JOIN pg_catalog.pg_class t ON t.oid = a.attrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid = t.relnamespace "
            "WHERE n.nspname = %s AND t.relname = %s AND a.attnum > 0 AND NOT a.attisdropped",
            (schema, table),
        )}
        if actual != expected:
            raise MigrationError("Phase 4 schema columns differ from the supported baseline.")
        definitions = {row[0] for row in connection.execute(
            "SELECT pg_catalog.pg_get_constraintdef(c.oid) "
            "FROM pg_catalog.pg_constraint c "
            "JOIN pg_catalog.pg_class t ON t.oid = c.conrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid = t.relnamespace "
            "WHERE n.nspname = %s AND t.relname = %s",
            (schema, table),
        )}
        if not set(_REQUIRED_CONSTRAINTS[table]) <= definitions:
            raise MigrationError("Phase 4 schema constraints differ from the supported baseline.")
        if table == "embedding_spaces" and not any("dimensions" in item and "16000" in item for item in definitions):
            raise MigrationError("Embedding dimension constraint is missing.")
        if table == "chunk_embeddings" and not any("vector_dims(embedding) = dimensions" in item for item in definitions):
            raise MigrationError("Vector dimension constraint is missing.")
    index = connection.execute(
        "SELECT 1 FROM pg_catalog.pg_indexes WHERE schemaname = %s "
        "AND tablename = 'indexed_chunks' AND indexname = 'indexed_chunks_collection_document_idx'",
        (schema,),
    ).fetchone()
    if index is None:
        raise MigrationError("Phase 4 collection/document index is missing.")


def _applied(connection: psycopg.Connection, schema: str) -> list[tuple[int, str]]:
    query = sql.SQL("SELECT version, checksum FROM {} ORDER BY version").format(
        sql.Identifier(schema, "rag_ai_schema_migrations"),
    )
    return list(connection.execute(query))


def _validate_history(applied: list[tuple[int, str]], migrations: tuple[Migration, ...]) -> int:
    if [version for version, _ in applied] != list(range(1, len(applied) + 1)):
        raise MigrationError("Database migration history has a gap or invalid version.")
    if len(applied) > len(migrations):
        raise MigrationError("Database schema version is newer than this application supports.")
    for (version, checksum), migration in zip(applied, migrations, strict=False):
        if version != migration.version or checksum != migration.checksum:
            raise MigrationError("Database migration checksum differs from the application.")
    return len(applied)


def require_current_schema(connection: psycopg.Connection) -> None:
    """Refuse persistence against absent, stale, or unknown schemas."""
    schema = _schema_name(connection)
    if "rag_ai_schema_migrations" not in _tables(connection, schema):
        raise MigrationError("Database schema has not been migrated.")
    applied = _applied(connection, schema)
    if _validate_history(applied, _migrations()) != len(_migrations()):
        raise MigrationError("Database schema migration is required.")


class MigrationRunner:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def status(self) -> str:
        with psycopg.connect(self.dsn, connect_timeout=5) as connection:
            schema = _schema_name(connection)
            tables = _tables(connection, schema)
            if "rag_ai_schema_migrations" in tables:
                version = _validate_history(_applied(connection, schema), _migrations())
                return f"version {version}"
            if _CORE_TABLES <= tables:
                return "unversioned Phase 4 schema"
            if _CORE_TABLES & tables:
                return "partial unversioned schema"
            return "empty"

    def upgrade(self) -> int:
        migrations = _migrations()
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                with connection.transaction():
                    connection.execute("SELECT pg_catalog.pg_advisory_xact_lock(74243004)")
                    schema = _schema_name(connection)
                    tables = _tables(connection, schema)
                    if "rag_ai_schema_migrations" not in tables:
                        if _CORE_TABLES & tables:
                            raise MigrationError(
                                "Unversioned Phase 4 schema found. Back up the database and run adopt-legacy explicitly."
                            )
                        connection.execute(
                            "CREATE TABLE rag_ai_schema_migrations ("
                            "version INTEGER PRIMARY KEY CHECK (version > 0), "
                            "checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                        )
                        applied: list[tuple[int, str]] = []
                    else:
                        applied = _applied(connection, schema)
                        if not applied and _CORE_TABLES & tables:
                            raise MigrationError("Migration history is empty but Phase 4 tables exist.")
                    version = _validate_history(applied, migrations)
                    for migration in migrations[version:]:
                        connection.execute(migration.script)
                        connection.execute(
                            "INSERT INTO rag_ai_schema_migrations(version, checksum) VALUES (%s, %s)",
                            (migration.version, migration.checksum),
                        )
                    _verify_phase4_shape(connection, schema)
                    return len(migrations)
        except psycopg.Error as exc:
            raise MigrationError("Database migration failed; inspect the schema and connection.") from exc

    def adopt_legacy(self) -> int:
        """Register a verified unversioned baseline without altering its core tables or data."""
        migrations = _migrations()
        try:
            with psycopg.connect(self.dsn, connect_timeout=5) as connection:
                with connection.transaction():
                    connection.execute("SELECT pg_catalog.pg_advisory_xact_lock(74243004)")
                    schema = _schema_name(connection)
                    tables = _tables(connection, schema)
                    if "rag_ai_schema_migrations" in tables:
                        raise MigrationError("Database already has migration history; use upgrade.")
                    _verify_phase4_shape(connection, schema)
                    connection.execute(
                        "CREATE TABLE rag_ai_schema_migrations ("
                        "version INTEGER PRIMARY KEY CHECK (version > 0), "
                        "checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                    )
                    connection.execute(
                        "INSERT INTO rag_ai_schema_migrations(version, checksum) VALUES (%s, %s)",
                        (migrations[0].version, migrations[0].checksum),
                    )
                    return migrations[0].version
        except psycopg.Error as exc:
            raise MigrationError("Legacy schema adoption failed; inspect the database.") from exc
