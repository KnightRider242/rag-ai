"""Isolated PostgreSQL schemas for opt-in integration tests."""

import os
from collections.abc import Iterator
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest


class _RedactedDsn(str):
    def __repr__(self) -> str:
        return "<redacted test database URL>"


@pytest.fixture
def isolated_database_url() -> Iterator[str]:
    base_dsn = os.getenv("RAG_AI_TEST_DATABASE_URL")
    if not base_dsn:
        pytest.skip("Set RAG_AI_TEST_DATABASE_URL for isolated PostgreSQL tests.")
    schema = f"rag_ai_test_{uuid4().hex}"
    with psycopg.connect(base_dsn, connect_timeout=5) as connection:
        installed = connection.execute(
            "SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'"
        ).fetchone()
        if installed is None:
            pytest.skip("The test database must already have the pgvector extension installed.")
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        test_dsn = psycopg.conninfo.make_conninfo(base_dsn, options=f"-c search_path={schema},public")
        yield _RedactedDsn(test_dsn)
    finally:
        with psycopg.connect(base_dsn, connect_timeout=5) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
