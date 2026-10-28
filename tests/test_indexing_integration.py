"""Opt-in real PostgreSQL and local Ollama checks using synthetic documents."""

import asyncio
import os
from uuid import uuid4

import httpx
import psycopg
import pytest
import pymupdf

from rag_ai.core.config import Settings
from rag_ai.documents.chunker import chunk_document
from rag_ai.documents.indexer import index_document
from rag_ai.documents.parser import parse_document
from rag_ai.embeddings.base import EmbeddingBatch, EmbeddingError
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.storage.migrate import MigrationRunner
from rag_ai.storage.postgres import ModelDimensionMismatch, ModelRevisionMismatch, PostgresIndexStore, StorageError
from rag_ai.main import create_app


def _parsed(text: str):
    content = text.encode("utf-8")
    document = parse_document(
        filename="synthetic.txt", content_type="text/plain", content=content,
        max_size_bytes=100_000, max_pdf_pages=10, max_extracted_characters=100_000,
    )
    return document, content, chunk_document(document)


@pytest.fixture
def database_url(isolated_database_url: str) -> str:
    MigrationRunner(isolated_database_url).upgrade()
    return isolated_database_url


def test_collection_scoped_vectors_and_model_dimensions(database_url: str) -> None:
    store = PostgresIndexStore(database_url)
    document, content, chunked = _parsed("Synthetic collection isolation text")
    first, second = uuid4(), uuid4()
    try:
        for collection_id in (first, second):
            store.save(
                collection_id=collection_id, document=document, original_content=content,
                chunked=chunked,
                first_batch=EmbeddingBatch(model_name="synthetic-model", model_revision="synthetic-revision", dimensions=3,
                                          vectors=[[0.1, 0.2, 0.3]]),
            )
        with psycopg.connect(database_url) as connection:
            for collection_id in (first, second):
                row = connection.execute(
                    "SELECT d.original_content, s.model_revision, s.dimensions, vector_dims(e.embedding) "
                    "FROM indexed_documents d JOIN embedding_spaces s ON s.collection_id = d.collection_id "
                    "JOIN indexed_chunks c ON c.collection_id = d.collection_id AND c.document_id = d.document_id "
                    "JOIN chunk_embeddings e ON e.collection_id = c.collection_id AND e.chunk_id = c.chunk_id "
                    "WHERE d.collection_id = %s", (collection_id,),
                ).fetchone()
                assert row == (content, "synthetic-revision", 3, 3)
            count = connection.execute(
                "SELECT count(*) FROM indexed_documents WHERE collection_id = %s", (uuid4(),)
            ).fetchone()
            assert count == (0,)
    finally:
        with psycopg.connect(database_url) as connection:
            connection.execute("DELETE FROM collections WHERE id = ANY(%s)", ([first, second],))


def test_model_dimension_conflict_rolls_back(database_url: str) -> None:
    store = PostgresIndexStore(database_url)
    collection_id = uuid4()
    first, first_bytes, first_chunks = _parsed("First synthetic text")
    second, second_bytes, second_chunks = _parsed("Second synthetic text")
    try:
        store.save(
            collection_id=collection_id, document=first, original_content=first_bytes,
            chunked=first_chunks,
            first_batch=EmbeddingBatch(model_name="synthetic-model", model_revision="synthetic-revision", dimensions=3,
                                      vectors=[[0.1, 0.2, 0.3]]),
        )
        with pytest.raises(ModelDimensionMismatch):
            store.save(
                collection_id=collection_id, document=second, original_content=second_bytes,
                chunked=second_chunks,
                first_batch=EmbeddingBatch(model_name="synthetic-model", model_revision="synthetic-revision", dimensions=2,
                                          vectors=[[0.1, 0.2]]),
            )
        with psycopg.connect(database_url) as connection:
            row = connection.execute(
                "SELECT count(*) FROM indexed_documents WHERE collection_id = %s", (collection_id,)
            ).fetchone()
            assert row == (1,)
        with pytest.raises(ModelRevisionMismatch):
            store.save(
                collection_id=collection_id, document=second, original_content=second_bytes,
                chunked=second_chunks,
                first_batch=EmbeddingBatch(model_name="synthetic-model", model_revision="changed-revision",
                                          dimensions=3, vectors=[[0.1, 0.2, 0.3]]),
            )
    finally:
        with psycopg.connect(database_url) as connection:
            connection.execute("DELETE FROM collections WHERE id = %s", (collection_id,))


def test_pdf_page_metadata_survives_persistence(database_url: str) -> None:
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "First synthetic page")
    pdf.new_page()
    pdf.new_page().insert_text((72, 72), "Third synthetic page")
    content = pdf.tobytes()
    pdf.close()
    document = parse_document(
        filename="synthetic.pdf", content_type="application/pdf", content=content,
        max_size_bytes=100_000, max_pdf_pages=10, max_extracted_characters=100_000,
    )
    chunked = chunk_document(document)
    collection_id = uuid4()
    store = PostgresIndexStore(database_url)
    try:
        store.save(
            collection_id=collection_id, document=document, original_content=content,
            chunked=chunked,
            first_batch=EmbeddingBatch(
                model_name="synthetic-model", model_revision="synthetic-revision", dimensions=2,
                vectors=[[0.1, 0.2] for _ in chunked.chunks],
            ),
        )
        with psycopg.connect(database_url) as connection:
            rows = connection.execute(
                "SELECT c.page_number, c.content, d.file_type, d.page_count "
                "FROM indexed_chunks c JOIN indexed_documents d "
                "ON d.collection_id = c.collection_id AND d.document_id = c.document_id "
                "WHERE c.collection_id = %s ORDER BY c.chunk_index", (collection_id,),
            ).fetchall()
            assert rows == [(1, "First synthetic page", "pdf", 3),
                            (3, "Third synthetic page", "pdf", 3)]
    finally:
        with psycopg.connect(database_url) as connection:
            connection.execute("DELETE FROM collections WHERE id = %s", (collection_id,))


def test_later_embedding_failure_leaves_no_persisted_document(database_url: str) -> None:
    document, content, _ = _parsed("Synthetic rollback text. " * 160)
    chunked = chunk_document(document, chunk_size=64, chunk_overlap=0)
    assert len(chunked.chunks) > 16
    collection_id = uuid4()
    calls = 0

    class FailingProvider:
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise EmbeddingError("Synthetic local provider failure.")
            return EmbeddingBatch(model_name="synthetic-model", model_revision="synthetic-revision",
                                  dimensions=2, vectors=[[0.1, 0.2] for _ in texts])

    try:
        with pytest.raises(EmbeddingError):
            index_document(
                collection_id=collection_id, document=document, original_content=content,
                chunked=chunked, provider=FailingProvider(), store=PostgresIndexStore(database_url),
                max_index_chunks=500, max_embedding_elements=100_000,
            )
        with psycopg.connect(database_url) as connection:
            assert connection.execute(
                "SELECT count(*) FROM collections WHERE id = %s", (collection_id,),
            ).fetchone() == (0,)
            assert connection.execute(
                "SELECT count(*) FROM pg_catalog.pg_class c "
                "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relname = 'rag_ai_index_stage' AND n.nspname LIKE 'pg_temp_%'"
            ).fetchone() == (0,)
    finally:
        with psycopg.connect(database_url) as connection:
            connection.execute("DELETE FROM collections WHERE id = %s", (collection_id,))


def test_final_write_failure_preserves_previous_index(database_url: str) -> None:
    document, content, chunked = _parsed("Synthetic original index text")
    collection_id = uuid4()
    store = PostgresIndexStore(database_url)
    batch = EmbeddingBatch(model_name="synthetic-model", model_revision="synthetic-revision",
                           dimensions=2, vectors=[[0.1, 0.2]])
    try:
        store.save(collection_id=collection_id, document=document, original_content=content,
                   chunked=chunked, first_batch=batch)
        duplicate = chunked.chunks[0].model_copy(update={"chunk_index": 1})
        invalid_chunks = chunked.model_copy(update={"chunks": [chunked.chunks[0], duplicate]})
        with pytest.raises(StorageError):
            store.save(
                collection_id=collection_id, document=document, original_content=content,
                chunked=invalid_chunks,
                first_batch=EmbeddingBatch(
                    model_name="synthetic-model", model_revision="synthetic-revision",
                    dimensions=2, vectors=[[0.1, 0.2], [0.3, 0.4]],
                ),
            )
        with psycopg.connect(database_url) as connection:
            assert connection.execute(
                "SELECT count(*), min(chunk_id) FROM indexed_chunks "
                "WHERE collection_id = %s AND document_id = %s", (collection_id, document.document_id),
            ).fetchone() == (1, chunked.chunks[0].chunk_id)
    finally:
        with psycopg.connect(database_url) as connection:
            connection.execute("DELETE FROM collections WHERE id = %s", (collection_id,))


def test_real_ollama_parse_chunk_embed_and_persist(database_url: str) -> None:
    if os.getenv("RAG_AI_RUN_OLLAMA_TESTS") != "1":
        pytest.skip("Set RAG_AI_RUN_OLLAMA_TESTS=1 for local Ollama integration.")
    store = PostgresIndexStore(database_url)
    document, content, chunked = _parsed("A small synthetic document for local embedding verification.")
    collection_id = uuid4()
    provider = OllamaEmbeddingProvider(
        base_url=os.getenv("RAG_AI_TEST_OLLAMA_URL", "http://127.0.0.1:11434"),
        model_name="embeddinggemma:300m", max_dimensions=4096,
    )
    try:
        result = index_document(
            collection_id=collection_id, document=document, original_content=content,
            chunked=chunked, provider=provider, store=store, max_index_chunks=500,
        )
        assert result.embedding_dimensions == 768
        assert result.embedding_model == "embeddinggemma:300m"
        assert len(result.embedding_model_revision) == 64
        with psycopg.connect(database_url) as connection:
            row = connection.execute(
                "SELECT d.document_id, s.model_name, s.model_revision, s.dimensions, vector_dims(e.embedding) "
                "FROM indexed_documents d JOIN embedding_spaces s ON s.collection_id = d.collection_id "
                "JOIN indexed_chunks c ON c.collection_id = d.collection_id AND c.document_id = d.document_id "
                "JOIN chunk_embeddings e ON e.collection_id = c.collection_id AND e.chunk_id = c.chunk_id "
                "WHERE d.collection_id = %s", (collection_id,),
            ).fetchone()
            assert row == (document.document_id, "embeddinggemma:300m",
                           result.embedding_model_revision, 768, 768)
    finally:
        provider.close()
        with psycopg.connect(database_url) as connection:
            connection.execute("DELETE FROM collections WHERE id = %s", (collection_id,))


def test_real_development_index_api(database_url: str) -> None:
    if os.getenv("RAG_AI_RUN_OLLAMA_TESTS") != "1":
        pytest.skip("Set RAG_AI_RUN_OLLAMA_TESTS=1 for local Ollama integration.")
    collection_id = uuid4()
    settings = Settings(
        app_env="development", enable_document_parse_endpoint=True,
        enable_document_index_endpoint=True, database_url=database_url,
    )

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(settings))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                f"/collections/{collection_id}/documents/index",
                files={"file": ("synthetic.txt", b"A synthetic API indexing document.", "text/plain")},
            )

    try:
        response = asyncio.run(request())
        assert response.status_code == 200
        body = response.json()
        assert body["collection_id"] == str(collection_id)
        assert body["embedding_model"] == "embeddinggemma:300m"
        assert len(body["embedding_model_revision"]) == 64
        assert body["embedding_dimensions"] == 768
        assert body["total_chunks"] == 1
        with psycopg.connect(database_url) as connection:
            row = connection.execute(
                "SELECT count(*), max(vector_dims(e.embedding)) "
                "FROM chunk_embeddings e WHERE e.collection_id = %s", (collection_id,),
            ).fetchone()
            assert row == (1, 768)
    finally:
        with psycopg.connect(database_url) as connection:
            connection.execute("DELETE FROM collections WHERE id = %s", (collection_id,))
