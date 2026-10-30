"""Opt-in pgvector retrieval tests; the fixture owns a disposable schema."""

from uuid import UUID, uuid4

import psycopg
import pytest
import pymupdf

from rag_ai.documents.chunker import chunk_document
from rag_ai.documents.parser import parse_document
from rag_ai.embeddings.base import EmbeddingBatch
from rag_ai.storage.migrate import MigrationRunner
from rag_ai.storage.postgres import (
    ModelDimensionMismatch, ModelNameMismatch, ModelRevisionMismatch,
    PostgresIndexStore, StorageError,
)


@pytest.fixture
def store(isolated_database_url: str) -> PostgresIndexStore:
    MigrationRunner(isolated_database_url).upgrade()
    return PostgresIndexStore(isolated_database_url)


def _index(
    store: PostgresIndexStore, collection_id: UUID, text: str, vector: list[float],
    *, model: str = "model-a", revision: str = "digest-a", filename: str = "synthetic.txt",
) -> tuple[str, str]:
    content = text.encode("utf-8")
    document = parse_document(
        filename=filename, content_type="text/plain", content=content,
        max_size_bytes=100_000, max_pdf_pages=10, max_extracted_characters=100_000,
    )
    chunks = chunk_document(document)
    assert len(chunks.chunks) == 1
    store.save(
        collection_id=collection_id, document=document, original_content=content,
        chunked=chunks, first_batch=EmbeddingBatch(
            model_name=model, model_revision=revision, dimensions=len(vector), vectors=[vector],
        ),
    )
    return document.document_id, chunks.chunks[0].chunk_id


def _search(store: PostgresIndexStore, collection_id: UUID, vector: list[float],
            *, model: str = "model-a", revision: str = "digest-a", top_k: int = 5):
    return store.search(
        collection_id=collection_id, query_vector=vector, model_name=model,
        model_revision=revision, dimensions=len(vector), top_k=top_k,
    )


def test_cosine_order_limit_ties_and_metadata(store: PostgresIndexStore) -> None:
    collection_id = uuid4()
    _index(store, collection_id, "Far synthetic", [0.0, 1.0])
    near_doc, near_chunk = _index(store, collection_id, "Near synthetic", [1.0, 0.0])
    _index(store, collection_id, "Middle synthetic", [1.0, 1.0])
    results = _search(store, collection_id, [1.0, 0.0])
    assert [item.text for item in results] == ["Near synthetic", "Middle synthetic", "Far synthetic"]
    assert results[0].document_id == near_doc
    assert results[0].chunk_id == near_chunk
    assert results[0].filename == "synthetic.txt"
    assert results[0].file_type == "txt"
    assert results[0].page_number is None
    assert results[0].cosine_distance == pytest.approx(0)
    assert len(_search(store, collection_id, [1.0, 0.0], top_k=2)) == 2
    assert [item.document_id for item in results] == [item.document_id for item in _search(store, collection_id, [1.0, 0.0])]

    tie_collection = uuid4()
    first, _ = _index(store, tie_collection, "Tie one", [1.0, 0.0])
    second, _ = _index(store, tie_collection, "Tie two", [1.0, 0.0])
    assert [item.document_id for item in _search(store, tie_collection, [1.0, 0.0])] == sorted([first, second])


def test_empty_collection_and_retained_model_metadata(store: PostgresIndexStore) -> None:
    collection_id = uuid4()
    assert _search(store, collection_id, [1.0, 0.0]) == []
    _index(store, collection_id, "Temporary synthetic", [1.0, 0.0], model="other")
    with psycopg.connect(store.dsn) as connection:
        connection.execute("DELETE FROM indexed_documents WHERE collection_id = %s", (collection_id,))
    assert _search(store, collection_id, [1.0, 0.0]) == []


def test_collection_isolation_and_model_compatibility(store: PostgresIndexStore) -> None:
    first, second = uuid4(), uuid4()
    _index(store, first, "First collection synthetic", [1.0, 0.0])
    _index(store, second, "Second collection synthetic", [1.0, 0.0], model="other", revision="other-revision")
    assert [item.text for item in _search(store, first, [1.0, 0.0])] == ["First collection synthetic"]
    with pytest.raises(ModelNameMismatch):
        _search(store, second, [1.0, 0.0])
    with pytest.raises(ModelRevisionMismatch):
        _search(store, first, [1.0, 0.0], revision="changed")
    with pytest.raises(ModelDimensionMismatch):
        _search(store, first, [1.0, 0.0, 0.0])
    assert _search(store, uuid4(), [1.0, 0.0]) == []


def test_mixed_dimensions_and_zero_vectors_are_excluded_before_cosine(store: PostgresIndexStore) -> None:
    collection_id = uuid4()
    _index(store, collection_id, "Compatible synthetic", [1.0, 0.0])
    _index(store, collection_id, "Three dimensions synthetic", [1.0, 0.0, 0.0], model="model-b")
    _index(store, collection_id, "Zero vector synthetic", [0.0, 0.0])
    assert [item.text for item in _search(store, collection_id, [1.0, 0.0])] == ["Compatible synthetic"]
    assert [item.text for item in _search(store, collection_id, [1.0, 0.0, 0.0], model="model-b")] == [
        "Three dimensions synthetic",
    ]


def test_pdf_page_metadata_in_search(store: PostgresIndexStore) -> None:
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Synthetic first page")
    pdf.new_page().insert_text((72, 72), "Synthetic second page")
    content = pdf.tobytes()
    pdf.close()
    document = parse_document(
        filename="synthetic.pdf", content_type="application/pdf", content=content,
        max_size_bytes=100_000, max_pdf_pages=10, max_extracted_characters=100_000,
    )
    chunks = chunk_document(document)
    collection_id = uuid4()
    store.save(
        collection_id=collection_id, document=document, original_content=content,
        chunked=chunks, first_batch=EmbeddingBatch(
            model_name="model-a", model_revision="digest-a", dimensions=2,
            vectors=[[1.0, 0.0], [0.0, 1.0]],
        ),
    )
    results = _search(store, collection_id, [0.0, 1.0])
    assert [item.page_number for item in results] == [2, 1]
    assert all(item.filename == "synthetic.pdf" and item.file_type == "pdf" for item in results)


def test_schema_version_rejected(isolated_database_url: str) -> None:
    with pytest.raises(StorageError, match="Database retrieval failed"):
        _search(PostgresIndexStore(isolated_database_url), uuid4(), [1.0, 0.0])


def test_database_failure_is_sanitized(monkeypatch: pytest.MonkeyPatch, store: PostgresIndexStore) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise psycopg.OperationalError("synthetic connection detail")

    with monkeypatch.context() as patch:
        patch.setattr("rag_ai.storage.postgres.psycopg.connect", fail)
        with pytest.raises(StorageError) as error:
            _search(store, uuid4(), [1.0, 0.0])
    assert "synthetic connection detail" not in str(error.value)
