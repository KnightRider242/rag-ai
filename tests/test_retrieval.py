"""Synthetic retrieval, adapter, and development API checks."""

import asyncio
import logging
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from rag_ai.core.config import Settings
from rag_ai.embeddings.base import EmbeddingBatch, EmbeddingError
from rag_ai.main import create_app
from rag_ai.retrieval.langchain import RAGAIBaseRetriever
from rag_ai.retrieval.models import RetrievedChunk, SearchRequest
from rag_ai.retrieval.service import RetrievalError, RetrievalService
from rag_ai.storage.postgres import (
    ModelDimensionMismatch, ModelNameMismatch, ModelRevisionMismatch, StorageError,
)


def _chunk(text: str = "Synthetic battery cooling text") -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc-1", chunk_id="chunk-1", chunk_index=0,
        filename="manual.pdf", file_type="pdf", page_number=2,
        text=text, cosine_distance=0.25,
    )


class FakeProvider:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = [1.0, 0.0] if vector is None else vector
        self.calls: list[list[str]] = []
        self.closed = False

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        self.calls.append(texts)
        return EmbeddingBatch(
            model_name="synthetic-model", model_revision="revision-1",
            dimensions=2, vectors=[self.vector],
        )

    def close(self) -> None:
        self.closed = True


class FakeStore:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = [_chunk()] if results is None else results
        self.calls: list[dict] = []

    def search(self, **kwargs: object) -> list[RetrievedChunk]:
        self.calls.append(kwargs)
        return self.results


@pytest.mark.parametrize("query,top_k", [
    ("", 5), ("  \n", 5), ("x" * 2001, 5), ("valid", 0), ("valid", 21),
])
def test_query_and_top_k_validation(query: str, top_k: int) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query=query, top_k=top_k)


def test_service_embeds_exactly_one_query_and_forwards_model_metadata() -> None:
    provider, store, collection_id = FakeProvider(), FakeStore(), uuid4()
    result = RetrievalService(provider=provider, store=store).retrieve(
        collection_id=collection_id, query="  battery cooling  ", top_k=3,
    )
    assert provider.calls == [["battery cooling"]]
    assert store.calls == [{
        "collection_id": collection_id, "query_vector": [1.0, 0.0],
        "model_name": "synthetic-model", "model_revision": "revision-1",
        "dimensions": 2, "top_k": 3,
    }]
    assert result.results == [_chunk()]


@pytest.mark.parametrize("vectors,dimensions", [
    ([], 2), ([[1.0], [0.0]], 2), ([[1.0]], 2),
    ([[float("nan"), 0.0]], 2), ([[float("inf"), 0.0]], 2),
    ([[0.0, 0.0]], 2),
])
def test_service_rejects_invalid_query_embeddings(vectors: list[list[float]], dimensions: int) -> None:
    class InvalidProvider:
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            return EmbeddingBatch(
                model_name="synthetic", model_revision="revision", dimensions=dimensions,
                vectors=vectors,
            )

    store = FakeStore()
    with pytest.raises(RetrievalError):
        RetrievalService(provider=InvalidProvider(), store=store).retrieve(
            collection_id=uuid4(), query="synthetic question",
        )
    assert store.calls == []


def test_invalid_query_does_not_call_provider() -> None:
    provider = FakeProvider()
    with pytest.raises(ValidationError):
        RetrievalService(provider=provider, store=FakeStore()).retrieve(
            collection_id=uuid4(), query=" \t ",
        )
    assert provider.calls == []


def test_langchain_retriever_invoke_preserves_metadata() -> None:
    collection_id = uuid4()
    provider = FakeProvider()
    adapter = RAGAIBaseRetriever(
        service=RetrievalService(provider=provider, store=FakeStore()),
        collection_id=collection_id, top_k=4,
    )
    documents = adapter.invoke("synthetic question")
    assert len(documents) == 1
    assert documents[0].page_content == _chunk().text
    assert documents[0].metadata == {
        "document_id": "doc-1", "chunk_id": "chunk-1", "chunk_index": 0,
        "collection_id": str(collection_id), "filename": "manual.pdf",
        "file_type": "pdf", "page_number": 2, "distance": 0.25,
    }
    assert provider.calls == [["synthetic question"]]


def _post(settings: Settings, collection_id: object, payload: dict) -> httpx.Response:
    async def request() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(settings)), base_url="http://test",
        ) as client:
            return await client.post(f"/collections/{collection_id}/search", json=payload)
    return asyncio.run(request())


def test_search_route_disabled_by_default_and_outside_development() -> None:
    path = "/collections/{collection_id}/search"
    assert path not in create_app(Settings(_env_file=None)).openapi()["paths"]
    assert path not in create_app(Settings(app_env="test", enable_document_retrieval_endpoint=True)).openapi()["paths"]
    assert _post(Settings(_env_file=None), uuid4(), {"query": "synthetic"}).status_code == 404


def test_search_route_requires_explicit_development_flag() -> None:
    path = "/collections/{collection_id}/search"
    assert path not in create_app(Settings(app_env="development")).openapi()["paths"]
    assert path in create_app(Settings(
        app_env="development", enable_document_retrieval_endpoint=True,
    )).openapi()["paths"]


def test_production_cannot_enable_search_route() -> None:
    with pytest.raises(ValidationError, match="Document retrieval cannot be enabled in production"):
        Settings(app_env="production", enable_document_retrieval_endpoint=True)


@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": " \n "}, {"query": "x" * 2001},
    {"query": "valid", "top_k": 0}, {"query": "valid", "top_k": 21},
])
def test_search_route_rejects_invalid_input(payload: dict) -> None:
    settings = Settings(app_env="development", enable_document_retrieval_endpoint=True)
    assert _post(settings, uuid4(), payload).status_code == 422


def test_search_route_reports_missing_database() -> None:
    settings = Settings(app_env="development", enable_document_retrieval_endpoint=True, database_url=None)
    response = _post(settings, uuid4(), {"query": "synthetic"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "database_unconfigured"


def test_search_route_returns_ranked_response_and_closes_provider(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    provider, store = FakeProvider(), FakeStore([_chunk("private synthetic document content")])
    monkeypatch.setattr("rag_ai.api.retrieval.OllamaEmbeddingProvider", lambda **kwargs: provider)
    monkeypatch.setattr("rag_ai.api.retrieval.PostgresIndexStore", lambda dsn: store)
    settings = Settings(
        app_env="development", enable_document_retrieval_endpoint=True,
        database_url="postgresql://unused@127.0.0.1:5432/unused",
    )
    collection_id = uuid4()
    with caplog.at_level(logging.INFO):
        response = _post(settings, collection_id, {"query": "synthetic question"})
    assert response.status_code == 200
    assert response.json() == {"collection_id": str(collection_id), "results": [store.results[0].model_dump()]}
    assert provider.closed
    assert "private synthetic document content" not in caplog.text
    assert "synthetic question" not in caplog.text


@pytest.mark.parametrize("error,code,status", [
    (EmbeddingError("private synthetic query"), "embedding_unavailable", 502),
    (ModelNameMismatch("private synthetic query"), "embedding_model_mismatch", 409),
    (ModelRevisionMismatch("private synthetic query"), "embedding_revision_mismatch", 409),
    (ModelDimensionMismatch("private synthetic query"), "embedding_dimension_mismatch", 409),
    (StorageError("private synthetic query"), "database_unavailable", 503),
])
def test_search_route_controls_errors_and_logging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    error: Exception, code: str, status: int,
) -> None:
    class FailingStore:
        def search(self, **kwargs: object) -> list[RetrievedChunk]:
            raise error

    provider = FakeProvider()
    monkeypatch.setattr("rag_ai.api.retrieval.OllamaEmbeddingProvider", lambda **kwargs: provider)
    monkeypatch.setattr("rag_ai.api.retrieval.PostgresIndexStore", lambda dsn: FailingStore())
    settings = Settings(
        app_env="development", enable_document_retrieval_endpoint=True,
        database_url="postgresql://unused@127.0.0.1:5432/unused",
    )
    with caplog.at_level(logging.INFO):
        response = _post(settings, uuid4(), {"query": "private synthetic query"})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "private synthetic query" not in response.text
    assert "private synthetic query" not in caplog.text
    assert provider.closed


def test_search_route_provider_failure_has_no_cloud_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingProvider(FakeProvider):
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            raise EmbeddingError("Local provider unavailable.")

    provider = FailingProvider()
    monkeypatch.setattr("rag_ai.api.retrieval.OllamaEmbeddingProvider", lambda **kwargs: provider)
    monkeypatch.setattr("rag_ai.api.retrieval.PostgresIndexStore", lambda dsn: FakeStore())
    settings = Settings(
        app_env="development", enable_document_retrieval_endpoint=True,
        database_url="postgresql://unused@127.0.0.1:5432/unused",
    )
    response = _post(settings, uuid4(), {"query": "synthetic"})
    assert response.status_code == 502
    assert provider.closed
