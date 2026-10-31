"""Development answer-route gating, contracts, and controlled failures."""

import asyncio
import logging
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from rag_ai.core.config import Settings
from rag_ai.embeddings.base import EmbeddingBatch, EmbeddingError
from rag_ai.generation.base import GeneratedAnswer, GenerationError
from rag_ai.main import create_app
from rag_ai.retrieval.models import RetrievedChunk
from rag_ai.storage.postgres import ModelNameMismatch, StorageError


def _chunk(text: str = "Synthetic relay disconnects power.") -> RetrievedChunk:
    return RetrievedChunk(
        document_id="doc-1", chunk_id="chunk-1", chunk_index=0,
        filename="manual.pdf", file_type="pdf", page_number=3,
        text=text, cosine_distance=0.1,
    )


class FakeEmbedding:
    def __init__(self) -> None:
        self.closed = False

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        return EmbeddingBatch(model_name="synthetic", model_revision="digest", dimensions=2,
                              vectors=[[1.0, 0.0]])

    def close(self) -> None:
        self.closed = True


class FakeGeneration:
    def __init__(self, result: GeneratedAnswer | None = None) -> None:
        self.result = result or GeneratedAnswer.model_validate({
            "status": "answered", "claims": [{"text": "The relay cuts power.", "source_ids": ["S1"]}],
        })
        self.closed = False
        self.calls = 0

    def generate(self, **kwargs) -> GeneratedAnswer:
        self.calls += 1
        return self.result

    def close(self) -> None:
        self.closed = True


class FakeStore:
    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = [_chunk()] if results is None else results
        self.calls: list[dict] = []

    def search(self, **kwargs) -> list[RetrievedChunk]:
        self.calls.append(kwargs)
        return self.results


def _post(settings: Settings, collection_id: object, payload: dict) -> httpx.Response:
    async def request() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(settings)), base_url="http://test",
        ) as client:
            return await client.post(f"/collections/{collection_id}/answer", json=payload)
    return asyncio.run(request())


def _settings(**kwargs) -> Settings:
    return Settings(
        app_env="development", enable_rag_answer_endpoint=True,
        database_url="postgresql://unused@127.0.0.1:5432/unused", _env_file=None, **kwargs,
    )


def _patch(monkeypatch: pytest.MonkeyPatch, embedding: FakeEmbedding,
           generation: FakeGeneration, store: FakeStore) -> None:
    monkeypatch.setattr("rag_ai.api.answers.OllamaEmbeddingProvider", lambda **kwargs: embedding)
    monkeypatch.setattr("rag_ai.api.answers.OllamaGenerationProvider", lambda **kwargs: generation)
    monkeypatch.setattr("rag_ai.api.answers.PostgresIndexStore", lambda dsn: store)


def test_answer_route_disabled_by_default_and_outside_development() -> None:
    path = "/collections/{collection_id}/answer"
    assert path not in create_app(Settings(_env_file=None)).openapi()["paths"]
    assert path not in create_app(Settings(app_env="test", enable_rag_answer_endpoint=True,
                                           _env_file=None)).openapi()["paths"]
    app = create_app(_settings())
    assert path in app.openapi()["paths"]
    assert "/collections/{collection_id}/search" not in app.openapi()["paths"]
    assert _post(Settings(_env_file=None), uuid4(), {"query": "synthetic"}).status_code == 404


def test_production_answer_opt_in_rejected() -> None:
    with pytest.raises(ValidationError, match="RAG answer endpoint cannot be enabled in production"):
        Settings(app_env="production", enable_rag_answer_endpoint=True, _env_file=None)


@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": "  "}, {"query": "x" * 1001},
    {"query": "valid", "top_k": 0}, {"query": "valid", "top_k": 6},
])
def test_answer_route_rejects_invalid_requests(payload: dict) -> None:
    assert _post(_settings(), uuid4(), payload).status_code == 422


def test_answer_route_requires_database() -> None:
    response = _post(Settings(app_env="development", enable_rag_answer_endpoint=True,
                              database_url=None, _env_file=None), uuid4(), {"query": "synthetic"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "database_unconfigured"


def test_answer_route_resolves_citations_and_closes_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    embedding, generation, store = FakeEmbedding(), FakeGeneration(), FakeStore()
    _patch(monkeypatch, embedding, generation, store)
    collection_id = uuid4()
    response = _post(_settings(), collection_id, {"query": "What cuts power?"})
    assert response.status_code == 200
    data = response.json()
    assert data["collection_id"] == str(collection_id)
    assert data["status"] == "answered"
    assert data["claims"][0]["citations"] == [{
        "source_id": "S1", "document_id": "doc-1", "chunk_id": "chunk-1",
        "filename": "manual.pdf", "file_type": "pdf", "page_number": 3,
    }]
    assert store.calls[0]["collection_id"] == collection_id
    assert embedding.closed and generation.closed


def test_empty_collection_abstains_without_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    embedding, generation, store = FakeEmbedding(), FakeGeneration(), FakeStore([])
    _patch(monkeypatch, embedding, generation, store)
    response = _post(_settings(), uuid4(), {"query": "Unsupported?"})
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["claims"] == []
    assert generation.calls == 0
    assert embedding.closed and generation.closed


@pytest.mark.parametrize("failure,status,code", [
    ("embedding", 502, "embedding_unavailable"),
    ("database", 503, "database_unavailable"),
    ("model", 409, "embedding_model_mismatch"),
    ("generation", 502, "generation_invalid"),
    ("citation", 502, "generation_invalid"),
    ("context", 413, "evidence_too_large"),
])
def test_answer_route_controls_errors_and_closes_clients(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    failure: str, status: int, code: str,
) -> None:
    class FailingEmbedding(FakeEmbedding):
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            if failure == "embedding":
                raise EmbeddingError("private synthetic query")
            return super().embed(texts)

    class FailingStore(FakeStore):
        def search(self, **kwargs) -> list[RetrievedChunk]:
            if failure == "database":
                raise StorageError("private synthetic document")
            if failure == "model":
                raise ModelNameMismatch("private synthetic document")
            if failure == "context":
                return [_chunk("private synthetic document " * 100)]
            return super().search(**kwargs)

    class FailingGeneration(FakeGeneration):
        def generate(self, **kwargs) -> GeneratedAnswer:
            if failure == "generation":
                raise GenerationError("private synthetic document")
            if failure == "citation":
                return GeneratedAnswer.model_validate({
                    "status": "answered", "claims": [{"text": "Wrong", "source_ids": ["S99"]}],
                })
            return super().generate(**kwargs)

    embedding, generation, store = FailingEmbedding(), FailingGeneration(), FailingStore()
    _patch(monkeypatch, embedding, generation, store)
    with caplog.at_level(logging.INFO):
        response = _post(_settings(), uuid4(), {"query": "private synthetic query"})
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert "private synthetic" not in response.text
    assert "private synthetic" not in caplog.text
    assert embedding.closed and generation.closed
