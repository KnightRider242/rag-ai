"""Development answer-route gating, contracts, and controlled failures."""

import asyncio
import json
import logging
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from rag_ai.core.config import Settings
from rag_ai.embeddings.base import EmbeddingBatch, EmbeddingError
from rag_ai.generation.base import GeneratedAnswer, GenerationError
from rag_ai.generation.openai import OpenAIGenerationProvider
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


def test_local_generation_ignores_present_cloud_key(monkeypatch: pytest.MonkeyPatch) -> None:
    embedding, generation, store = FakeEmbedding(), FakeGeneration(), FakeStore()
    _patch(monkeypatch, embedding, generation, store)
    cloud_calls = []
    monkeypatch.setattr("rag_ai.api.answers.OpenAIGenerationProvider",
                        lambda **kwargs: cloud_calls.append(kwargs))
    response = _post(_settings(openai_api_key="synthetic-test-key"), uuid4(), {"query": "Local?"})
    assert response.status_code == 200
    assert generation.calls == 1 and cloud_calls == []


@pytest.mark.parametrize("enabled,allowlisted", [(False, True), (True, False)])
def test_cloud_denied_before_retrieval_or_outbound_request(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, allowlisted: bool,
) -> None:
    collection_id = uuid4()
    attempted = []
    monkeypatch.setattr("rag_ai.api.answers.OllamaEmbeddingProvider",
                        lambda **kwargs: attempted.append("embedding"))
    monkeypatch.setattr("rag_ai.api.answers.OpenAIGenerationProvider",
                        lambda **kwargs: attempted.append("cloud"))
    settings = _settings(
        generation_provider="openai", enable_cloud_generation=enabled,
        cloud_generation_collection_ids=[collection_id] if allowlisted else [],
        openai_api_key="synthetic-test-key",
    )
    response = _post(settings, collection_id, {"query": "Synthetic?"})
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "cloud_generation_forbidden"
    assert attempted == []


def test_authorized_cloud_generation_uses_existing_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection_id = uuid4()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={
            "object": "response", "status": "completed", "model": "gpt-6-luna",
            "output": [{"type": "message", "status": "completed", "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps({
                    "status": "answered", "claims": [{"text": "The relay cuts power.",
                                                    "source_ids": ["S1"]}],
                })},
            ]}],
        })

    embedding, store = FakeEmbedding(), FakeStore()
    monkeypatch.setattr("rag_ai.api.answers.OllamaEmbeddingProvider", lambda **kwargs: embedding)
    monkeypatch.setattr("rag_ai.api.answers.PostgresIndexStore", lambda dsn: store)
    cloud: list[OpenAIGenerationProvider] = []

    def create_cloud(**kwargs) -> OpenAIGenerationProvider:
        provider = OpenAIGenerationProvider(transport=httpx.MockTransport(handle), **kwargs)
        cloud.append(provider)
        return provider

    monkeypatch.setattr("rag_ai.api.answers.OpenAIGenerationProvider", create_cloud)
    settings = _settings(
        generation_provider="openai", enable_cloud_generation=True,
        cloud_generation_collection_ids=[collection_id], openai_api_key="synthetic-test-key",
    )
    response = _post(settings, collection_id, {"query": "What cuts power?"})
    assert response.status_code == 200
    assert response.json()["claims"][0]["citations"][0]["chunk_id"] == "chunk-1"
    assert store.calls[0]["collection_id"] == collection_id
    assert len(requests) == 1 and cloud[0]._client.is_closed and embedding.closed


def test_cloud_response_abstention_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    collection_id = uuid4()
    embedding, generation, store = FakeEmbedding(), FakeGeneration(GeneratedAnswer.model_validate({
        "status": "insufficient_evidence", "claims": [],
    })), FakeStore()
    monkeypatch.setattr("rag_ai.api.answers.OllamaEmbeddingProvider", lambda **kwargs: embedding)
    monkeypatch.setattr("rag_ai.api.answers.OpenAIGenerationProvider", lambda **kwargs: generation)
    monkeypatch.setattr("rag_ai.api.answers.PostgresIndexStore", lambda dsn: store)
    response = _post(_settings(
        generation_provider="openai", enable_cloud_generation=True,
        cloud_generation_collection_ids=[collection_id], openai_api_key="synthetic-test-key",
    ), collection_id, {"query": "Unsupported?"})
    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["claims"] == []
    assert generation.closed and embedding.closed


def test_cloud_provider_failure_is_sanitized_and_closed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    collection_id = uuid4()

    class FailingCloud(FakeGeneration):
        def generate(self, **kwargs) -> GeneratedAnswer:
            raise GenerationError("private synthetic provider detail")

    embedding, generation, store = FakeEmbedding(), FailingCloud(), FakeStore()
    monkeypatch.setattr("rag_ai.api.answers.OllamaEmbeddingProvider", lambda **kwargs: embedding)
    monkeypatch.setattr("rag_ai.api.answers.OpenAIGenerationProvider", lambda **kwargs: generation)
    monkeypatch.setattr("rag_ai.api.answers.PostgresIndexStore", lambda dsn: store)
    with caplog.at_level(logging.INFO):
        response = _post(_settings(
            generation_provider="openai", enable_cloud_generation=True,
            cloud_generation_collection_ids=[collection_id], openai_api_key="synthetic-test-key",
        ), collection_id, {"query": "private synthetic question"})
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "generation_invalid"
    assert "private synthetic" not in response.text and "private synthetic" not in caplog.text
    assert embedding.closed and generation.closed


def test_ollama_failure_never_falls_back_to_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingGeneration(FakeGeneration):
        def generate(self, **kwargs) -> GeneratedAnswer:
            raise GenerationError("synthetic local outage")

    embedding, generation, store = FakeEmbedding(), FailingGeneration(), FakeStore()
    _patch(monkeypatch, embedding, generation, store)
    cloud_calls = []
    monkeypatch.setattr("rag_ai.api.answers.OpenAIGenerationProvider",
                        lambda **kwargs: cloud_calls.append(kwargs))
    response = _post(_settings(
        enable_cloud_generation=True, cloud_generation_collection_ids=[uuid4()],
        openai_api_key="synthetic-test-key",
    ), uuid4(), {"query": "Synthetic?"})
    assert response.status_code == 502 and cloud_calls == []


def test_cloud_configuration_rejects_missing_key_bad_uuid_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY is required"):
        _settings(generation_provider="openai", enable_cloud_generation=True,
                  cloud_generation_collection_ids=[uuid4()])
    with pytest.raises(ValidationError):
        _settings(generation_provider="openai", openai_generation_model="unsupported")
    with pytest.raises(ValidationError, match="Cloud generation is available only in development"):
        Settings(app_env="production", generation_provider="openai", _env_file=None)
    monkeypatch.setenv("CLOUD_GENERATION_COLLECTION_IDS", '["not-a-uuid"]')
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_cloud_allowlist_parses_json_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    collection_id = uuid4()
    monkeypatch.setenv("CLOUD_GENERATION_COLLECTION_IDS", json.dumps([str(collection_id)]))
    settings = Settings(app_env="development", _env_file=None)
    assert settings.cloud_generation_collection_ids == {collection_id}
