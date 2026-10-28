"""Local embedding and indexing boundary tests with synthetic data."""

import asyncio
import json
from hashlib import sha256
import logging
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from rag_ai.core.config import Settings
from rag_ai.documents.chunker import chunk_document
from rag_ai.documents.indexer import IndexingError, index_document
from rag_ai.documents.parser import parse_document
from rag_ai.embeddings.base import EmbeddingBatch, EmbeddingError
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.main import create_app


def _document(text: str = "Synthetic indexing text"):
    content = text.encode("utf-8")
    document = parse_document(
        filename="example.txt", content_type="text/plain", content=content,
        max_size_bytes=1_000_000, max_pdf_pages=10, max_extracted_characters=1_000_000,
    )
    return document, content, chunk_document(document)


def _provider(response_body: dict, status_code: int = 200) -> OllamaEmbeddingProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{
                "name": "embeddinggemma:300m", "digest": "a" * 64,
            }]})
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
        assert body["model"] == "embeddinggemma:300m"
        assert body["truncate"] is False
        return httpx.Response(status_code, json=response_body)

    return OllamaEmbeddingProvider(
        base_url="http://127.0.0.1:11434", model_name="embeddinggemma:300m",
        max_dimensions=4096, transport=httpx.MockTransport(handler),
    )


def test_provider_uses_actual_response_dimensions() -> None:
    provider = _provider({"model": "embeddinggemma:300m", "embeddings": [[0.1, 0.2, 0.3]]})

    result = provider.embed(["synthetic text"])

    assert result.dimensions == 3
    assert result.model_revision == "a" * 64
    assert result.vectors == [[0.1, 0.2, 0.3]]


@pytest.mark.parametrize("payload", [
    {"model": "wrong-model", "embeddings": [[0.1]]},
    {"model": "embeddinggemma:300m", "embeddings": []},
    {"model": "embeddinggemma:300m", "embeddings": [[]]},
    {"model": "embeddinggemma:300m", "embeddings": [[float("nan")]]},
    {"model": "embeddinggemma:300m", "embeddings": [[0.1, 0.2, 0.3, 0.4]]},
])
def test_provider_rejects_invalid_responses(payload: dict) -> None:
    provider = _provider(payload)
    provider.max_dimensions = 3

    with pytest.raises(EmbeddingError):
        provider.embed(["synthetic text"])


def test_provider_rejects_http_failure_without_leaking_text() -> None:
    provider = _provider({"error": "private synthetic text"}, status_code=500)

    with pytest.raises(EmbeddingError) as error:
        provider.embed(["private synthetic text"])

    assert "private synthetic text" not in str(error.value)


def test_provider_rejects_oversized_response() -> None:
    provider = _provider({"model": "embeddinggemma:300m", "embeddings": [[0.1] * 5000]})
    provider.max_dimensions = 3

    with pytest.raises(EmbeddingError):
        provider.embed(["synthetic text"])


def test_provider_rejects_model_revision_change_during_embedding() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/api/tags":
            calls += 1
            return httpx.Response(200, json={"models": [{
                "name": "embeddinggemma:300m", "digest": ("a" if calls == 1 else "b") * 64,
            }]})
        return httpx.Response(200, json={"model": "embeddinggemma:300m", "embeddings": [[0.1]]})

    provider = OllamaEmbeddingProvider(
        base_url="http://127.0.0.1:11434", model_name="embeddinggemma:300m",
        max_dimensions=4096, transport=httpx.MockTransport(handler),
    )

    with pytest.raises(EmbeddingError, match="revision changed"):
        provider.embed(["synthetic text"])


@pytest.mark.parametrize("url", [
    "https://api.example.com", "http://192.168.1.2:11434", "http://127.0.0.1:11434/other",
    "http://user:secret@127.0.0.1:11434", "http://127.0.0.1:11434?x=1",
])
def test_settings_reject_nonlocal_or_ambiguous_ollama_urls(url: str) -> None:
    with pytest.raises(ValidationError, match="OLLAMA_BASE_URL"):
        Settings(ollama_base_url=url)


@pytest.mark.parametrize("limit", [0, 2_000_001])
def test_settings_reject_invalid_embedding_element_limits(limit: int) -> None:
    with pytest.raises(ValidationError, match="max_embedding_elements"):
        Settings(max_embedding_elements=limit)


def test_direct_provider_rejects_external_url_without_echoing_it() -> None:
    with pytest.raises(EmbeddingError) as error:
        OllamaEmbeddingProvider(
            base_url="https://external.example/private", model_name="embeddinggemma:300m",
            max_dimensions=4096,
        )
    assert "external.example" not in str(error.value)


def test_provider_rejects_redirect_without_contacting_destination(caplog: pytest.LogCaptureFixture) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://external.example/secret"})

    provider = OllamaEmbeddingProvider(
        base_url="http://127.0.0.1:11434", model_name="embeddinggemma:300m",
        max_dimensions=4096, transport=httpx.MockTransport(handler),
    )
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EmbeddingError) as error:
            provider.embed(["private synthetic document text"])
    assert requests == ["http://127.0.0.1:11434/api/tags"]
    assert "private synthetic document text" not in str(error.value) + caplog.text
    assert "external.example" not in str(error.value) + caplog.text


def test_local_failure_has_no_cloud_fallback(caplog: pytest.LogCaptureFixture) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        raise httpx.ConnectError("synthetic local failure")

    provider = OllamaEmbeddingProvider(
        base_url="http://127.0.0.1:11434", model_name="embeddinggemma:300m",
        max_dimensions=4096, transport=httpx.MockTransport(handler),
    )
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(EmbeddingError) as error:
            provider.embed(["private synthetic document text"])
    assert requests == ["http://127.0.0.1:11434/api/tags"]
    assert "private synthetic document text" not in str(error.value) + caplog.text


def test_index_chunk_cap_rejects_before_provider_or_store() -> None:
    document, content, _ = _document("a" * 150)
    chunked = chunk_document(document, chunk_size=64, chunk_overlap=16)

    class FailProvider:
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            raise AssertionError("provider must not be called")

    class FailStore:
        def save(self, **kwargs: object) -> None:
            raise AssertionError("store must not be called")

    with pytest.raises(IndexingError) as error:
        index_document(
            collection_id=uuid4(), document=document, original_content=content,
            chunked=chunked, provider=FailProvider(), store=FailStore(), max_index_chunks=1,
        )
    assert error.value.code == "index_chunk_limit"


def test_embedding_element_budget_rejects_after_dimension_probe() -> None:
    document, content, _ = _document("Synthetic budget text. " * 160)
    chunked = chunk_document(document, chunk_size=64, chunk_overlap=0)
    assert len(chunked.chunks) > 16
    calls: list[int] = []

    class CountingProvider:
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            calls.append(len(texts))
            return EmbeddingBatch(model_name="synthetic", model_revision="revision", dimensions=3,
                                  vectors=[[0.1, 0.2, 0.3] for _ in texts])

    class FailStore:
        def save(self, **kwargs: object) -> None:
            raise AssertionError("budget rejection must prevent persistence")

    with pytest.raises(IndexingError) as error:
        index_document(
            collection_id=uuid4(), document=document, original_content=content,
            chunked=chunked, provider=CountingProvider(), store=FailStore(),
            max_index_chunks=500, max_embedding_elements=len(chunked.chunks) * 3 - 1,
        )
    assert error.value.code == "embedding_element_limit"
    assert calls == [16]


def test_embedding_batches_stay_at_or_below_sixteen_on_success() -> None:
    document, content, _ = _document("Synthetic batch text. " * 160)
    chunked = chunk_document(document, chunk_size=64, chunk_overlap=0)
    assert len(chunked.chunks) > 32
    calls: list[int] = []

    class CountingProvider:
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            calls.append(len(texts))
            return EmbeddingBatch(model_name="synthetic", model_revision="revision", dimensions=2,
                                  vectors=[[0.1, 0.2] for _ in texts])

    class ConsumingStore:
        def save(self, **kwargs: object) -> None:
            batches = [kwargs["first_batch"], *kwargs["remaining_batches"]]
            assert sum(len(batch.vectors) for batch in batches) == len(chunked.chunks)

    index_document(
        collection_id=uuid4(), document=document, original_content=content,
        chunked=chunked, provider=CountingProvider(), store=ConsumingStore(),
        max_index_chunks=500, max_embedding_elements=100_000,
    )
    assert len(calls) > 2
    assert max(calls) == 16
    assert sum(calls) == len(chunked.chunks)


def test_embedding_batches_are_bounded_and_later_failure_stops_persistence() -> None:
    document, content, _ = _document("Synthetic batch text. " * 160)
    chunked = chunk_document(document, chunk_size=64, chunk_overlap=0)
    assert len(chunked.chunks) > 16
    calls: list[int] = []

    class FailingProvider:
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            calls.append(len(texts))
            if len(calls) == 2:
                raise EmbeddingError("Local embedding service failed.")
            return EmbeddingBatch(model_name="synthetic", model_revision="revision", dimensions=3,
                                  vectors=[[0.1, 0.2, 0.3] for _ in texts])

    class ConsumingStore:
        finalized = False

        def save(self, **kwargs: object) -> None:
            list(kwargs["remaining_batches"])
            self.finalized = True

    store = ConsumingStore()
    with pytest.raises(EmbeddingError):
        index_document(
            collection_id=uuid4(), document=document, original_content=content,
            chunked=chunked, provider=FailingProvider(), store=store,
            max_index_chunks=500, max_embedding_elements=100_000,
        )
    assert calls == [16, min(16, len(chunked.chunks) - 16)]
    assert not store.finalized


def test_index_response_uses_provider_dimensions_and_preserves_metadata() -> None:
    document, content, chunked = _document()
    collection_id = uuid4()

    class FakeProvider:
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            return EmbeddingBatch(model_name="embeddinggemma:300m", model_revision="a" * 64, dimensions=3,
                                  vectors=[[0.1, 0.2, 0.3] for _ in texts])

    class RecordingStore:
        saved: dict | None = None

        def save(self, **kwargs: object) -> None:
            self.saved = kwargs

    store = RecordingStore()
    result = index_document(
        collection_id=collection_id, document=document, original_content=content,
        chunked=chunked, provider=FakeProvider(), store=store, max_index_chunks=500,
    )

    assert result.collection_id == collection_id
    assert result.document_id == sha256(content).hexdigest()
    assert result.embedding_dimensions == 3
    assert result.embedding_model_revision == "a" * 64
    assert result.total_chunks == 1
    assert store.saved is not None and store.saved["original_content"] == content


def test_index_rejects_nonfinite_vectors_from_any_provider_before_storage() -> None:
    document, content, chunked = _document()

    class InvalidProvider:
        def embed(self, texts: list[str]) -> EmbeddingBatch:
            return EmbeddingBatch(
                model_name="synthetic-model", model_revision="synthetic-revision",
                dimensions=1, vectors=[[float("nan")] for _ in texts],
            )

    class FailStore:
        def save(self, **kwargs: object) -> None:
            raise AssertionError("invalid vectors must not be persisted")

    with pytest.raises(IndexingError, match="inconsistent vectors"):
        index_document(
            collection_id=uuid4(), document=document, original_content=content,
            chunked=chunked, provider=InvalidProvider(), store=FailStore(), max_index_chunks=500,
        )


def test_index_route_is_disabled_by_default_and_in_production() -> None:
    assert "/collections/{collection_id}/documents/index" not in create_app(Settings(_env_file=None)).openapi()["paths"]
    assert "/collections/{collection_id}/documents/index" not in create_app(
        Settings(app_env="production", enable_document_index_endpoint=False)
    ).openapi()["paths"]


def test_index_route_requires_both_development_flags() -> None:
    settings = Settings(app_env="development", enable_document_index_endpoint=True)
    assert "/collections/{collection_id}/documents/index" not in create_app(settings).openapi()["paths"]

    settings = Settings(app_env="development", enable_document_parse_endpoint=True,
                        enable_document_index_endpoint=True)
    assert "/collections/{collection_id}/documents/index" in create_app(settings).openapi()["paths"]


def test_production_cannot_enable_index_route() -> None:
    with pytest.raises(ValidationError, match="Document indexing cannot be enabled in production"):
        Settings(app_env="production", enable_document_index_endpoint=True)


def test_index_route_reports_missing_database_without_upload_processing() -> None:
    settings = Settings(app_env="development", enable_document_parse_endpoint=True,
                        enable_document_index_endpoint=True, database_url=None)

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(settings))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(f"/collections/{uuid4()}/documents/index",
                                     files={"file": ("example.txt", b"synthetic", "text/plain")})

    response = asyncio.run(request())
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "database_unconfigured"


def test_index_route_reuses_ingestion_validation() -> None:
    settings = Settings(
        app_env="development", enable_document_parse_endpoint=True,
        enable_document_index_endpoint=True,
        database_url="postgresql://unused@127.0.0.1:5432/unused",
    )

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(settings))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(f"/collections/{uuid4()}/documents/index",
                                     files={"file": ("bad.csv", b"synthetic", "text/csv")})

    response = asyncio.run(request())
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_file_type"


def test_index_route_returns_413_for_embedding_element_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            pass

        def embed(self, texts: list[str]) -> EmbeddingBatch:
            return EmbeddingBatch(model_name="synthetic", model_revision="revision", dimensions=3,
                                  vectors=[[0.1, 0.2, 0.3] for _ in texts])

        def close(self) -> None:
            pass

    class FailStore:
        def __init__(self, dsn: str) -> None:
            pass

        def save(self, **kwargs: object) -> None:
            raise AssertionError("over-budget document must not reach storage")

    monkeypatch.setattr("rag_ai.api.documents.OllamaEmbeddingProvider", FakeProvider)
    monkeypatch.setattr("rag_ai.api.documents.PostgresIndexStore", FailStore)
    settings = Settings(
        app_env="development", enable_document_parse_endpoint=True,
        enable_document_index_endpoint=True, max_embedding_elements=1,
        database_url="postgresql://unused@127.0.0.1:5432/unused",
    )

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(settings))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                f"/collections/{uuid4()}/documents/index",
                files={"file": ("synthetic.txt", b"Private synthetic document text", "text/plain")},
            )

    response = asyncio.run(request())
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "embedding_element_limit"
    assert "Private synthetic document text" not in response.text
