"""Synthetic visual semantic retrieval checks with optional disposable PostgreSQL."""

from hashlib import sha256
import math
import os
from uuid import uuid4

import psycopg
import pymupdf
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from rag_ai.core.config import Settings
from rag_ai.documents.visual_elements import extract_visual_elements
from rag_ai.embeddings.base import EmbeddingBatch, EmbeddingError
from rag_ai.main import create_app
from rag_ai.retrieval.visual import (
    TEXT_FORMAT_VERSION, VisualRetrievalError, VisualRetrievalService,
    interpretation_embedding_text,
)
from rag_ai.storage.interpretations import PostgresInterpretationStore, StoredInterpretation
from rag_ai.storage.migrate import MigrationError, MigrationRunner
from rag_ai.storage.postgres import ModelRevisionMismatch, PostgresIndexStore
from rag_ai.storage.visual import PostgresVisualStore
from rag_ai.storage.visual_retrieval import (
    PostgresVisualRetrievalStore, VisualIndexSourceChanged, VisualInterpretationNotFound,
)
from rag_ai.vision.base import VisionOutput

REVISION = "a" * 64


class FakeProvider:
    on_embed = None
    revision = REVISION
    def __init__(self, **kwargs):
        self.calls = []
        self.closed = False

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        self.calls.append(texts)
        if self.on_embed is not None:
            self.on_embed()
        text = texts[0].lower()
        vector = [1.0, 0.0] if "pump" in text else [0.0, 1.0]
        return EmbeddingBatch(model_name="synthetic-embedding", model_revision=self.revision,
                              dimensions=2, vectors=[vector])

    def close(self) -> None:
        self.closed = True


def _pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page(width=300, height=200)
    page.draw_rect(pymupdf.Rect(30, 40, 90, 100))
    content = pdf.tobytes()
    pdf.close()
    return content


def _seed(dsn: str, collection_id, content: bytes, description: str) -> StoredInterpretation:
    document_id = sha256(content).hexdigest()
    candidates = extract_visual_elements(filename="synthetic.pdf", content=content,
                                         max_size_bytes=100_000, max_pdf_pages=5)
    with psycopg.connect(dsn) as connection:
        connection.execute("INSERT INTO collections(id) VALUES (%s)", (collection_id,))
        connection.execute("INSERT INTO indexed_documents(collection_id, document_id, filename, "
                           "file_type, page_count, original_content) VALUES (%s, %s, 'synthetic.pdf', "
                           "'pdf', 1, %s)", (collection_id, document_id, content))
    PostgresVisualStore(dsn).replace(collection_id=collection_id, document_id=document_id,
                                     source_sha256=document_id, page_count=1, candidates=candidates)
    element = PostgresInterpretationStore(dsn).load_element(
        collection_id=collection_id, document_id=document_id, element_id=candidates[0].element_id)
    return PostgresInterpretationStore(dsn).insert(
        element=element, source_sha256=document_id, page_count=1, request_key=None,
        task="describe_figure", prompt_version="stored_visual_v1",
        model_name="synthetic-vision", model_digest="b" * 64,
        render_scale=1, rendered_image_sha256="c" * 64,
        output=VisionOutput(description=description, observations=["synthetic observation"]),
    )


def _service(dsn: str, provider: FakeProvider | None = None) -> VisualRetrievalService:
    return VisualRetrievalService(provider=provider or FakeProvider(),
                                  store=PostgresVisualRetrievalStore(dsn))


def test_textualization_is_deterministic_and_marks_model_output() -> None:
    record = StoredInterpretation(
        collection_id=uuid4(), document_id="a" * 64, element_id="b" * 64,
        interpretation_id=uuid4(), request_key=None, analysis_task="describe_figure",
        prompt_version="stored_visual_v1", model_name="vision", model_digest="c" * 64,
        render_scale=1, rendered_image_sha256="d" * 64, description="  Red pump  ",
        observations=["  label P1  "], limitations=["unreadable gauge"],
        evidence_kind="model_generated_interpretation", verification_status="unverified",
        created_at="2026-11-01T00:00:00Z",
    )
    text = interpretation_embedding_text(record)
    assert text == ("Unverified model-generated visual interpretation\n"
                    "Description: Red pump\nObservation: label P1")
    assert "unreadable gauge" not in text
    assert "model_generated_interpretation" not in text


class _StubStore:
    def __init__(self, record):
        self.record = record
        self.saved = []
        self.searches = []

    def load_interpretation(self, **kwargs):
        return self.record

    def index_interpretation(self, **kwargs):
        self.saved.append(kwargs)
        return False

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return []


@pytest.mark.parametrize("vectors", [[], [[0.0, 0.0]], [[math.nan, 1.0]],
                                     [[math.inf, 1.0]], [[1.0]], [[1.0, 0.0], [0.0, 1.0]]])
def test_invalid_vectors_never_reach_storage(vectors) -> None:
    class BadProvider:
        def embed(self, texts):
            return EmbeddingBatch(model_name="synthetic-embedding", model_revision=REVISION,
                                  dimensions=2, vectors=vectors)
    store = _StubStore(None)
    with pytest.raises(VisualRetrievalError):
        VisualRetrievalService(provider=BadProvider(), store=store).search(
            collection_id=uuid4(), query="pump")
    assert store.searches == []


def test_visual_retrieval_routes_are_separately_gated() -> None:
    path = f"/collections/{uuid4()}"
    for environment in ("production", "test", "development"):
        with TestClient(create_app(Settings(app_env=environment, _env_file=None))) as client:
            assert client.post(path + "/visual-search", json={"query": "pump"}).status_code == 404
            assert client.post(path + "/visual-interpretations/index", json={}).status_code == 404
    with pytest.raises(ValidationError, match="Visual retrieval cannot be enabled in production"):
        Settings(app_env="production", enable_visual_retrieval_endpoint=True, _env_file=None)
    with TestClient(create_app(Settings(app_env="development", enable_visual_retrieval_endpoint=True,
                                        _env_file=None))) as client:
        assert client.post(path + "/visual-search", json={"query": "pump"}).status_code == 503
        assert client.post(path + "/visual-search", json={"query": " ", "top_k": 1}).status_code == 422
        assert client.post(path + "/visual-search", json={"query": "pump", "prompt": "x"}).status_code == 422
        assert client.post(path + "/visual-interpretations/index", json={}).status_code == 422


def test_version_3_upgrade_preserves_text_and_visual_records(isolated_database_url: str, monkeypatch) -> None:
    from rag_ai.storage import migrate as migration_module
    first_three = migration_module._migrations()[:3]
    with monkeypatch.context() as patch:
        patch.setattr(migration_module, "_migrations", lambda: first_three)
        assert MigrationRunner(isolated_database_url).upgrade() == 3
        collection_id = uuid4()
        record = _seed(isolated_database_url, collection_id, _pdf(), "red pump")
        with psycopg.connect(isolated_database_url) as connection:
            connection.execute("INSERT INTO indexed_chunks (collection_id, chunk_id, document_id, "
                               "chunk_index, page_number, content, chunk_size, chunk_overlap) "
                               "VALUES (%s, 'synthetic-chunk', %s, 0, 1, 'synthetic text', 100, 0)",
                               (collection_id, record.document_id))
            connection.execute("INSERT INTO embedding_spaces(collection_id, model_name, model_revision, dimensions) "
                               "VALUES (%s, 'text-model', 'text-revision', 2)", (collection_id,))
            connection.execute("INSERT INTO chunk_embeddings(collection_id, chunk_id, model_name, dimensions, embedding) "
                               "VALUES (%s, 'synthetic-chunk', 'text-model', 2, '[0.1,0.2]'::vector)",
                               (collection_id,))
    assert MigrationRunner(isolated_database_url).upgrade() == 4
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT count(*) FROM visual_interpretations").fetchone() == (1,)
        assert connection.execute("SELECT content FROM indexed_chunks").fetchone() == ("synthetic text",)
        assert connection.execute("SELECT vector_dims(embedding) FROM chunk_embeddings").fetchone() == (2,)


def test_failed_version_4_upgrade_rolls_back(isolated_database_url: str, monkeypatch) -> None:
    from rag_ai.storage import migrate as migration_module
    first_three = migration_module._migrations()[:3]
    with monkeypatch.context() as patch:
        patch.setattr(migration_module, "_migrations", lambda: first_three)
        MigrationRunner(isolated_database_url).upgrade()
    bad_sql = "CREATE TABLE visual_interpretation_embeddings (id INTEGER); SELECT 1 / 0"
    bad_fourth = migration_module.Migration(4, bad_sql, sha256(bad_sql.encode()).hexdigest())
    with monkeypatch.context() as patch:
        patch.setattr(migration_module, "_migrations", lambda: (*first_three, bad_fourth))
        with pytest.raises(MigrationError):
            MigrationRunner(isolated_database_url).upgrade()
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT count(*) FROM rag_ai_schema_migrations").fetchone() == (3,)
        assert connection.execute("SELECT to_regclass('visual_interpretation_embeddings')").fetchone() == (None,)
    assert MigrationRunner(isolated_database_url).upgrade() == 4


def test_index_retry_ranking_scope_and_cascades(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    content = _pdf()
    first, second = uuid4(), uuid4()
    pump = _seed(isolated_database_url, first, content, "red pump")
    motor = PostgresInterpretationStore(isolated_database_url).insert(
        element=PostgresInterpretationStore(isolated_database_url).load_element(
            collection_id=first, document_id=pump.document_id, element_id=pump.element_id),
        source_sha256=pump.document_id, page_count=1, request_key=None,
        task="describe_figure", prompt_version="stored_visual_v1",
        model_name="synthetic-vision", model_digest="b" * 64,
        render_scale=1, rendered_image_sha256="c" * 64,
        output=VisionOutput(description="blue motor", observations=["synthetic observation"]),
    )
    other = _seed(isolated_database_url, second, content, "red pump")
    service = _service(isolated_database_url)
    for record in (pump, motor):
        indexed = service.index(collection_id=first, document_id=record.document_id,
                                element_id=record.element_id, interpretation_id=record.interpretation_id)
        assert indexed.already_indexed is False
        assert indexed.text_format_version == TEXT_FORMAT_VERSION
        assert indexed.text_sha256 == sha256(interpretation_embedding_text(record).encode()).hexdigest()
    repeated = service.index(collection_id=first, document_id=pump.document_id,
                             element_id=pump.element_id, interpretation_id=pump.interpretation_id)
    assert repeated.already_indexed is True
    results = service.search(collection_id=first, query="pump", top_k=2).results
    assert service.provider.calls[-1] == ["pump"]
    assert [item.interpretation_id for item in results] == [pump.interpretation_id, motor.interpretation_id]
    assert results[0].cosine_distance < results[1].cosine_distance
    assert results[0].page_number == 1 and results[0].x0 < results[0].x1
    assert results[0].description == "red pump"
    assert results[0].evidence_kind == "model_generated_interpretation"
    assert results[0].verification_status == "unverified"
    assert service.search(collection_id=second, query="pump").results == []
    with pytest.raises(VisualInterpretationNotFound):
        service.index(collection_id=second, document_id=pump.document_id,
                      element_id=pump.element_id, interpretation_id=pump.interpretation_id)
    with psycopg.connect(isolated_database_url) as connection:
        connection.execute("INSERT INTO embedding_spaces(collection_id, model_name, model_revision, dimensions) "
                           "VALUES (%s, 'synthetic-embedding', %s, 2)", (second, REVISION))
        connection.commit()
        with pytest.raises(psycopg.Error):
            connection.execute("INSERT INTO visual_interpretation_embeddings (collection_id, document_id, "
                               "element_id, interpretation_id, model_name, model_revision, dimensions, "
                               "text_format_version, text_sha256, embedding) VALUES "
                               "(%s, %s, %s, %s, 'synthetic-embedding', %s, 2, %s, %s, '[1,0]'::vector)",
                               (second, other.document_id, other.element_id, pump.interpretation_id,
                                REVISION, TEXT_FORMAT_VERSION, "d" * 64))
        connection.rollback()
        for vector in ("[0,0]", "[1,0,0]"):
            with pytest.raises(psycopg.Error):
                connection.execute("INSERT INTO visual_interpretation_embeddings (collection_id, document_id, "
                                   "element_id, interpretation_id, model_name, model_revision, dimensions, "
                                   "text_format_version, text_sha256, embedding) VALUES "
                                   "(%s, %s, %s, %s, 'synthetic-embedding', %s, 2, 'invalid-test', %s, %s::vector)",
                                   (first, pump.document_id, pump.element_id, pump.interpretation_id,
                                    REVISION, "d" * 64, vector))
            connection.rollback()
        connection.execute("DELETE FROM visual_interpretations WHERE collection_id = %s "
                           "AND interpretation_id = %s", (first, motor.interpretation_id))
        assert connection.execute("SELECT count(*) FROM visual_interpretation_embeddings WHERE collection_id = %s",
                                  (first,)).fetchone() == (1,)
    PostgresIndexStore(isolated_database_url).delete_document(
        collection_id=first, document_id=pump.document_id)
    assert service.search(collection_id=first, query="pump").results == []
    assert PostgresInterpretationStore(isolated_database_url).get(
        collection_id=second, document_id=other.document_id,
        element_id=other.element_id, interpretation_id=other.interpretation_id) is not None


def test_revision_mismatch_and_deletion_during_embedding_roll_back(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    collection_id = uuid4()
    record = _seed(isolated_database_url, collection_id, _pdf(), "red pump")
    provider = FakeProvider()
    service = _service(isolated_database_url, provider)
    service.index(collection_id=collection_id, document_id=record.document_id,
                  element_id=record.element_id, interpretation_id=record.interpretation_id)
    provider.revision = "b" * 64
    with pytest.raises(ModelRevisionMismatch):
        service.index(collection_id=collection_id, document_id=record.document_id,
                      element_id=record.element_id, interpretation_id=record.interpretation_id)
    with pytest.raises(ModelRevisionMismatch):
        service.search(collection_id=collection_id, query="pump")
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT count(*) FROM visual_interpretation_embeddings").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM embedding_spaces").fetchone() == (1,)
    collection_id_2 = uuid4()
    record_2 = _seed(isolated_database_url, collection_id_2, _pdf(), "red pump")
    def delete_during_embed():
        with psycopg.connect(isolated_database_url) as connection:
            connection.execute("DELETE FROM visual_interpretations WHERE collection_id = %s "
                               "AND interpretation_id = %s", (collection_id_2, record_2.interpretation_id))
    provider_2 = FakeProvider()
    provider_2.on_embed = delete_during_embed
    with pytest.raises(VisualIndexSourceChanged):
        _service(isolated_database_url, provider_2).index(
            collection_id=collection_id_2, document_id=record_2.document_id,
            element_id=record_2.element_id, interpretation_id=record_2.interpretation_id)
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT count(*) FROM visual_interpretation_embeddings "
                                  "WHERE collection_id = %s", (collection_id_2,)).fetchone() == (0,)


def test_source_replacement_cascades_visual_embeddings(isolated_database_url: str) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    collection_id = uuid4()
    record = _seed(isolated_database_url, collection_id, _pdf(), "red pump")
    service = _service(isolated_database_url)
    service.index(collection_id=collection_id, document_id=record.document_id,
                  element_id=record.element_id, interpretation_id=record.interpretation_id)
    PostgresVisualStore(isolated_database_url).replace(
        collection_id=collection_id, document_id=record.document_id,
        source_sha256=record.document_id, page_count=1, candidates=[])
    with psycopg.connect(isolated_database_url) as connection:
        assert connection.execute("SELECT count(*) FROM visual_interpretation_embeddings").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM visual_interpretations").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM indexed_documents").fetchone() == (1,)


def test_api_index_and_search_with_mocked_provider(isolated_database_url: str, monkeypatch) -> None:
    MigrationRunner(isolated_database_url).upgrade()
    collection_id = uuid4()
    record = _seed(isolated_database_url, collection_id, _pdf(), "red pump")
    monkeypatch.setattr("rag_ai.api.visual_retrieval.OllamaEmbeddingProvider", FakeProvider)
    settings = Settings(app_env="development", enable_visual_retrieval_endpoint=True,
                        database_url=SecretStr(isolated_database_url), _env_file=None)
    path = f"/collections/{collection_id}"
    body = {"document_id": record.document_id, "element_id": record.element_id,
            "interpretation_id": str(record.interpretation_id)}
    with TestClient(create_app(settings)) as client:
        indexed = client.post(path + "/visual-interpretations/index", json=body)
        assert indexed.status_code == 200
        assert client.post(path + "/visual-interpretations/index", json=body).json()["already_indexed"] is True
        found = client.post(path + "/visual-search", json={"query": "pump", "top_k": 1})
        assert found.status_code == 200
        assert found.json()["results"][0]["interpretation_id"] == str(record.interpretation_id)
        assert client.post(path + "/visual-interpretations/index", json={**body, "prompt": "x"}).status_code == 422
        assert client.post(path + "/visual-interpretations/index", json={**body, "document_id": "bad"}).status_code == 422
        assert client.post(path.replace(str(collection_id), str(uuid4())) + "/visual-search",
                           json={"query": "pump"}).json()["results"] == []
        class FailedProvider(FakeProvider):
            def embed(self, texts):
                raise EmbeddingError("private provider detail")
        monkeypatch.setattr("rag_ai.api.visual_retrieval.OllamaEmbeddingProvider", FailedProvider)
        failed = client.post(path + "/visual-search", json={"query": "pump"})
        assert failed.status_code == 502
        assert failed.json() == {"detail": {"code": "embedding_unavailable"}}
        assert "private" not in failed.text


def test_live_local_visual_retrieval(isolated_database_url: str) -> None:
    """Explicit opt-in smoke check; ranking quality is reported separately."""
    if os.getenv("RAG_AI_RUN_VISUAL_RETRIEVAL_LIVE") != "1":
        pytest.skip("Set RAG_AI_RUN_VISUAL_RETRIEVAL_LIVE=1 for local EmbeddingGemma.")
    from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
    MigrationRunner(isolated_database_url).upgrade()
    collection_id = uuid4()
    pump = _seed(isolated_database_url, collection_id, _pdf(), "red pump")
    motor = PostgresInterpretationStore(isolated_database_url).insert(
        element=PostgresInterpretationStore(isolated_database_url).load_element(
            collection_id=collection_id, document_id=pump.document_id, element_id=pump.element_id),
        source_sha256=pump.document_id, page_count=1, request_key=None,
        task="describe_figure", prompt_version="stored_visual_v1",
        model_name="synthetic-vision", model_digest="b" * 64,
        render_scale=1, rendered_image_sha256="c" * 64,
        output=VisionOutput(description="blue motor", observations=["synthetic observation"]),
    )
    provider = OllamaEmbeddingProvider(base_url="http://127.0.0.1:11434",
                                       model_name="embeddinggemma:300m", max_dimensions=4096)
    try:
        service = _service(isolated_database_url, provider)
        for record in (pump, motor):
            service.index(collection_id=collection_id, document_id=record.document_id,
                          element_id=record.element_id, interpretation_id=record.interpretation_id)
        results = service.search(collection_id=collection_id, query="red pump", top_k=2).results
        assert len(results) == 2
        assert {item.interpretation_id for item in results} == {
            pump.interpretation_id, motor.interpretation_id}
        print("live_visual_top_match=", results[0].interpretation_id == pump.interpretation_id)
    finally:
        provider.close()
