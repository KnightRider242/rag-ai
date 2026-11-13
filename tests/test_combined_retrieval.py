"""Combined retrieval ranking, API, and disposable PostgreSQL checks."""

from hashlib import sha256
import os
from uuid import uuid4

import psycopg
import pymupdf
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from rag_ai.core.config import Settings
from rag_ai.documents.chunker import chunk_document
from rag_ai.documents.parser import parse_document
from rag_ai.documents.visual_elements import extract_visual_elements
from rag_ai.embeddings.base import EmbeddingBatch, EmbeddingError
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.generation.base import GeneratedAnswer, GenerationConfig
from rag_ai.main import create_app
from rag_ai.rag.context import ContextLimits
from rag_ai.rag.multimodal import CitationError, MultimodalRAGService
from rag_ai.retrieval.combined import CombinedRetrievalService
from rag_ai.retrieval.models import RetrievedChunk
from rag_ai.retrieval.visual import RetrievedVisualInterpretation, VisualRetrievalError, VisualRetrievalService
from rag_ai.storage.combined_retrieval import PostgresCombinedRetrievalStore
from rag_ai.storage.answer_sources import PostgresAnswerSourceResolver
from rag_ai.storage.interpretations import PostgresInterpretationStore
from rag_ai.storage.migrate import MigrationRunner
from rag_ai.storage.postgres import ModelDimensionMismatch, ModelNameMismatch, ModelRevisionMismatch, PostgresIndexStore
from rag_ai.storage.visual import PostgresVisualStore
from rag_ai.storage.visual_retrieval import PostgresVisualRetrievalStore
from rag_ai.vision.base import VisionOutput

REVISION = "a" * 64


class Provider:
    def __init__(self, **kwargs):
        self.calls = []
        self.closed = False
        self.vector = [1.0, 0.0]

    def embed(self, texts):
        self.calls.append(texts)
        return EmbeddingBatch(model_name="synthetic", model_revision=REVISION,
                              dimensions=len(self.vector), vectors=[self.vector])

    def close(self):
        self.closed = True


class Store:
    def __init__(self, texts=None, visuals=None):
        self.texts = texts or []
        self.visuals = visuals or []
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.texts, self.visuals


def _text(identity: str) -> RetrievedChunk:
    return RetrievedChunk(document_id=identity, chunk_id=identity, chunk_index=0,
                          filename="synthetic.txt", file_type="txt", page_number=None,
                          text="synthetic evidence", cosine_distance=0.1)


def _visual(identity: str, interpretation_id=None) -> RetrievedVisualInterpretation:
    return RetrievedVisualInterpretation(
        document_id=identity, filename="synthetic.pdf", element_id=identity,
        interpretation_id=interpretation_id or uuid4(), page_number=1,
        coordinate_system="pdf_displayed_points", x0=30, y0=40, x1=90, y1=100,
        element_type="vector_path", analysis_task="describe_figure",
        prompt_version="stored_visual_v1", interpretation_model_name="synthetic-vision",
        interpretation_model_digest="b" * 64, description="pump",
        observations=["red"], limitations=[], evidence_kind="model_generated_interpretation",
        verification_status="unverified", cosine_distance=0.2,
    )


def test_one_embedding_rrf_ties_dedup_and_limit():
    provider = Provider()
    store = Store([_text("b"), _text("a")], [_visual("v"), _visual("v"), _visual("w")])
    response = CombinedRetrievalService(provider=provider, store=store).search(
        collection_id=uuid4(), query=" pump ", top_k=3)
    assert provider.calls == [["pump"]]
    assert len(store.calls) == 1 and store.calls[0]["visual_limit"] == 100
    assert [r.evidence_type for r in response.results] == [
        "text_chunk", "visual_interpretation", "text_chunk"]
    assert [r.rank for r in response.results] == [1, 2, 3]
    assert [r.rrf_score for r in response.results] == [1 / 61, 1 / 61, 1 / 62]
    assert response.results[1].cosine_distance == 0.2
    assert len([r for r in CombinedRetrievalService(provider=provider, store=store).search(
        collection_id=uuid4(), query="pump", top_k=20).results
                if r.evidence_type == "visual_interpretation"]) == 2


@pytest.mark.parametrize("vector", [[float("nan"), 0], [0, 0], [float("inf"), 1]])
def test_invalid_vector_does_not_search(vector):
    provider, store = Provider(), Store()
    provider.vector = vector
    with pytest.raises(VisualRetrievalError):
        CombinedRetrievalService(provider=provider, store=store).search(
            collection_id=uuid4(), query="pump")
    assert store.calls == []


def test_route_gate_and_errors(monkeypatch):
    path = f"/collections/{uuid4()}/multimodal-search"
    assert TestClient(create_app(Settings(_env_file=None, app_env="development"))).post(
        path, json={"query": "pump"}).status_code == 404
    assert TestClient(create_app(Settings(_env_file=None, app_env="test",
        enable_multimodal_retrieval_endpoint=True))).post(path, json={"query": "pump"}).status_code == 404
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="production", enable_multimodal_retrieval_endpoint=True)
    settings = Settings(_env_file=None, app_env="development",
                        enable_multimodal_retrieval_endpoint=True)
    client = TestClient(create_app(settings))
    assert client.post(path, json={"query": " "}).status_code == 422
    assert client.post(path, json={"query": "pump"}).json()["detail"]["code"] == "database_unconfigured"
    settings = Settings(_env_file=None, app_env="development",
                        enable_multimodal_retrieval_endpoint=True,
                        database_url=SecretStr("postgresql://synthetic@localhost/test"))
    provider = Provider()
    monkeypatch.setattr("rag_ai.api.combined_retrieval.OllamaEmbeddingProvider", lambda **kwargs: provider)
    monkeypatch.setattr("rag_ai.api.combined_retrieval.PostgresCombinedRetrievalStore", lambda dsn: Store([_text("a")]))
    response = TestClient(create_app(settings)).post(path, json={"query": "pump"})
    assert response.status_code == 200 and response.json()["results"][0]["evidence_type"] == "text_chunk"
    assert provider.closed and provider.calls == [["pump"]]
    class FailedProvider(Provider):
        def embed(self, texts):
            raise EmbeddingError("confidential content")
    monkeypatch.setattr("rag_ai.api.combined_retrieval.OllamaEmbeddingProvider", lambda **kwargs: FailedProvider())
    failure = TestClient(create_app(settings)).post(path, json={"query": "pump"})
    assert failure.status_code == 502 and "confidential" not in failure.text


def _index_text(dsn, collection, text="synthetic text", model="synthetic", revision=REVISION,
                vector=None):
    content = text.encode()
    document = parse_document(filename="synthetic.txt", content_type="text/plain", content=content,
                              max_size_bytes=100_000, max_pdf_pages=5, max_extracted_characters=100_000)
    PostgresIndexStore(dsn).save(collection_id=collection, document=document,
        original_content=content, chunked=chunk_document(document),
        first_batch=EmbeddingBatch(model_name=model, model_revision=revision,
                                   dimensions=len(vector or [1.0, 0.0]),
                                   vectors=[vector or [1.0, 0.0]]))
    return document.document_id


def _index_visual(dsn, collection, *, second_element=False):
    pdf = pymupdf.open()
    page = pdf.new_page(width=300, height=200)
    page.draw_rect(pymupdf.Rect(30, 40, 90, 100))
    if second_element:
        page.draw_rect(pymupdf.Rect(150, 40, 210, 100))
    content = pdf.tobytes()
    pdf.close()
    document_id = sha256(content).hexdigest()
    candidates = extract_visual_elements(filename="synthetic.pdf", content=content,
                                         max_size_bytes=100_000, max_pdf_pages=5)
    with psycopg.connect(dsn) as connection:
        connection.execute("INSERT INTO collections(id) VALUES (%s) ON CONFLICT DO NOTHING", (collection,))
        connection.execute("INSERT INTO indexed_documents(collection_id, document_id, filename, "
                           "file_type, page_count, original_content) VALUES (%s, %s, 'synthetic.pdf', "
                           "'pdf', 1, %s)", (collection, document_id, content))
    PostgresVisualStore(dsn).replace(collection_id=collection, document_id=document_id,
                                     source_sha256=document_id, page_count=1, candidates=candidates)
    interpretation_store = PostgresInterpretationStore(dsn)
    element = interpretation_store.load_element(collection_id=collection, document_id=document_id,
                                                element_id=candidates[0].element_id)
    interpretation = interpretation_store.insert(
        element=element, source_sha256=document_id, page_count=1, request_key=None,
        task="describe_figure", prompt_version="stored_visual_v1", model_name="synthetic-vision",
        model_digest="b" * 64, render_scale=1, rendered_image_sha256="c" * 64,
        output=VisionOutput(description="red pump", observations=["synthetic observation"]),
    )
    VisualRetrievalService(provider=Provider(), store=PostgresVisualRetrievalStore(dsn)).index(
        collection_id=collection, document_id=document_id, element_id=element.element_id,
        interpretation_id=interpretation.interpretation_id)
    return document_id


def test_disposable_combined_search(isolated_database_url):
    dsn = isolated_database_url
    MigrationRunner(dsn).upgrade()
    collection, other = uuid4(), uuid4()
    assert PostgresCombinedRetrievalStore(dsn).search(collection_id=collection,
        query_vector=[1, 0], model_name="synthetic", model_revision=REVISION,
        dimensions=2, text_limit=20, visual_limit=100) == ([], [])
    _index_text(dsn, collection)
    service = CombinedRetrievalService(provider=Provider(), store=PostgresCombinedRetrievalStore(dsn))
    assert [r.evidence_type for r in service.search(collection_id=collection, query="pump").results] == ["text_chunk"]
    visual_document = _index_visual(dsn, collection)
    result = service.search(collection_id=collection, query="pump")
    assert [r.evidence_type for r in result.results] == ["text_chunk", "visual_interpretation"]
    visual = result.results[1]
    assert visual.document_id == visual_document and visual.page_number == 1
    assert visual.x0 < visual.x1 and visual.interpretation_model_digest == "b" * 64
    assert visual.verification_status == "unverified"
    assert service.search(collection_id=collection, query="pump", top_k=1).results[0].evidence_type == "text_chunk"
    assert service.search(collection_id=other, query="pump").results == []
    _index_visual(dsn, other)
    assert [r.evidence_type for r in service.search(collection_id=other, query="pump").results] == ["visual_interpretation"]
    with pytest.raises(ModelRevisionMismatch):
        PostgresCombinedRetrievalStore(dsn).search(collection_id=collection, query_vector=[1, 0],
            model_name="synthetic", model_revision="d" * 64, dimensions=2,
            text_limit=20, visual_limit=100)
    with pytest.raises(ModelDimensionMismatch):
        PostgresCombinedRetrievalStore(dsn).search(collection_id=collection, query_vector=[1, 0, 0],
            model_name="synthetic", model_revision=REVISION, dimensions=3,
            text_limit=20, visual_limit=100)
    with pytest.raises(ModelNameMismatch):
        PostgresCombinedRetrievalStore(dsn).search(collection_id=collection, query_vector=[1, 0],
            model_name="other", model_revision=REVISION, dimensions=2,
            text_limit=20, visual_limit=100)
    _index_text(dsn, collection, text="second model", model="other", revision="d" * 64)
    with pytest.raises(ModelNameMismatch):
        service.search(collection_id=collection, query="pump")


def test_answer_source_recheck_fails_closed_after_deletion_and_cross_collection(isolated_database_url):
    dsn = isolated_database_url
    MigrationRunner(dsn).upgrade()
    collection, other = uuid4(), uuid4()
    text_document = _index_text(dsn, collection)
    visual_document = _index_visual(dsn, collection)
    sources = CombinedRetrievalService(provider=Provider(),
        store=PostgresCombinedRetrievalStore(dsn)).search(
            collection_id=collection, query="pump", top_k=2).results
    resolver = PostgresAnswerSourceResolver(dsn)
    assert resolver.verify_sources(collection_id=collection, sources=sources)
    assert not resolver.verify_sources(collection_id=other, sources=sources)
    text_source = next(item for item in sources if item.evidence_type == "text_chunk")
    with psycopg.connect(dsn) as connection:
        connection.execute("UPDATE indexed_chunks SET content = %s WHERE collection_id = %s "
                           "AND document_id = %s AND chunk_id = %s",
                           ("changed synthetic text", collection, text_source.document_id, text_source.chunk_id))
    assert not resolver.verify_sources(collection_id=collection, sources=sources)
    with psycopg.connect(dsn) as connection:
        connection.execute("DELETE FROM indexed_documents WHERE collection_id = %s AND document_id = %s",
                           (collection, visual_document))
    assert not resolver.verify_sources(collection_id=collection, sources=sources)
    with psycopg.connect(dsn) as connection:
        connection.execute("DELETE FROM indexed_documents WHERE collection_id = %s AND document_id = %s",
                           (collection, text_document))
    assert not resolver.verify_sources(collection_id=collection, sources=sources)


def test_answer_source_recheck_fractional_pdf_coordinates(isolated_database_url):
    dsn = isolated_database_url
    MigrationRunner(dsn).upgrade()
    collection = uuid4()
    document_id = _index_visual(dsn, collection)
    with psycopg.connect(dsn) as connection:
        element_id = connection.execute(
            "SELECT element_id FROM visual_elements WHERE collection_id = %s AND document_id = %s",
            (collection, document_id),
        ).fetchone()[0]
        connection.execute(
            "UPDATE visual_elements SET x0 = 12.1234, y0 = 20.5678 "
            "WHERE collection_id = %s AND document_id = %s AND element_id = %s",
            (collection, document_id, element_id),
        )
    sources = CombinedRetrievalService(provider=Provider(),
        store=PostgresCombinedRetrievalStore(dsn)).search(
            collection_id=collection, query="pump", top_k=1).results
    assert len(sources) == 1 and sources[0].evidence_type == "visual_interpretation"
    assert (sources[0].x0, sources[0].y0) == (12.1234, 20.5678)
    resolver = PostgresAnswerSourceResolver(dsn)
    assert resolver.verify_sources(collection_id=collection, sources=sources)
    with psycopg.connect(dsn) as connection:
        connection.execute(
            "UPDATE visual_elements SET x0 = 12.1235 "
            "WHERE collection_id = %s AND document_id = %s AND element_id = %s",
            (collection, document_id, element_id),
        )
    assert not resolver.verify_sources(collection_id=collection, sources=sources)


def test_answer_rejects_source_deleted_during_generation(isolated_database_url):
    dsn = isolated_database_url
    MigrationRunner(dsn).upgrade()
    collection = uuid4()
    document_id = _index_text(dsn, collection)

    class DeletingGenerator:
        def generate(self, **kwargs):
            with psycopg.connect(dsn) as connection:
                connection.execute("DELETE FROM indexed_documents WHERE collection_id = %s "
                                   "AND document_id = %s", (collection, document_id))
            return GeneratedAnswer.model_validate({"status": "answered", "claims": [
                {"text": "Synthetic claim", "source_ids": ["T1"]}]})

    service = MultimodalRAGService(
        retriever=CombinedRetrievalService(provider=Provider(),
            store=PostgresCombinedRetrievalStore(dsn)), generator=DeletingGenerator(),
        resolver=PostgresAnswerSourceResolver(dsn), context_limits=ContextLimits(),
        generation_config=GenerationConfig())
    with pytest.raises(CitationError):
        service.answer(collection_id=collection, query="pump", top_k=1)


def test_visual_limit_applies_after_element_deduplication(isolated_database_url):
    dsn = isolated_database_url
    MigrationRunner(dsn).upgrade()
    collection = uuid4()
    document_id = _index_visual(dsn, collection, second_element=True)
    with psycopg.connect(dsn) as connection:
        first_element, original_id = connection.execute(
            "SELECT element_id, interpretation_id FROM visual_interpretations "
            "WHERE collection_id = %s AND document_id = %s",
            (collection, document_id),
        ).fetchone()
        second_element = connection.execute(
            "SELECT element_id FROM visual_elements WHERE collection_id = %s "
            "AND document_id = %s AND element_id <> %s ORDER BY element_id LIMIT 1",
            (collection, document_id, first_element),
        ).fetchone()[0]
        first_ids = [original_id]
        for element_id, vector, count in ((first_element, "[1,0]", 101),
                                          (second_element, "[0.9,0.1]", 1)):
            for _ in range(count):
                interpretation_id = uuid4()
                connection.execute(
                    "INSERT INTO visual_interpretations (collection_id, document_id, element_id, "
                    "interpretation_id, analysis_task, prompt_version, model_name, model_digest, "
                    "render_scale, rendered_image_sha256, description, observations, limitations, "
                    "evidence_kind, verification_status) "
                    "SELECT collection_id, document_id, %s, %s, analysis_task, prompt_version, "
                    "model_name, model_digest, render_scale, rendered_image_sha256, description, "
                    "observations, limitations, evidence_kind, verification_status "
                    "FROM visual_interpretations WHERE collection_id = %s AND document_id = %s "
                    "AND element_id = %s AND interpretation_id = %s",
                    (element_id, interpretation_id, collection, document_id, first_element, original_id),
                )
                connection.execute(
                    "INSERT INTO visual_interpretation_embeddings (collection_id, document_id, "
                    "element_id, interpretation_id, model_name, model_revision, dimensions, "
                    "text_format_version, text_sha256, embedding) "
                    "SELECT collection_id, document_id, %s, %s, model_name, model_revision, "
                    "dimensions, text_format_version, text_sha256, %s::vector "
                    "FROM visual_interpretation_embeddings WHERE collection_id = %s "
                    "AND document_id = %s AND element_id = %s AND interpretation_id = %s",
                    (element_id, interpretation_id, vector, collection, document_id,
                     first_element, original_id),
                )
                if element_id == first_element:
                    first_ids.append(interpretation_id)
    service = CombinedRetrievalService(provider=Provider(),
        store=PostgresCombinedRetrievalStore(dsn))
    first = service.search(collection_id=collection, query="pump", top_k=2).results
    second = service.search(collection_id=collection, query="pump", top_k=2).results
    assert [item.element_id for item in first] == [first_element, second_element]
    assert [item.interpretation_id for item in first] == [item.interpretation_id for item in second]
    assert first[0].interpretation_id == min(first_ids)
    assert [item.rank for item in first] == [1, 2]
    assert [item.rrf_score for item in first] == [1 / 61, 1 / 62]
    assert first[0].cosine_distance < first[1].cosine_distance


def test_same_snapshot_survives_concurrent_deletion(isolated_database_url, monkeypatch):
    dsn = isolated_database_url
    MigrationRunner(dsn).upgrade()
    collection = uuid4()
    text_document = _index_text(dsn, collection)
    visual_document = _index_visual(dsn, collection)
    import rag_ai.storage.combined_retrieval as module
    original = module.require_current_schema
    def delete_after_snapshot(connection):
        original(connection)
        with psycopg.connect(dsn) as writer:
            writer.execute("DELETE FROM indexed_documents WHERE collection_id = %s "
                           "AND document_id IN (%s, %s)",
                           (collection, text_document, visual_document))
    monkeypatch.setattr(module, "require_current_schema", delete_after_snapshot)
    result = CombinedRetrievalService(provider=Provider(),
        store=PostgresCombinedRetrievalStore(dsn)).search(collection_id=collection, query="pump")
    assert [item.evidence_type for item in result.results] == [
        "text_chunk", "visual_interpretation"]


def test_live_local_embeddinggemma_smoke(isolated_database_url):
    if os.getenv("RAG_AI_RUN_COMBINED_LIVE") != "1":
        pytest.skip("Set RAG_AI_RUN_COMBINED_LIVE=1 for local EmbeddingGemma smoke test.")
    dsn = isolated_database_url
    MigrationRunner(dsn).upgrade()
    collection = uuid4()
    provider = OllamaEmbeddingProvider(base_url="http://127.0.0.1:11434",
                                       model_name="embeddinggemma:300m", max_dimensions=4096)
    try:
        for text in ("A red pump moves coolant through the system.",
                     "A blue motor drives the conveyor belt."):
            content = text.encode()
            document = parse_document(filename="synthetic.txt", content_type="text/plain",
                content=content, max_size_bytes=100_000, max_pdf_pages=5,
                max_extracted_characters=100_000)
            PostgresIndexStore(dsn).save(collection_id=collection, document=document,
                original_content=content, chunked=chunk_document(document),
                first_batch=provider.embed([text]))
        result = CombinedRetrievalService(provider=provider,
            store=PostgresCombinedRetrievalStore(dsn)).search(
                collection_id=collection, query="Which pump moves coolant?", top_k=2)
        assert len(result.results) == 2
        assert all(item.evidence_type == "text_chunk" for item in result.results)
        print("Live ranking observation:", [item.text for item in result.results])
    finally:
        provider.close()
