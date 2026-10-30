"""Retrieval metric tests and opt-in local semantic evaluation."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from rag_ai.documents.chunker import chunk_document
from rag_ai.documents.indexer import index_document
from rag_ai.documents.parser import parse_document
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.retrieval.evaluation import EvaluationCase, evaluate_retrieval
from rag_ai.retrieval.models import RetrievedChunk, SearchResponse
from rag_ai.retrieval.service import RetrievalService
from rag_ai.storage.migrate import MigrationRunner
from rag_ai.storage.postgres import PostgresIndexStore


class RankedService:
    def __init__(self) -> None:
        self.ranks = {
            "first": ["chunk-b", "chunk-a"], "second": ["chunk-c"],
            "third": ["a", "b", "c", "target"],
        }

    def retrieve(self, *, collection_id, query: str, top_k: int) -> SearchResponse:
        return SearchResponse(collection_id=collection_id, results=[RetrievedChunk(
            document_id="synthetic-doc", chunk_id=chunk_id, chunk_index=index,
            filename="synthetic.txt", file_type="txt", page_number=None,
            text="synthetic", cosine_distance=float(index),
        ) for index, chunk_id in enumerate(self.ranks[query][:top_k])])


def test_recall_hit_and_mrr_use_actual_ranked_ids() -> None:
    report = evaluate_retrieval(
        RankedService(), collection_id=uuid4(), top_k=2, cases=[
            EvaluationCase(query="first", expected_chunk_ids=["chunk-a", "chunk-b"]),
            EvaluationCase(query="second", expected_chunk_ids=["missing"]),
        ],
    )
    assert report.cases[0].actual_chunk_ids == ["chunk-b", "chunk-a"]
    assert report.cases[0].recall_at_k == 1
    assert report.cases[0].hit_at_k is True
    assert report.cases[0].reciprocal_rank == 1
    assert report.cases[1].recall_at_k == 0
    assert report.cases[1].hit_at_k is False
    assert report.cases[1].reciprocal_rank == 0
    assert report.mean_recall_at_k == 0.5
    assert report.mean_hit_at_k == 0.5
    assert report.mean_reciprocal_rank == 0.5


def test_evaluation_rejects_invalid_cases() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(query="  ", expected_chunk_ids=["chunk"])
    with pytest.raises(ValidationError):
        EvaluationCase(query="valid", expected_chunk_ids=[])
    with pytest.raises(ValidationError):
        EvaluationCase(query="valid", expected_chunk_ids=["a", "a"])
    with pytest.raises(ValueError):
        evaluate_retrieval(RankedService(), collection_id=uuid4(), cases=[])


def test_mrr_is_truncated_at_requested_k() -> None:
    case = [EvaluationCase(query="third", expected_chunk_ids=["target"])]
    assert evaluate_retrieval(RankedService(), collection_id=uuid4(), cases=case, top_k=3).mean_reciprocal_rank == 0
    assert evaluate_retrieval(RankedService(), collection_id=uuid4(), cases=case, top_k=4).mean_reciprocal_rank == 0.25


def _semantic_dataset() -> dict:
    return json.loads((Path(__file__).parent / "evaluation" / "semantic_cases.json").read_text(encoding="utf-8"))


def test_semantic_dataset_has_independent_one_chunk_evidence() -> None:
    dataset = _semantic_dataset()
    documents, questions = dataset["documents"], dataset["questions"]
    source_ids = [item["id"] for item in documents]
    assert 6 <= len(documents) <= 9
    assert 3 <= len(questions) <= 5
    assert len(source_ids) == len(set(source_ids))
    assert len({item["query"] for item in questions}) == len(questions)
    for question in questions:
        assert len(question["relevant_document_ids"]) == 1
        assert set(question["relevant_document_ids"]) <= set(source_ids)
    for item in documents:
        content = item["text"].encode("utf-8")
        parsed = parse_document(
            filename=f"{item['id']}.txt", content_type="text/plain", content=content,
            max_size_bytes=100_000, max_pdf_pages=10, max_extracted_characters=100_000,
        )
        assert len(chunk_document(parsed).chunks) == 1


def test_local_semantic_evaluation(isolated_database_url: str) -> None:
    if os.getenv("RAG_AI_RUN_OLLAMA_TESTS") != "1":
        pytest.skip("Set RAG_AI_RUN_OLLAMA_TESTS=1 for local Ollama evaluation.")
    dataset = _semantic_dataset()
    documents, questions = dataset["documents"], dataset["questions"]
    MigrationRunner(isolated_database_url).upgrade()
    store = PostgresIndexStore(isolated_database_url)
    collection_id = uuid4()
    provider = OllamaEmbeddingProvider(
        base_url=os.getenv("RAG_AI_TEST_OLLAMA_URL", "http://127.0.0.1:11434"),
        model_name="embeddinggemma:300m", max_dimensions=4096,
    )
    try:
        chunk_ids: dict[str, str] = {}
        source_by_chunk: dict[str, str] = {}
        for item in documents:
            content = item["text"].encode("utf-8")
            parsed = parse_document(
                filename=f"{item['id']}.txt", content_type="text/plain", content=content,
                max_size_bytes=100_000, max_pdf_pages=10, max_extracted_characters=100_000,
            )
            chunks = chunk_document(parsed)
            assert len(chunks.chunks) == 1
            index_document(
                collection_id=collection_id, document=parsed, original_content=content,
                chunked=chunks, provider=provider, store=store, max_index_chunks=500,
            )
            chunk_ids[item["id"]] = chunks.chunks[0].chunk_id
            source_by_chunk[chunks.chunks[0].chunk_id] = item["id"]
        assert len(chunk_ids) > 3
        evaluation_cases = [EvaluationCase(
            query=question["query"],
            expected_chunk_ids=[chunk_ids[source] for source in question["relevant_document_ids"]],
        ) for question in questions]
        service = RetrievalService(provider=provider, store=store)
        at_one = evaluate_retrieval(
            service, collection_id=collection_id, cases=evaluation_cases, top_k=1,
        )
        at_three = evaluate_retrieval(
            service, collection_id=collection_id, cases=evaluation_cases, top_k=3,
        )
        rankings = [{
            "query": question["query"],
            "expected": question["relevant_document_ids"],
            "actual_top_3": [source_by_chunk[chunk_id] for chunk_id in result.actual_chunk_ids],
        } for question, result in zip(questions, at_three.cases, strict=True)]
        print("Local semantic evaluation:", json.dumps({
            "documents": len(documents), "questions": len(questions),
            "recall_at_1": at_one.mean_recall_at_k,
            "recall_at_3": at_three.mean_recall_at_k,
            "hit_at_1": at_one.mean_hit_at_k,
            "hit_at_3": at_three.mean_hit_at_k,
            "mrr_at_3": at_three.mean_reciprocal_rank,
            "rankings": rankings,
        }))
        for question, result in zip(questions, at_one.cases, strict=True):
            if not result.hit_at_k:
                ranked = service.retrieve(
                    collection_id=collection_id, query=question["query"], top_k=3,
                ).results
                print("Top-1 miss diagnostic:", json.dumps({
                    "query": question["query"],
                    "ranked": [{"source": source_by_chunk[chunk.chunk_id],
                                "distance": chunk.cosine_distance} for chunk in ranked],
                }))
        # Four unambiguous, single-evidence questions should mostly rank first,
        # and every labeled source should appear within three among nine candidates.
        assert at_one.mean_recall_at_k >= 0.75
        assert at_one.mean_hit_at_k >= 0.75
        assert at_three.mean_recall_at_k == 1.0
        assert at_three.mean_hit_at_k == 1.0
        assert at_three.mean_reciprocal_rank >= 0.8
    finally:
        provider.close()
