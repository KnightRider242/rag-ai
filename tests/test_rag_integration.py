"""Opt-in synthetic parse-to-cited-answer local integration."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from rag_ai.documents.chunker import chunk_document
from rag_ai.documents.indexer import index_document
from rag_ai.documents.parser import parse_document
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.generation.base import GenerationConfig
from rag_ai.generation.ollama import OllamaGenerationProvider
from rag_ai.rag.context import ContextLimits
from rag_ai.rag.service import RAGService
from rag_ai.retrieval.service import RetrievalService
from rag_ai.storage.migrate import MigrationRunner
from rag_ai.storage.postgres import PostgresIndexStore


def test_local_rag_grounded_answers_and_collection_isolation(isolated_database_url: str) -> None:
    if os.getenv("RAG_AI_RUN_OLLAMA_TESTS") != "1":
        pytest.skip("Set RAG_AI_RUN_OLLAMA_TESTS=1 for local Ollama generation integration.")
    dataset = json.loads((Path(__file__).parent / "evaluation" / "rag_cases.json").read_text(encoding="utf-8"))
    MigrationRunner(isolated_database_url).upgrade()
    store = PostgresIndexStore(isolated_database_url)
    collection_id, other_collection = uuid4(), uuid4()
    embedding = OllamaEmbeddingProvider(
        base_url=os.getenv("RAG_AI_TEST_OLLAMA_URL", "http://127.0.0.1:11434"),
        model_name="embeddinggemma:300m", max_dimensions=4096,
    )
    generation = OllamaGenerationProvider(
        base_url=os.getenv("RAG_AI_TEST_OLLAMA_URL", "http://127.0.0.1:11434"),
        model_name="qwen3:4b-instruct",
    )
    try:
        expected: dict[str, tuple[str, str]] = {}
        for item in dataset["documents"]:
            content = item["text"].encode("utf-8")
            parsed = parse_document(
                filename=f"{item['id']}.txt", content_type="text/plain", content=content,
                max_size_bytes=100_000, max_pdf_pages=10, max_extracted_characters=100_000,
            )
            chunks = chunk_document(parsed)
            assert len(chunks.chunks) == 1
            target_collection = other_collection if item["id"] == "warranty_other_collection" else collection_id
            index_document(
                collection_id=target_collection, document=parsed, original_content=content,
                chunked=chunks, provider=embedding, store=store, max_index_chunks=500,
            )
            expected[item["id"]] = (parsed.document_id, chunks.chunks[0].chunk_id)
        service = RAGService(
            retriever=RetrievalService(provider=embedding, store=store),
            generator=generation, context_limits=ContextLimits(),
            generation_config=GenerationConfig(),
        )
        observed = []
        for case in dataset["questions"]:
            print("Evaluating synthetic question:", case["query"])
            answer = service.answer(collection_id=collection_id, query=case["query"], top_k=2)
            citations = [citation for claim in answer.claims for citation in claim.citations]
            observed.append({
                "query": case["query"], "status": answer.status,
                "sources": [citation.filename for citation in citations],
            })
            assert answer.status == case["expected_status"]
            if case["expected_source"] is None:
                assert answer.claims == []
            else:
                assert answer.claims and citations
                assert all(citation.document_id == expected[case["expected_source"]][0]
                           and citation.chunk_id == expected[case["expected_source"]][1]
                           and citation.page_number is None for citation in citations)
                claim_text = " ".join(claim.text.lower() for claim in answer.claims)
                assert all(term in claim_text for term in case["required_terms"])
            assert all(citation.chunk_id != expected["warranty_other_collection"][1]
                       for citation in citations)
        print("Local RAG evaluation:", json.dumps(observed))
    finally:
        generation.close()
        embedding.close()
