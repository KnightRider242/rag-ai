"""One-query, two-source local retrieval with deterministic rank fusion."""

from typing import Protocol
from uuid import UUID

from rag_ai.embeddings.base import EmbeddingProvider
from rag_ai.retrieval.models import (
    CombinedSearchResponse, CombinedTextResult, CombinedVisualResult, RetrievedChunk,
    SearchRequest,
)
from rag_ai.retrieval.visual import RetrievedVisualInterpretation, _vector


class CombinedSearchStore(Protocol):
    def search(self, *, collection_id: UUID, query_vector: list[float], model_name: str,
               model_revision: str, dimensions: int, text_limit: int,
               visual_limit: int) -> tuple[list[RetrievedChunk], list[RetrievedVisualInterpretation]]: ...


class CombinedRetrievalService:
    def __init__(self, *, provider: EmbeddingProvider, store: CombinedSearchStore) -> None:
        self.provider = provider
        self.store = store

    def search(self, *, collection_id: UUID, query: str, top_k: int = 5) -> CombinedSearchResponse:
        request = SearchRequest(query=query, top_k=top_k)
        batch = self.provider.embed([request.query])
        vector = _vector(batch)
        chunks, visuals = self.store.search(
            collection_id=collection_id, query_vector=vector, model_name=batch.model_name,
            model_revision=batch.model_revision, dimensions=batch.dimensions,
            text_limit=20, visual_limit=100,
        )
        candidates: list[tuple[float, int, str, RetrievedChunk | RetrievedVisualInterpretation]] = []
        for rank, chunk in enumerate(chunks, 1):
            candidates.append((1 / (60 + rank), 0,
                               f"{chunk.document_id}:{chunk.chunk_index:010d}:{chunk.chunk_id}", chunk))
        seen_elements: set[tuple[str, str]] = set()
        for rank, visual in enumerate(visuals, 1):
            identity = (visual.document_id, visual.element_id)
            if identity in seen_elements:
                continue
            seen_elements.add(identity)
            candidates.append((1 / (60 + rank), 1,
                               f"{visual.document_id}:{visual.element_id}:{visual.interpretation_id}", visual))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        results: list[CombinedTextResult | CombinedVisualResult] = []
        for rank, (score, modality, _, source) in enumerate(candidates[:request.top_k], 1):
            if modality == 0:
                assert isinstance(source, RetrievedChunk)
                results.append(CombinedTextResult(**source.model_dump(), rank=rank, rrf_score=score))
            else:
                assert isinstance(source, RetrievedVisualInterpretation)
                results.append(CombinedVisualResult(**source.model_dump(), rank=rank, rrf_score=score))
        return CombinedSearchResponse(
            collection_id=collection_id, embedding_model_name=batch.model_name,
            embedding_model_revision=batch.model_revision, results=results,
        )
