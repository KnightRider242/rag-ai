"""Provider-independent query embedding and retrieval coordination."""

import math
from typing import Protocol
from uuid import UUID

from rag_ai.embeddings.base import EmbeddingProvider
from rag_ai.retrieval.models import RetrievedChunk, SearchRequest, SearchResponse


class SearchStore(Protocol):
    def search(
        self, *, collection_id: UUID, query_vector: list[float], model_name: str,
        model_revision: str, dimensions: int, top_k: int,
    ) -> list[RetrievedChunk]: ...


class RetrievalError(ValueError):
    """The embedding provider returned an unusable query vector."""


class RetrievalService:
    def __init__(self, *, provider: EmbeddingProvider, store: SearchStore) -> None:
        self.provider = provider
        self.store = store

    def retrieve(self, *, collection_id: UUID, query: str, top_k: int = 5) -> SearchResponse:
        """Embed exactly one query, validate it, and request ranked chunks."""
        request = SearchRequest(query=query, top_k=top_k)
        batch = self.provider.embed([request.query])
        if len(batch.vectors) != 1:
            raise RetrievalError("Embedding service returned an invalid query vector.")
        vector = batch.vectors[0]
        if len(vector) != batch.dimensions or any(not math.isfinite(value) for value in vector):
            raise RetrievalError("Embedding service returned an invalid query vector.")
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0:
            raise RetrievalError("Embedding service returned an unusable query vector.")
        results = self.store.search(
            collection_id=collection_id, query_vector=vector,
            model_name=batch.model_name, model_revision=batch.model_revision,
            dimensions=batch.dimensions, top_k=request.top_k,
        )
        return SearchResponse(collection_id=collection_id, results=results)
