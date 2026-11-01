"""Local semantic retrieval over unverified visual interpretations."""

from hashlib import sha256
import math
import re
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, Field

from rag_ai.embeddings.base import EmbeddingBatch, EmbeddingProvider
from rag_ai.retrieval.models import SearchRequest
from rag_ai.storage.interpretations import StoredInterpretation

TEXT_FORMAT_VERSION = "visual_interpretation_v1"
MAX_VISUAL_EMBED_TEXT_BYTES = 8192


class VisualRetrievalError(ValueError):
    """An interpretation or embedding cannot safely be indexed or searched."""


class VisualIndexResult(BaseModel):
    collection_id: UUID
    document_id: str
    element_id: str
    interpretation_id: UUID
    embedding_model_name: str
    embedding_model_revision: str
    dimensions: int
    text_format_version: str
    text_sha256: str
    already_indexed: bool


class RetrievedVisualInterpretation(BaseModel):
    document_id: str
    filename: str
    element_id: str
    interpretation_id: UUID
    page_number: int
    coordinate_system: str
    x0: float
    y0: float
    x1: float
    y1: float
    element_type: str
    analysis_task: str
    prompt_version: str
    interpretation_model_name: str
    interpretation_model_digest: str
    description: str
    observations: list[str]
    limitations: list[str]
    evidence_kind: str
    verification_status: str
    cosine_distance: float = Field(allow_inf_nan=False)


class VisualSearchResponse(BaseModel):
    collection_id: UUID
    embedding_model_name: str
    embedding_model_revision: str
    results: list[RetrievedVisualInterpretation]


class VisualRetrievalStore(Protocol):
    def load_interpretation(self, *, collection_id: UUID, document_id: str,
                            element_id: str, interpretation_id: UUID) -> StoredInterpretation: ...

    def index_interpretation(self, *, interpretation: StoredInterpretation, text_sha256: str,
                             text_format_version: str, batch: EmbeddingBatch) -> bool: ...

    def search(self, *, collection_id: UUID, query_vector: list[float], model_name: str,
               model_revision: str, dimensions: int, top_k: int,
               text_format_version: str) -> list[RetrievedVisualInterpretation]: ...


def interpretation_embedding_text(interpretation: StoredInterpretation) -> str:
    """Textualize only the model description and observations, with a trust label."""
    description = interpretation.description.strip()
    observations = [item.strip() for item in interpretation.observations]
    if not description or any(not item for item in observations):
        raise VisualRetrievalError("Stored visual interpretation is invalid.")
    lines = ["Unverified model-generated visual interpretation", f"Description: {description}"]
    lines.extend(f"Observation: {item}" for item in observations)
    result = "\n".join(lines)
    if len(result.encode("utf-8")) > MAX_VISUAL_EMBED_TEXT_BYTES:
        raise VisualRetrievalError("Stored visual interpretation exceeds the embedding limit.")
    return result


def _vector(batch: EmbeddingBatch) -> list[float]:
    if (len(batch.vectors) != 1 or not re.fullmatch(r"[0-9a-f]{64}", batch.model_revision)
            or len(batch.vectors[0]) != batch.dimensions
            or any(not math.isfinite(value) for value in batch.vectors[0])):
        raise VisualRetrievalError("Embedding service returned an invalid vector or model identity.")
    vector = batch.vectors[0]
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise VisualRetrievalError("Embedding service returned an unusable vector.")
    return vector


class VisualRetrievalService:
    def __init__(self, *, provider: EmbeddingProvider, store: VisualRetrievalStore) -> None:
        self.provider = provider
        self.store = store

    def index(self, *, collection_id: UUID, document_id: str, element_id: str,
              interpretation_id: UUID) -> VisualIndexResult:
        interpretation = self.store.load_interpretation(
            collection_id=collection_id, document_id=document_id, element_id=element_id,
            interpretation_id=interpretation_id,
        )
        text = interpretation_embedding_text(interpretation)
        batch = self.provider.embed([text])
        _vector(batch)
        text_sha256 = sha256(text.encode("utf-8")).hexdigest()
        already_indexed = self.store.index_interpretation(
            interpretation=interpretation, text_sha256=text_sha256,
            text_format_version=TEXT_FORMAT_VERSION, batch=batch,
        )
        return VisualIndexResult(
            collection_id=collection_id, document_id=document_id, element_id=element_id,
            interpretation_id=interpretation_id, embedding_model_name=batch.model_name,
            embedding_model_revision=batch.model_revision, dimensions=batch.dimensions,
            text_format_version=TEXT_FORMAT_VERSION, text_sha256=text_sha256,
            already_indexed=already_indexed,
        )

    def search(self, *, collection_id: UUID, query: str, top_k: int = 5) -> VisualSearchResponse:
        request = SearchRequest(query=query, top_k=top_k)
        batch = self.provider.embed([request.query])
        vector = _vector(batch)
        results = self.store.search(
            collection_id=collection_id, query_vector=vector, model_name=batch.model_name,
            model_revision=batch.model_revision, dimensions=batch.dimensions,
            top_k=request.top_k, text_format_version=TEXT_FORMAT_VERSION,
        )
        return VisualSearchResponse(collection_id=collection_id,
                                    embedding_model_name=batch.model_name,
                                    embedding_model_revision=batch.model_revision,
                                    results=results)
