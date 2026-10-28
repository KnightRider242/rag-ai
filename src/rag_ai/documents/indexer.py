"""Parse/chunk output to collection-scoped local vector indexing."""

import math
from typing import Iterable, Protocol
from uuid import UUID

from pydantic import BaseModel

from rag_ai.documents.chunker import ChunkedDocument
from rag_ai.documents.parser import FileType, ParsedDocument
from rag_ai.embeddings.base import EMBEDDING_BATCH_SIZE, EmbeddingBatch, EmbeddingProvider


class IndexStore(Protocol):
    def save(
        self, *, collection_id: UUID, document: ParsedDocument, original_content: bytes,
        chunked: ChunkedDocument, first_batch: EmbeddingBatch,
        remaining_batches: Iterable[EmbeddingBatch],
    ) -> None: ...


class IndexedDocument(BaseModel):
    collection_id: UUID
    document_id: str
    filename: str
    file_type: FileType
    total_chunks: int
    embedding_model: str
    embedding_model_revision: str
    embedding_dimensions: int


class IndexingError(ValueError):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def index_document(
    *, collection_id: UUID, document: ParsedDocument, original_content: bytes,
    chunked: ChunkedDocument, provider: EmbeddingProvider, store: IndexStore,
    max_index_chunks: int, max_embedding_elements: int = 1_000_000,
) -> IndexedDocument:
    """Embed bounded batches and stage them before one atomic final write."""
    if not chunked.chunks or len(chunked.chunks) > max_index_chunks:
        raise IndexingError("index_chunk_limit", "Document exceeds the indexing chunk limit.", 413)

    first_count = min(EMBEDDING_BATCH_SIZE, len(chunked.chunks))
    first_batch = provider.embed([chunk.text for chunk in chunked.chunks[:first_count]])
    model_name = first_batch.model_name
    model_revision = first_batch.model_revision
    dimensions = first_batch.dimensions

    def validate_batch(batch: EmbeddingBatch, expected_count: int) -> None:
        if len(batch.vectors) != expected_count:
            raise IndexingError("invalid_embeddings", "Embedding service returned an invalid vector count.", 502)
        if (batch.model_name != model_name or batch.model_revision != model_revision or batch.dimensions != dimensions
                or any(len(vector) != dimensions or any(not math.isfinite(value) for value in vector)
                       for vector in batch.vectors)):
            raise IndexingError("invalid_embeddings", "Embedding service returned inconsistent vectors.", 502)

    validate_batch(first_batch, first_count)
    if len(chunked.chunks) * dimensions > max_embedding_elements:
        raise IndexingError(
            "embedding_element_limit", "Document exceeds the embedding element limit.", 413,
        )

    def remaining_batches() -> Iterable[EmbeddingBatch]:
        for start in range(first_count, len(chunked.chunks), EMBEDDING_BATCH_SIZE):
            texts = [chunk.text for chunk in chunked.chunks[start:start + EMBEDDING_BATCH_SIZE]]
            batch = provider.embed(texts)
            validate_batch(batch, len(texts))
            yield batch

    store.save(
        collection_id=collection_id, document=document, original_content=original_content,
        chunked=chunked, first_batch=first_batch, remaining_batches=remaining_batches(),
    )
    return IndexedDocument(
        collection_id=collection_id, document_id=document.document_id,
        filename=document.filename, file_type=document.file_type, total_chunks=len(chunked.chunks),
        embedding_model=model_name, embedding_model_revision=model_revision,
        embedding_dimensions=dimensions,
    )
