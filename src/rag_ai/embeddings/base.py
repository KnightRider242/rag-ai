"""Provider-independent embedding contract."""

from typing import Protocol

from pydantic import BaseModel, Field

EMBEDDING_BATCH_SIZE = 16


class EmbeddingBatch(BaseModel):
    model_name: str = Field(min_length=1, max_length=200)
    model_revision: str = Field(min_length=1, max_length=200)
    dimensions: int = Field(ge=1, le=16_000)
    vectors: list[list[float]]


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> EmbeddingBatch: ...


class EmbeddingError(ValueError):
    """A local embedding request failed or returned invalid data."""
