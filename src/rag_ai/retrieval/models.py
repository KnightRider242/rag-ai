"""Typed query and retrieval results shared by the API and adapters."""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from rag_ai.documents.parser import FileType


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("Query must contain non-whitespace text.")
        return query


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    chunk_index: int
    filename: str
    file_type: FileType
    page_number: int | None
    text: str
    cosine_distance: float = Field(allow_inf_nan=False)


class SearchResponse(BaseModel):
    collection_id: UUID
    results: list[RetrievedChunk]
