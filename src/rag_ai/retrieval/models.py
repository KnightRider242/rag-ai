"""Typed query and retrieval results shared by the API and adapters."""

from uuid import UUID
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class CombinedTextResult(RetrievedChunk):
    model_config = ConfigDict(extra="forbid")
    evidence_type: Literal["text_chunk"] = "text_chunk"
    rank: int = Field(ge=1)
    rrf_score: float = Field(gt=0, allow_inf_nan=False)


class CombinedVisualResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_type: Literal["visual_interpretation"] = "visual_interpretation"
    rank: int = Field(ge=1)
    rrf_score: float = Field(gt=0, allow_inf_nan=False)
    document_id: str
    element_id: str
    interpretation_id: UUID
    filename: str
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
    verification_status: Literal["unverified"]
    cosine_distance: float = Field(allow_inf_nan=False)


CombinedResult = Annotated[CombinedTextResult | CombinedVisualResult, Field(discriminator="evidence_type")]


class CombinedSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: UUID
    embedding_model_name: str
    embedding_model_revision: str
    results: list[CombinedResult]
