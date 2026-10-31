"""Typed public answer contract and request validation."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_ai.documents.parser import FileType


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("Question must contain non-whitespace text.")
        return query


class Citation(BaseModel):
    source_id: str
    document_id: str
    chunk_id: str
    filename: str
    file_type: FileType
    page_number: int | None


class AnswerClaim(BaseModel):
    text: str
    citations: list[Citation] = Field(min_length=1)


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_id: UUID
    status: Literal["answered", "insufficient_evidence"]
    claims: list[AnswerClaim]

    @model_validator(mode="after")
    def validate_status(self) -> "AnswerResponse":
        if self.status == "answered" and not self.claims:
            raise ValueError("An answered response requires claims.")
        if self.status == "insufficient_evidence" and self.claims:
            raise ValueError("An abstention cannot contain claims.")
        return self
