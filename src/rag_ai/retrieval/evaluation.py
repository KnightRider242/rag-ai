"""Small, reproducible ranking metrics over actual retrieval results."""

from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from rag_ai.retrieval.models import SearchRequest
from rag_ai.retrieval.service import RetrievalService


class EvaluationCase(BaseModel):
    query: str
    expected_chunk_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case(self) -> "EvaluationCase":
        SearchRequest(query=self.query)
        if len(set(self.expected_chunk_ids)) != len(self.expected_chunk_ids):
            raise ValueError("Expected chunk IDs must be unique.")
        return self


class EvaluationResult(BaseModel):
    query: str
    expected_chunk_ids: list[str]
    actual_chunk_ids: list[str]
    recall_at_k: float
    hit_at_k: bool
    reciprocal_rank: float


class EvaluationReport(BaseModel):
    top_k: int
    cases: list[EvaluationResult]
    mean_recall_at_k: float
    mean_hit_at_k: float
    mean_reciprocal_rank: float


def evaluate_retrieval(
    service: RetrievalService, *, collection_id: UUID,
    cases: Sequence[EvaluationCase], top_k: int = 5,
) -> EvaluationReport:
    """Compute recall, hit rate, and MRR from the service's ranked outputs."""
    SearchRequest(query="evaluation", top_k=top_k)
    if not cases:
        raise ValueError("At least one evaluation case is required.")
    results: list[EvaluationResult] = []
    for case in cases:
        actual = [chunk.chunk_id for chunk in service.retrieve(
            collection_id=collection_id, query=case.query, top_k=top_k,
        ).results]
        expected = set(case.expected_chunk_ids)
        hits = expected.intersection(actual)
        first_rank = next((index for index, chunk_id in enumerate(actual, 1) if chunk_id in expected), None)
        results.append(EvaluationResult(
            query=case.query, expected_chunk_ids=case.expected_chunk_ids,
            actual_chunk_ids=actual, recall_at_k=len(hits) / len(expected),
            hit_at_k=bool(hits), reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        ))
    count = len(results)
    return EvaluationReport(
        top_k=top_k, cases=results,
        mean_recall_at_k=sum(item.recall_at_k for item in results) / count,
        mean_hit_at_k=sum(item.hit_at_k for item in results) / count,
        mean_reciprocal_rank=sum(item.reciprocal_rank for item in results) / count,
    )
