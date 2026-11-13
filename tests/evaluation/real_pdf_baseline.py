"""Test-only baseline over the existing local development API."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from hashlib import sha256
import re
from time import perf_counter
from typing import Callable, Literal, Protocol
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator


Stage = Literal["parsing", "indexing", "extraction", "interpretation", "retrieval", "context", "generation", "citation_verification"]
Category = Literal["text", "visual", "mixed", "unanswerable"]


class VisualSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    element_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    task: Literal["describe_figure", "describe_diagram", "interpret_table"]


class DocumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    path: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    approved_for_local_evaluation: bool = False
    selected_visual_elements: list[VisualSelection] = Field(default_factory=list, max_length=8)


class SourceLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str
    page_number: int = Field(ge=1)
    chunk_index: int | None = Field(default=None, ge=0)
    element_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class CaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    category: Category
    query: str = Field(min_length=1, max_length=1000)
    expected_status: Literal["answered", "insufficient_evidence"]
    expected_answer: str | None = None
    labels_verified_from_source: bool = False
    relevant_text: list[SourceLabel] = Field(default_factory=list)
    relevant_visual: list[SourceLabel] = Field(default_factory=list)


class EvaluationManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_kind: Literal["synthetic", "real"]
    top_k: int = Field(default=3, ge=1, le=5)
    documents: list[DocumentSpec] = Field(min_length=1, max_length=3)
    cases: list[CaseSpec] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def check_references(self) -> "EvaluationManifest":
        aliases = [item.alias for item in self.documents]
        if len(set(aliases)) != len(aliases) or len({item.id for item in self.cases}) != len(self.cases):
            raise ValueError("Document aliases and case IDs must be unique.")
        if self.dataset_kind == "real" and not all(
            item.approved_for_local_evaluation and item.sha256 for item in self.documents
        ):
            raise ValueError("Every real PDF requires explicit local evaluation approval and a source hash.")
        for document in self.documents:
            selected = [item.element_id for item in document.selected_visual_elements]
            if len(selected) != len(set(selected)):
                raise ValueError("Selected visual element IDs must be unique per document.")
        for case in self.cases:
            for label in case.relevant_text + case.relevant_visual:
                if label.document not in aliases:
                    raise ValueError("A source label references an unknown document.")
            if any(label.element_id is not None or label.chunk_index is None for label in case.relevant_text):
                raise ValueError("Text labels require a chunk index and no element ID.")
            if any(label.chunk_index is not None for label in case.relevant_visual):
                raise ValueError("Visual labels cannot specify chunk indices.")
            if case.expected_status == "insufficient_evidence" and (case.relevant_text or case.relevant_visual):
                raise ValueError("An abstention cannot have relevant evidence labels.")
        return self


class EndpointResponse(Protocol):
    status_code: int
    def json(self) -> dict | list: ...


class EvaluationClient(Protocol):
    async def post(self, url: str, **kwargs: object) -> EndpointResponse: ...
    async def get(self, url: str, **kwargs: object) -> EndpointResponse: ...


class EvaluationStageError(RuntimeError):
    def __init__(self, stage: Stage, code: str) -> None:
        super().__init__(f"{stage}: {code}")
        self.stage, self.code = stage, code


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    category: Category
    ground_truth_verified: bool = False
    failure_stage: Stage | None = None
    failure_code: str | None = None
    answer_status: Literal["answered", "insufficient_evidence"] | None = None
    text_hit: bool | None = None
    text_relevant_found: int | None = None
    text_relevant_total: int | None = None
    text_recall: float | None = None
    text_reciprocal_rank: float | None = None
    visual_hit: bool | None = None
    combined_coverage: bool | None = None
    citation_identity_correct: int = 0
    citation_count: int = 0
    answer_status_correct: bool | None = None
    abstention_correct: bool | None = None
    human_answer_supported: bool | None = None
    human_visual_interpretation_accurate: bool | None = None
    text_retrieval_seconds: float | None = None
    visual_retrieval_seconds: float | None = None
    combined_retrieval_seconds: float | None = None
    answer_endpoint_seconds: float | None = None
    generation_seconds: float | None = None


class Metric(BaseModel):
    numerator: float
    denominator: int
    value: float | None


class HumanReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_supported: bool | None = None
    visual_interpretation_accurate: bool | None = None


class VisualCandidate(BaseModel):
    document_alias: str
    element_id: str
    page_number: int
    element_type: str
    coordinate_system: str
    x0: str
    y0: str
    x1: str
    y1: str


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_kind: Literal["synthetic", "real"]
    document_count: int
    case_count: int
    cases: list[CaseResult]
    metrics: dict[str, Metric]
    stage_failures: dict[str, int]
    selected_visual_interpretations: int
    visual_candidate_count: int
    visual_candidates: list[VisualCandidate]
    visual_interpretation_seconds: list[float]


def _metric(values: list[float]) -> Metric:
    return Metric(numerator=sum(values), denominator=len(values),
                  value=sum(values) / len(values) if values else None)


def summarize(manifest: EvaluationManifest, results: list[CaseResult], *,
              selected_visual_interpretations: int,
              visual_candidates: list[VisualCandidate],
              visual_interpretation_seconds: list[float]) -> EvaluationReport:
    """Aggregate only scored cases; retain raw numerators and denominators."""
    fields = ("text_hit", "text_recall", "text_reciprocal_rank", "visual_hit",
              "combined_coverage", "answer_status_correct", "abstention_correct", "human_answer_supported",
              "human_visual_interpretation_accurate", "text_retrieval_seconds",
              "visual_retrieval_seconds", "combined_retrieval_seconds",
              "answer_endpoint_seconds", "generation_seconds")
    metrics = {name: _metric([float(value) for result in results
                              if (value := getattr(result, name)) is not None]) for name in fields}
    metrics["citation_identity"] = Metric(
        numerator=sum(item.citation_identity_correct for item in results),
        denominator=sum(item.citation_count for item in results), value=None)
    citation = metrics["citation_identity"]
    citation.value = citation.numerator / citation.denominator if citation.denominator else None
    scored_text = [item for item in results if item.text_relevant_total is not None]
    found = sum(item.text_relevant_found or 0 for item in scored_text)
    total = sum(item.text_relevant_total or 0 for item in scored_text)
    metrics["text_recall_micro"] = Metric(numerator=found, denominator=total,
        value=found / total if total else None)
    metrics["visual_interpretation_seconds"] = _metric(visual_interpretation_seconds)
    failures: dict[str, int] = {}
    for result in results:
        if result.failure_stage:
            failures[result.failure_stage] = failures.get(result.failure_stage, 0) + 1
    return EvaluationReport(dataset_kind=manifest.dataset_kind,
        document_count=len(manifest.documents), case_count=len(manifest.cases), cases=results,
        metrics=metrics, stage_failures=failures,
        selected_visual_interpretations=selected_visual_interpretations,
        visual_candidate_count=len(visual_candidates), visual_candidates=visual_candidates,
        visual_interpretation_seconds=visual_interpretation_seconds)


def apply_human_reviews(report: EvaluationReport,
                        reviews: Mapping[str, HumanReview]) -> EvaluationReport:
    """Attach independent, post-run judgments without storing source or answer text."""
    unknown = set(reviews) - {case.case_id for case in report.cases}
    if unknown:
        raise ValueError("Human review references an unknown case.")
    updated = report.model_copy(deep=True)
    for case in updated.cases:
        review = reviews.get(case.case_id)
        if review is not None:
            case.human_answer_supported = review.answer_supported
            case.human_visual_interpretation_accurate = review.visual_interpretation_accurate
    for name in ("human_answer_supported", "human_visual_interpretation_accurate"):
        updated.metrics[name] = _metric([float(value) for case in updated.cases
            if (value := getattr(case, name)) is not None])
    return updated


def _response(response: EndpointResponse, stage: Stage) -> dict | list:
    if response.status_code != 200:
        code = "http_error"
        try:
            payload = response.json()
            if isinstance(payload, dict) and isinstance(payload.get("detail"), dict):
                candidate = payload["detail"].get("code")
                if isinstance(candidate, str) and candidate.isidentifier():
                    code = candidate
        except ValueError:
            pass
        if stage == "generation" and code == "evidence_too_large":
            stage = "context"
        raise EvaluationStageError(stage, code)
    return response.json()


def _seconds(start: float) -> float:
    return round(perf_counter() - start, 6)


def _text_source_keys(case: CaseSpec, chunks: Mapping[tuple[str, int, int], dict]) -> set[str]:
    return {chunks[(label.document, label.page_number, label.chunk_index)]["chunk_id"]
            for label in case.relevant_text if label.chunk_index is not None
            and (label.document, label.page_number, label.chunk_index) in chunks}


def _citation_correct(citation: dict, *, chunks_by_id: Mapping[str, dict],
                      elements_by_id: Mapping[str, dict],
                      interpretations_by_element: Mapping[str, dict]) -> bool:
    if citation.get("evidence_type") == "text_chunk":
        chunk = chunks_by_id.get(citation.get("chunk_id"))
        return bool(chunk and re.fullmatch(r"T[1-5]", str(citation.get("source_id", "")))
                    and citation.get("document_id") == chunk["document_id"]
                    and citation.get("page_number") == chunk["page_number"]
                    and citation.get("chunk_index") == chunk["chunk_index"]
                    and citation.get("filename") == chunk["filename"]
                    and citation.get("file_type") == chunk["file_type"])
    if citation.get("evidence_type") == "visual_interpretation":
        element = elements_by_id.get(citation.get("element_id"))
        interpretation = interpretations_by_element.get(citation.get("element_id"))
        if (not element or not interpretation
                or not re.fullmatch(r"V[1-5]", str(citation.get("source_id", "")))
                or citation.get("document_id") != element["document_id"]):
            return False
        coordinates = ("x0", "y0", "x1", "y1")
        return (citation.get("page_number") == element["page_number"]
            and citation.get("coordinate_system") == element["coordinate_system"]
            and citation.get("filename") == element["filename"]
            and all(Decimal(str(citation.get(key))) == Decimal(str(element[key])) for key in coordinates)
            and citation.get("evidence_kind") == "model_generated_interpretation"
            and citation.get("verification_status") == "unverified"
            and citation.get("element_type") == element["element_type"]
            and citation.get("interpretation_id") == interpretation["interpretation_id"]
            and citation.get("analysis_task") == interpretation["analysis_task"]
            and citation.get("prompt_version") == interpretation["prompt_version"]
            and citation.get("interpretation_model_name") == interpretation["model_name"]
            and citation.get("interpretation_model_digest") == interpretation["model_digest"])
    return False


async def run_baseline(client: EvaluationClient, manifest: EvaluationManifest, *,
                       pdf_bytes: Mapping[str, bytes], collection_id: UUID,
                       generation_times: list[float] | None = None,
                       answer_observer: Callable[[str, dict], None] | None = None) -> EvaluationReport:
    """Run existing API stages, returning only sanitized IDs, counts, scores, and timings."""
    document_ids: dict[str, str] = {}
    chunks: dict[tuple[str, int, int], dict] = {}
    chunks_by_id: dict[str, dict] = {}
    elements_by_id: dict[str, dict] = {}
    interpretations_by_element: dict[str, dict] = {}
    visual_candidates: list[VisualCandidate] = []
    visual_interpretation_seconds: list[float] = []
    selected_count = 0

    async def send(stage: Stage, method: Literal["post", "get"], url: str,
                   **kwargs: object) -> dict | list:
        try:
            response = await getattr(client, method)(url, **kwargs)
            return _response(response, stage)
        except (httpx.HTTPError, ValueError) as exc:
            raise EvaluationStageError(stage, "transport_or_response_invalid") from exc

    try:
        for document in manifest.documents:
            content = pdf_bytes.get(document.alias)
            if not isinstance(content, bytes):
                raise EvaluationStageError("parsing", "source_missing")
            if document.sha256 and sha256(content).hexdigest() != document.sha256:
                raise EvaluationStageError("parsing", "source_hash_mismatch")
            upload = {"file": (f"{document.alias}.pdf", content, "application/pdf")}
            parsed = await send("parsing", "post", "/documents/parse", files=upload)
            if (not isinstance(parsed, dict) or parsed.get("file_type") != "pdf"
                    or parsed.get("document_id") != sha256(content).hexdigest()):
                raise EvaluationStageError("parsing", "invalid_parse_response")
            chunked = await send("parsing", "post", "/documents/chunk", files=upload)
            indexed = await send("indexing", "post",
                f"/collections/{collection_id}/documents/index", files=upload)
            if not isinstance(chunked, dict) or not isinstance(indexed, dict) or indexed.get("document_id") != parsed.get("document_id"):
                raise EvaluationStageError("parsing", "source_identity_mismatch")
            document_ids[document.alias] = indexed["document_id"]
            for chunk in chunked["chunks"]:
                record = {"document_id": indexed["document_id"], "chunk_id": chunk["chunk_id"],
                          "chunk_index": chunk["chunk_index"], "page_number": chunk["page_number"],
                          "filename": f"{document.alias}.pdf", "file_type": "pdf"}
                chunks[(document.alias, chunk["page_number"], chunk["chunk_index"])] = record
                chunks_by_id[chunk["chunk_id"]] = record
            prefix = f"/collections/{collection_id}/documents/{indexed['document_id']}/visual-elements"
            await send("extraction", "post", f"{prefix}/index")
            for offset in range(0, 1200, 100):
                elements = await send("extraction", "get", prefix,
                    params={"limit": 100, "offset": offset})
                if not isinstance(elements, list):
                    raise EvaluationStageError("extraction", "invalid_element_list")
                for element in elements:
                    elements_by_id[element["element_id"]] = {
                        **element, "filename": f"{document.alias}.pdf"}
                    visual_candidates.append(VisualCandidate(document_alias=document.alias,
                        element_id=element["element_id"], page_number=element["page_number"],
                        element_type=element["element_type"],
                        coordinate_system=element["coordinate_system"],
                        **{key: str(element[key]) for key in ("x0", "y0", "x1", "y1")}))
                if len(elements) < 100:
                    break
            for selection in document.selected_visual_elements:
                element = elements_by_id.get(selection.element_id)
                if not element or element["document_id"] != indexed["document_id"]:
                    raise EvaluationStageError("extraction", "selected_element_missing")
                start = perf_counter()
                try:
                    interpretation = await send("interpretation", "post",
                        f"{prefix}/{selection.element_id}/interpretations",
                        json={"task": selection.task})
                finally:
                    visual_interpretation_seconds.append(_seconds(start))
                if not isinstance(interpretation, dict):
                    raise EvaluationStageError("interpretation", "invalid_interpretation")
                interpretations_by_element[selection.element_id] = interpretation
                await send("indexing", "post", f"/collections/{collection_id}/visual-interpretations/index",
                    json={"document_id": indexed["document_id"], "element_id": selection.element_id,
                          "interpretation_id": interpretation["interpretation_id"]})
                selected_count += 1
    except EvaluationStageError as exc:
        results = [CaseResult(case_id=case.id, category=case.category,
                              ground_truth_verified=case.labels_verified_from_source,
                              failure_stage=exc.stage, failure_code=exc.code) for case in manifest.cases]
        return summarize(manifest, results, selected_visual_interpretations=selected_count,
                         visual_candidates=visual_candidates,
                         visual_interpretation_seconds=visual_interpretation_seconds)

    results: list[CaseResult] = []
    for case in manifest.cases:
        result = CaseResult(case_id=case.id, category=case.category,
                            ground_truth_verified=case.labels_verified_from_source)
        request = {"query": case.query, "top_k": manifest.top_k}
        path = f"/collections/{collection_id}"
        expected_text = _text_source_keys(case, chunks)
        expected_visual = {label.element_id for label in case.relevant_visual
            if label.element_id is not None and label.element_id in elements_by_id
            and elements_by_id[label.element_id]["document_id"] == document_ids[label.document]
            and elements_by_id[label.element_id]["page_number"] == label.page_number}
        try:
            start = perf_counter()
            try:
                text_response = await send("retrieval", "post", f"{path}/search", json=request)
            finally:
                result.text_retrieval_seconds = _seconds(start)
            text_ids = [item["chunk_id"] for item in text_response["results"]]
            if case.labels_verified_from_source and len(expected_text) == len(case.relevant_text) and expected_text:
                hits = expected_text.intersection(text_ids)
                result.text_hit = bool(hits)
                result.text_relevant_found = len(hits)
                result.text_relevant_total = len(expected_text)
                result.text_recall = len(hits) / len(expected_text)
                result.text_reciprocal_rank = next((1 / rank for rank, item in enumerate(text_ids, 1)
                    if item in expected_text), 0.0)
            if selected_count:
                start = perf_counter()
                try:
                    visual_response = await send("retrieval", "post", f"{path}/visual-search", json=request)
                finally:
                    result.visual_retrieval_seconds = _seconds(start)
                visual_ids = {item["element_id"] for item in visual_response["results"]}
                if case.labels_verified_from_source and len(expected_visual) == len(case.relevant_visual) and expected_visual:
                    result.visual_hit = bool(expected_visual.intersection(visual_ids))
            start = perf_counter()
            try:
                combined = await send("retrieval", "post", f"{path}/multimodal-search", json=request)
            finally:
                result.combined_retrieval_seconds = _seconds(start)
            combined_text = {item["chunk_id"] for item in combined["results"] if item["evidence_type"] == "text_chunk"}
            combined_visual = {item["element_id"] for item in combined["results"] if item["evidence_type"] == "visual_interpretation"}
            if case.labels_verified_from_source and (expected_text or expected_visual):
                complete = len(expected_text) == len(case.relevant_text) and len(expected_visual) == len(case.relevant_visual)
                if complete:
                    result.combined_coverage = expected_text <= combined_text and expected_visual <= combined_visual
            before_generation = len(generation_times) if generation_times is not None else 0
            start = perf_counter()
            try:
                answer = await send("generation", "post", f"{path}/multimodal-answer", json=request)
            finally:
                result.answer_endpoint_seconds = _seconds(start)
            if generation_times is not None and len(generation_times) > before_generation:
                result.generation_seconds = generation_times[-1]
            if answer_observer is not None:
                answer_observer(case.id, answer)
            result.answer_status = answer["status"]
            if case.labels_verified_from_source:
                result.answer_status_correct = answer["status"] == case.expected_status
                if case.expected_status == "insufficient_evidence":
                    result.abstention_correct = answer["status"] == "insufficient_evidence"
            for claim in answer["claims"]:
                for citation in claim["citations"]:
                    result.citation_count += 1
                    result.citation_identity_correct += int(_citation_correct(citation,
                        chunks_by_id=chunks_by_id, elements_by_id=elements_by_id,
                        interpretations_by_element=interpretations_by_element))
            if result.citation_identity_correct != result.citation_count:
                result.failure_stage = "citation_verification"
                result.failure_code = "citation_identity_mismatch"
        except EvaluationStageError as exc:
            result.failure_stage, result.failure_code = exc.stage, exc.code
        results.append(result)
    return summarize(manifest, results, selected_visual_interpretations=selected_count,
                     visual_candidates=visual_candidates,
                     visual_interpretation_seconds=visual_interpretation_seconds)
