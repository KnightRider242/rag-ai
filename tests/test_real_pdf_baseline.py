"""Synthetic evaluator checks and an explicitly opted-in real-PDF baseline."""

import asyncio
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

import httpx
import pymupdf
import pytest
from pydantic import ValidationError

from evaluation.real_pdf_baseline import (
    EvaluationManifest, EvaluationReport, HumanReview, apply_human_reviews, run_baseline,
)
from rag_ai.core.config import Settings
from rag_ai.documents.parser import parse_document
from rag_ai.documents.visual_elements import extract_visual_elements
from rag_ai.generation.ollama import OllamaGenerationProvider
from rag_ai.main import create_app
from rag_ai.storage.migrate import MigrationRunner
from test_local_rag_e2e import _validated_test_conninfo


EXAMPLE = Path(__file__).parent / "evaluation" / "real_pdf_manifest.example.json"


def _synthetic_pdf(alias: str) -> bytes:
    """Author a small PDF in memory; no document bytes enter version control."""
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=400)
        if alias == "pump_manual":
            page.insert_text((60, 80), "The blue coolant pump moves liquid through the radiator.")
            page.insert_text((60, 110), "Valve V7 opens at 80 C.")
        else:
            page.insert_text((60, 80), "Safety relay K2 disconnects motor power after emergency stop.")
            page.insert_text((60, 110), "Alarm reset requires manual acknowledgement.")
            page.insert_text((60, 140), "Pressure alerts are read in the alarm table.")
        page = pdf.new_page(width=600, height=400)
        if alias == "pump_manual":
            labels = ("VALVE V7", "PUMP P1", "RADIATOR R1")
        else:
            labels = ("STOP BUTTON", "RELAY K2", "MOTOR")
        for index, label in enumerate(labels):
            x = 40 + index * 180
            page.draw_rect(pymupdf.Rect(x, 100, x + 140, 160))
            page.insert_text((x + 8, 132), label)
            if index < 2:
                page.draw_line((x + 140, 130), (x + 180, 130))
        page = pdf.new_page(width=600, height=400)
        rows = (("A1", "high pressure"), ("A2", "low coolant")) if alias == "alarm_manual" else (
            ("V7", "pump inlet"), ("R1", "radiator outlet"))
        for row_index, (code, meaning) in enumerate(rows):
            y = 80 + row_index * 50
            page.draw_rect(pymupdf.Rect(50, y, 400, y + 40))
            page.draw_line((130, y), (130, y + 40))
            page.insert_text((60, y + 25), code)
            page.insert_text((145, y + 25), meaning)
        return pdf.tobytes()


def _example_manifest() -> EvaluationManifest:
    return EvaluationManifest.model_validate_json(EXAMPLE.read_text(encoding="utf-8"))


def test_example_manifest_has_sixteen_independent_synthetic_questions() -> None:
    manifest = _example_manifest()
    assert manifest.dataset_kind == "synthetic"
    assert Counter(case.category for case in manifest.cases) == {
        "text": 4, "visual": 4, "mixed": 4, "unanswerable": 4}
    assert len({case.id for case in manifest.cases}) == 16
    assert all(not case.labels_verified_from_source for case in manifest.cases)


def test_sanitized_result_schema_matches_runtime_model() -> None:
    schema = Path(__file__).parent / "evaluation" / "real_pdf_results.schema.json"
    assert json.loads(schema.read_text(encoding="utf-8")) == EvaluationReport.model_json_schema()


def test_authored_pdf_fixture_has_text_figures_and_tables() -> None:
    for alias, required in (("pump_manual", "Valve V7 opens at 80 C"),
                            ("alarm_manual", "Safety relay K2")):
        content = _synthetic_pdf(alias)
        parsed = parse_document(filename=f"{alias}.pdf", content_type="application/pdf",
            content=content, max_size_bytes=100_000, max_pdf_pages=5,
            max_extracted_characters=100_000)
        assert parsed.page_count == 3
        assert required in parsed.extraction_metadata.pages[0].text
        candidates = extract_visual_elements(filename=f"{alias}.pdf", content=content,
            max_size_bytes=100_000, max_pdf_pages=5)
        assert any(item.page_number == 2 and item.element_type == "vector_path" for item in candidates)
        assert any(item.page_number == 3 for item in candidates)


def test_manifest_rejects_unapproved_real_sources_and_invalid_labels() -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["dataset_kind"] = "real"
    with pytest.raises(ValidationError, match="explicit local evaluation approval"):
        EvaluationManifest.model_validate(data)
    data["dataset_kind"] = "synthetic"
    data["cases"][0]["relevant_text"][0]["document"] = "unknown"
    with pytest.raises(ValidationError, match="unknown document"):
        EvaluationManifest.model_validate(data)


def test_manifest_bounds_selected_inference_and_runner_rejects_changed_source() -> None:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["documents"][0]["selected_visual_elements"] = [
        {"element_id": format(index, "064x"), "task": "describe_diagram"}
        for index in range(9)]
    with pytest.raises(ValidationError):
        EvaluationManifest.model_validate(data)
    client = _FakeClient()
    manifest = _small_manifest(client)
    manifest.documents[0].sha256 = "0" * 64
    report = asyncio.run(run_baseline(client, manifest,
        pdf_bytes={"pump_manual": b"synthetic"}, collection_id=uuid4()))
    assert report.cases[0].failure_stage == "parsing"
    assert report.cases[0].failure_code == "source_hash_mismatch"
    assert client.calls == []


class _Response:
    def __init__(self, payload: dict | list, status_code: int = 200) -> None:
        self.payload, self.status_code = payload, status_code

    def json(self) -> dict | list:
        return self.payload


class _FakeClient:
    def __init__(self, *, bad_coordinate: bool = False, fail_stage: str | None = None) -> None:
        self.calls: list[str] = []
        self.bad_coordinate, self.fail_stage = bad_coordinate, fail_stage
        self.document_id = "a" * 64
        self.element_id = "b" * 64
        self.chunk_id = "c" * 64
        self.interpretation_id = str(uuid4())

    async def post(self, url: str, **kwargs: object) -> _Response:
        self.calls.append(url)
        if self.fail_stage and self.fail_stage in url:
            return _Response({"detail": {"code": "synthetic_failure"}}, 502)
        if url == "/documents/parse":
            self.document_id = sha256(kwargs["files"]["file"][1]).hexdigest()
            return _Response({"document_id": self.document_id, "file_type": "pdf"})
        if url == "/documents/chunk":
            return _Response({"chunks": [{"chunk_id": self.chunk_id,
                "chunk_index": 0, "page_number": 1}]})
        if url.endswith("/documents/index"):
            return _Response({"document_id": self.document_id})
        if url.endswith("/visual-elements/index"):
            return _Response({})
        if url.endswith("/interpretations"):
            return _Response({"interpretation_id": self.interpretation_id,
                "analysis_task": "describe_diagram", "prompt_version": "v1",
                "model_name": "local-vlm", "model_digest": "d" * 64})
        if url.endswith("/visual-interpretations/index"):
            return _Response({})
        if url.endswith("/search") and not url.endswith("/multimodal-search"):
            return _Response({"results": [{"chunk_id": self.chunk_id}]})
        if url.endswith("/visual-search"):
            return _Response({"results": [{"element_id": self.element_id}]})
        if url.endswith("/multimodal-search"):
            return _Response({"results": [
                {"evidence_type": "text_chunk", "chunk_id": self.chunk_id},
                {"evidence_type": "visual_interpretation", "element_id": self.element_id}]})
        if url.endswith("/multimodal-answer"):
            query = kwargs.get("json", {}).get("query", "")
            if query == "What is the warranty period?":
                return _Response({"status": "insufficient_evidence", "claims": []})
            return _Response({"status": "answered", "claims": [{"citations": [
                {"evidence_type": "text_chunk", "document_id": self.document_id,
                 "source_id": "T1", "chunk_id": self.chunk_id,
                 "chunk_index": 0, "page_number": 1,
                 "filename": "pump_manual.pdf", "file_type": "pdf"},
                {"evidence_type": "visual_interpretation", "document_id": self.document_id,
                 "source_id": "V1", "filename": "pump_manual.pdf",
                 "element_id": self.element_id, "interpretation_id": self.interpretation_id,
                 "page_number": 2, "element_type": "vector_path",
                 "coordinate_system": "displayed_pdf_points",
                 "x0": 10.0001 if self.bad_coordinate else 10.0, "y0": 20.0,
                 "x1": 100.0, "y1": 80.0, "analysis_task": "describe_diagram",
                 "prompt_version": "v1", "interpretation_model_name": "local-vlm",
                 "interpretation_model_digest": "d" * 64,
                 "evidence_kind": "model_generated_interpretation",
                 "verification_status": "unverified"}]}]})
        raise AssertionError(f"Unexpected route: {url}")

    async def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append(url)
        return _Response([{"document_id": self.document_id, "element_id": self.element_id,
            "page_number": 2, "element_type": "vector_path",
            "coordinate_system": "displayed_pdf_points",
            "x0": "10.0000", "y0": "20.0000", "x1": "100.0000", "y1": "80.0000"}])


def _small_manifest(client: _FakeClient) -> EvaluationManifest:
    return EvaluationManifest.model_validate({
        "dataset_kind": "synthetic", "documents": [{"alias": "pump_manual", "path": "synthetic.pdf",
            "selected_visual_elements": [{"element_id": client.element_id, "task": "describe_diagram"}]}],
        "cases": [{"id": "mixed", "category": "mixed", "query": "What follows the pump?",
            "expected_status": "answered", "labels_verified_from_source": True,
            "relevant_text": [{"document": "pump_manual", "page_number": 1, "chunk_index": 0}],
            "relevant_visual": [{"document": "pump_manual", "page_number": 2,
                                 "element_id": client.element_id}]}]})


def test_runner_exercises_all_existing_stages_and_scores_provenance() -> None:
    client = _FakeClient()
    observed: list[str] = []
    report = asyncio.run(run_baseline(client, _small_manifest(client),
        pdf_bytes={"pump_manual": _synthetic_pdf("pump_manual")}, collection_id=uuid4(),
        generation_times=[0.01], answer_observer=lambda case_id, answer: observed.append(case_id)))
    case = report.cases[0]
    assert report.selected_visual_interpretations == 1
    assert report.visual_candidate_count == 1
    assert report.metrics["text_hit"].numerator == 1
    assert report.metrics["text_recall_micro"].denominator == 1
    assert report.metrics["visual_hit"].numerator == 1
    assert report.metrics["combined_coverage"].numerator == 1
    assert report.metrics["citation_identity"].numerator == 2
    assert report.metrics["citation_identity"].denominator == 2
    assert case.human_answer_supported is None
    assert case.human_visual_interpretation_accurate is None
    assert case.generation_seconds is None  # A previous timing is not this request's timing.
    assert any(path.endswith("/interpretations") for path in client.calls)
    assert any(path.endswith("/multimodal-answer") for path in client.calls)
    assert observed == ["mixed"]
    assert "The blue coolant pump" not in report.model_dump_json()


def test_runner_flags_coordinate_mutation_and_stage_failure() -> None:
    changed = _FakeClient(bad_coordinate=True)
    report = asyncio.run(run_baseline(changed, _small_manifest(changed),
        pdf_bytes={"pump_manual": b"synthetic"}, collection_id=uuid4()))
    assert report.cases[0].failure_stage == "citation_verification"
    assert report.metrics["citation_identity"].numerator == 1
    assert report.metrics["citation_identity"].denominator == 2
    failing = _FakeClient(fail_stage="visual-elements/index")
    report = asyncio.run(run_baseline(failing, _small_manifest(failing),
        pdf_bytes={"pump_manual": b"synthetic"}, collection_id=uuid4()))
    assert report.cases[0].failure_stage == "extraction"
    assert report.stage_failures == {"extraction": 1}
    assert "synthetic_failure" == report.cases[0].failure_code


def test_post_run_human_review_has_explicit_denominator() -> None:
    client = _FakeClient()
    report = asyncio.run(run_baseline(client, _small_manifest(client),
        pdf_bytes={"pump_manual": b"synthetic"}, collection_id=uuid4()))
    assert report.metrics["human_answer_supported"].denominator == 0
    judged = apply_human_reviews(report, {"mixed": HumanReview(
        answer_supported=False, visual_interpretation_accurate=True)})
    assert judged.metrics["human_answer_supported"].numerator == 0
    assert judged.metrics["human_answer_supported"].denominator == 1
    assert judged.metrics["human_visual_interpretation_accurate"].numerator == 1
    assert report.cases[0].human_answer_supported is None
    with pytest.raises(ValueError, match="unknown case"):
        apply_human_reviews(report, {"fabricated": HumanReview(answer_supported=True)})


def test_unanswerable_case_scores_abstention_separately() -> None:
    client = _FakeClient()
    manifest = _small_manifest(client)
    manifest.cases.append(type(manifest.cases[0]).model_validate({
        "id": "unknown_warranty", "category": "unanswerable",
        "query": "What is the warranty period?",
        "expected_status": "insufficient_evidence", "labels_verified_from_source": True}))
    report = asyncio.run(run_baseline(client, manifest,
        pdf_bytes={"pump_manual": b"synthetic"}, collection_id=uuid4()))
    assert report.metrics["abstention_correct"].numerator == 1
    assert report.metrics["abstention_correct"].denominator == 1
    assert report.metrics["answer_status_correct"].denominator == 2
    assert report.cases[1].citation_count == 0


@pytest.fixture
def approved_real_evaluation(request: pytest.FixtureRequest,
                             monkeypatch: pytest.MonkeyPatch) -> tuple[str, EvaluationManifest, dict[str, bytes]]:
    if os.getenv("RAG_AI_RUN_REAL_PDF_EVAL") != "1":
        pytest.skip("Set RAG_AI_RUN_REAL_PDF_EVAL=1 with approved PDFs and labels.")
    manifest_path = os.getenv("RAG_AI_REAL_PDF_MANIFEST")
    if not manifest_path:
        pytest.fail("Set RAG_AI_REAL_PDF_MANIFEST to an external approved manifest.")
    path = Path(manifest_path).resolve(strict=True)
    repo = Path(__file__).resolve().parents[1]
    if path.is_relative_to(repo):
        pytest.fail("The real-document manifest must be outside version control.")
    manifest = EvaluationManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.dataset_kind != "real" or not 2 <= len(manifest.documents) <= 3 or Counter(case.category for case in manifest.cases) != {
        "text": 4, "visual": 4, "mixed": 4, "unanswerable": 4}:
        pytest.fail("A real baseline needs four independently authored cases per category.")
    if not any(doc.selected_visual_elements for doc in manifest.documents):
        pytest.fail("Select bounded, inspected element IDs before live vision analysis.")
    pdfs: dict[str, bytes] = {}
    for document in manifest.documents:
        source = Path(document.path).resolve(strict=True)
        if source.is_relative_to(repo) or source.suffix.lower() != ".pdf" or source.stat().st_size > 10 * 1024 * 1024:
            pytest.fail("Approved PDFs must be external, bounded PDF files.")
        pdfs[document.alias] = source.read_bytes()
        if document.sha256 and sha256(pdfs[document.alias]).hexdigest() != document.sha256:
            pytest.fail("An approved PDF no longer matches its manifest hash.")
    base_url = os.getenv("RAG_AI_TEST_DATABASE_URL")
    if not base_url:
        pytest.skip("Set RAG_AI_TEST_DATABASE_URL for a verified disposable database.")
    try:
        safe_url = _validated_test_conninfo(base_url, os.environ)
    except ValueError as exc:
        pytest.fail(str(exc))
    monkeypatch.setenv("RAG_AI_TEST_DATABASE_URL", safe_url)
    # Approval, source paths/hashes, and destination are checked before first DB connection.
    return request.getfixturevalue("isolated_database_url"), manifest, pdfs


def test_live_real_pdf_baseline(approved_real_evaluation: tuple[str, EvaluationManifest, dict[str, bytes]],
                                monkeypatch: pytest.MonkeyPatch) -> None:
    dsn, manifest, pdfs = approved_real_evaluation
    MigrationRunner(dsn).upgrade()
    settings = Settings(app_env="development", database_url=dsn,
        enable_document_parse_endpoint=True, enable_document_index_endpoint=True,
        enable_document_retrieval_endpoint=True, enable_visual_element_storage_endpoint=True,
        enable_visual_interpretation_endpoint=True, enable_visual_retrieval_endpoint=True,
        enable_multimodal_retrieval_endpoint=True, enable_multimodal_answer_endpoint=True,
        embedding_model="embeddinggemma:300m", generation_model="qwen3:4b-instruct",
        vision_model="qwen3-vl:4b-instruct", _env_file=None)
    generation_times: list[float] = []
    original_generate = OllamaGenerationProvider.generate

    def timed_generate(self: OllamaGenerationProvider, **kwargs):
        start = perf_counter()
        try:
            return original_generate(self, **kwargs)
        finally:
            generation_times.append(round(perf_counter() - start, 6))

    monkeypatch.setattr(OllamaGenerationProvider, "generate", timed_generate)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(settings)),
                                     base_url="http://test", timeout=180.0) as client:
            return await run_baseline(client, manifest, pdf_bytes=pdfs,
                collection_id=uuid4(), generation_times=generation_times)

    report = asyncio.run(exercise())
    print("Sanitized real-PDF baseline:", report.model_dump_json())
    assert not report.stage_failures
