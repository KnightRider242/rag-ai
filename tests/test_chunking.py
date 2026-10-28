"""Functional and security tests for deterministic document chunking."""

import asyncio
import logging
from hashlib import sha256
from pathlib import Path

import httpx
import pymupdf
import pytest
from pydantic import ValidationError
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_ai.core.config import Settings
from rag_ai.documents.chunker import ChunkedDocument, ChunkingError, chunk_document
from rag_ai.documents.parser import ParsedDocument, parse_document
from rag_ai.main import create_app


def text_document(content: str, filename: str = "notes.txt") -> ParsedDocument:
    file_type = "md" if filename.endswith(".md") else "txt"
    return parse_document(
        filename=filename,
        content_type="text/markdown" if file_type == "md" else "text/plain",
        content=content.encode("utf-8"),
        max_size_bytes=1_000_000,
        max_pdf_pages=500,
        max_extracted_characters=1_000_000,
    )


def pdf_document(*page_texts: str) -> ParsedDocument:
    with pymupdf.open() as pdf:
        for text in page_texts:
            page = pdf.new_page()
            if text:
                page.insert_text((72, 72), text)
        content = pdf.tobytes()
    return parse_document(
        filename="report.pdf",
        content_type="application/pdf",
        content=content,
        max_size_bytes=1_000_000,
        max_pdf_pages=500,
        max_extracted_characters=1_000_000,
    )


def post_chunk(settings: Settings, filename: str, content: bytes, content_type: str) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(settings))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/documents/chunk", files={"file": (filename, content, content_type)}
            )

    return asyncio.run(request())


def development_settings(**overrides: int) -> Settings:
    return Settings(app_env="development", enable_document_parse_endpoint=True, **overrides)


def test_short_document_produces_one_chunk() -> None:
    result = chunk_document(text_document("Short text"), chunk_size=100, chunk_overlap=10)

    assert result.total_chunks == 1
    assert result.chunks[0].text == "Short text"
    assert result.chunks[0].chunk_index == 0


def test_long_document_produces_ordered_size_bounded_chunks() -> None:
    result = chunk_document(text_document("abcdefghijklmnopqrstuvwxyz" * 8), chunk_size=64, chunk_overlap=16)

    assert result.total_chunks > 1
    assert [chunk.chunk_index for chunk in result.chunks] == list(range(result.total_chunks))
    assert all(0 < len(chunk.text) <= 64 for chunk in result.chunks)


def test_overlap_applies_to_character_splits() -> None:
    result = chunk_document(text_document("a" * 150), chunk_size=64, chunk_overlap=16)

    assert [len(chunk.text) for chunk in result.chunks] == [64, 64, 54]
    assert result.chunk_overlap == 16


@pytest.mark.parametrize("size", [64, 10_000])
def test_minimum_and_maximum_chunk_sizes_are_accepted(size: int) -> None:
    result = chunk_document(text_document("synthetic text"), chunk_size=size, chunk_overlap=0)

    assert result.total_chunks == 1
    assert result.chunk_size == size


@pytest.mark.parametrize(
    ("size", "overlap", "limit"),
    [(0, 0, 10), (63, 0, 10), (64, -1, 10), (64, 33, 10),
     (64, 64, 10), (64, 0, 0), (10_001, 0, 10)],
)
def test_invalid_chunk_configuration_is_rejected(size: int, overlap: int, limit: int) -> None:
    with pytest.raises(ChunkingError) as exc_info:
        chunk_document(
            text_document("valid"), chunk_size=size, chunk_overlap=overlap,
            max_chunks_per_document=limit,
        )

    assert exc_info.value.code == "invalid_chunk_configuration"


def test_settings_reject_excessive_overlap() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP must not exceed half"):
        Settings(chunk_size=64, chunk_overlap=33)


@pytest.mark.parametrize(
    "overrides",
    [
        {"chunk_size": 0},
        {"chunk_size": 63},
        {"chunk_size": 10_001},
        {"chunk_overlap": -1},
        {"max_chunks_per_document": 0},
        {"max_chunks_per_document": 10_001},
    ],
)
def test_settings_reject_out_of_range_chunking_values(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        Settings(**overrides)


def test_chunk_ids_are_deterministic_and_include_configuration() -> None:
    document = text_document("abcdefghij" * 20)
    first = chunk_document(document, chunk_size=64, chunk_overlap=16)
    second = chunk_document(document, chunk_size=64, chunk_overlap=16)
    changed = chunk_document(document, chunk_size=64, chunk_overlap=15)

    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]
    assert first.chunks[0].chunk_id != changed.chunks[0].chunk_id


def test_identical_documents_have_identical_chunks() -> None:
    first = chunk_document(text_document("same bytes", "one.txt"), chunk_size=64, chunk_overlap=16)
    second = chunk_document(text_document("same bytes", "two.txt"), chunk_size=64, chunk_overlap=16)

    assert first.document_id == second.document_id
    assert [chunk.chunk_id for chunk in first.chunks] == [chunk.chunk_id for chunk in second.chunks]


def test_different_document_content_changes_ids() -> None:
    first = chunk_document(text_document("first"), chunk_size=64, chunk_overlap=16)
    second = chunk_document(text_document("second"), chunk_size=64, chunk_overlap=16)

    assert first.document_id != second.document_id
    assert first.chunks[0].chunk_id != second.chunks[0].chunk_id


def test_pdf_chunks_preserve_page_boundaries_and_metadata() -> None:
    result = chunk_document(pdf_document("A" * 100, "B" * 100), chunk_size=64, chunk_overlap=16)

    assert result.file_type == "pdf"
    assert [chunk.page_number for chunk in result.chunks] == [1, 1, 2, 2]
    assert all(chunk.filename == "report.pdf" for chunk in result.chunks)
    assert all("A" not in chunk.text or "B" not in chunk.text for chunk in result.chunks)
    assert len({chunk.chunk_id for chunk in result.chunks}) == result.total_chunks


def test_empty_pdf_page_is_skipped() -> None:
    result = chunk_document(pdf_document("First", "", "Third"), chunk_size=100, chunk_overlap=0)

    assert [chunk.page_number for chunk in result.chunks] == [1, 3]


def test_identical_text_on_different_pdf_pages_has_distinct_chunk_ids() -> None:
    result = chunk_document(pdf_document("Same", "Same"), chunk_size=100, chunk_overlap=0)

    assert [chunk.page_number for chunk in result.chunks] == [1, 2]
    assert result.chunks[0].chunk_id != result.chunks[1].chunk_id


@pytest.mark.parametrize("filename", ["notes.txt", "notes.md"])
def test_text_and_markdown_metadata_has_no_page_number(filename: str) -> None:
    result = chunk_document(text_document("Content", filename), chunk_size=100, chunk_overlap=0)

    assert result.filename == filename
    assert result.chunks[0].file_type == ("md" if filename.endswith(".md") else "txt")
    assert result.chunks[0].page_number is None


def test_chunk_count_limit_is_enforced() -> None:
    with pytest.raises(ChunkingError) as exc_info:
        chunk_document(
            text_document("abcdefghijklmnopqrstuvwxyz" * 8), chunk_size=64, chunk_overlap=0,
            max_chunks_per_document=2,
        )

    assert exc_info.value.code == "too_many_chunks"
    assert exc_info.value.status_code == 413


def test_over_budget_input_is_rejected_before_splitter_materializes_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(self: RecursiveCharacterTextSplitter, text: str) -> list[str]:
        raise AssertionError("splitter should not run for provably excessive input")

    monkeypatch.setattr(RecursiveCharacterTextSplitter, "split_text", fail_if_called)
    with pytest.raises(ChunkingError, match="chunk count limit"):
        chunk_document(text_document("a" * 100_000), chunk_size=64,
                       chunk_overlap=0, max_chunks_per_document=2)


def test_document_instructions_remain_inert(tmp_path: Path) -> None:
    target = tmp_path / "executed"
    payload = f"__import__('pathlib').Path({str(target)!r}).write_text('bad')"

    result = chunk_document(text_document(payload), chunk_size=max(64, len(payload) + 1), chunk_overlap=0)

    assert payload in result.chunks[0].text
    assert not target.exists()


def test_document_content_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    marker = "SYNTHETIC_PRIVATE_CHUNK_MARKER_987"
    with caplog.at_level(logging.DEBUG):
        result = chunk_document(text_document(marker), chunk_size=100, chunk_overlap=0)

    assert result.total_chunks == 1
    assert marker not in caplog.text


def test_chunk_route_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENABLE_DOCUMENT_PARSE_ENDPOINT", raising=False)
    settings = Settings(_env_file=None)

    assert "/documents/chunk" not in create_app(settings).openapi()["paths"]
    assert post_chunk(settings, "notes.txt", b"hello", "text/plain").status_code == 404


def test_chunk_route_unavailable_in_production() -> None:
    settings = Settings(app_env="production", enable_document_parse_endpoint=False)

    assert "/documents/chunk" not in create_app(settings).openapi()["paths"]
    assert post_chunk(settings, "notes.txt", b"hello", "text/plain").status_code == 404


def test_chunk_route_unavailable_in_test_environment() -> None:
    settings = Settings(app_env="test", enable_document_parse_endpoint=True)

    assert "/documents/chunk" not in create_app(settings).openapi()["paths"]
    assert post_chunk(settings, "notes.txt", b"hello", "text/plain").status_code == 404


def test_explicit_development_configuration_enables_chunk_route() -> None:
    settings = development_settings(chunk_size=64, chunk_overlap=16)
    response = post_chunk(settings, "notes.txt", b"hello world", "text/plain")

    assert "/documents/chunk" in create_app(settings).openapi()["paths"]
    assert response.status_code == 200
    result = ChunkedDocument.model_validate(response.json())
    assert result.total_chunks == 1
    assert result.chunks[0].document_id == sha256(b"hello world").hexdigest()
    assert result.chunk_size == 64
    assert result.chunk_overlap == 16


@pytest.mark.parametrize(
    ("filename", "content_type", "content", "expected_status"),
    [
        ("bad.csv", "text/csv", b"invalid", 415),
        ("bad.pdf", "application/pdf", b"not a PDF", 422),
        ("bad.txt", "text/plain", b"\xff", 422),
    ],
)
def test_chunk_route_reuses_ingestion_validation(
    filename: str, content_type: str, content: bytes, expected_status: int
) -> None:
    response = post_chunk(development_settings(), filename, content, content_type)

    assert response.status_code == expected_status


def test_chunk_route_enforces_chunk_count_limit() -> None:
    settings = development_settings(chunk_size=64, chunk_overlap=0, max_chunks_per_document=2)
    response = post_chunk(settings, "notes.txt", b"abcdefghijklmnopqrstuvwxyz" * 8, "text/plain")

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_many_chunks"


def test_chunk_route_reuses_upload_size_limit() -> None:
    settings = development_settings(max_upload_size_bytes=3)
    response = post_chunk(settings, "notes.txt", b"four", "text/plain")

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "file_too_large"
