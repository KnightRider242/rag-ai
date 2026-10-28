"""Synthetic document extraction and HTTP contract tests."""

import asyncio
from hashlib import sha256
from pathlib import Path

import httpx
import pymupdf
import pytest

from rag_ai.core.config import Settings
from rag_ai.documents.parser import DocumentParseError, ParsedDocument, parse_document
from rag_ai.main import create_app


def pdf_bytes(*page_texts: str) -> bytes:
    with pymupdf.open() as document:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        return document.tobytes()


def parse(
    filename: str,
    content_type: str,
    content: bytes,
    max_size: int = 1024 * 1024,
    max_pages: int = 500,
    max_characters: int = 2_000_000,
) -> ParsedDocument:
    return parse_document(
        filename=filename,
        content_type=content_type,
        content=content,
        max_size_bytes=max_size,
        max_pdf_pages=max_pages,
        max_extracted_characters=max_characters,
    )


def post_file(
    filename: str,
    content_type: str,
    content: bytes,
    settings: Settings | None = None,
) -> httpx.Response:
    if settings is None:
        settings = Settings(app_env="development", enable_document_parse_endpoint=True)
    app = create_app(settings)

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/documents/parse", files={"file": (filename, content, content_type)}
            )

    return asyncio.run(request())


def test_valid_pdf_parsing() -> None:
    result = parse("report.pdf", "application/pdf", pdf_bytes("Quarterly report"))

    assert result.file_type == "pdf"
    assert result.page_count == 1
    assert "Quarterly report" in result.text


def test_multi_page_pdf_extraction() -> None:
    result = parse("report.pdf", "application/pdf", pdf_bytes("First page", "Second page"))

    assert result.page_count == 2
    assert [page.page_number for page in result.extraction_metadata.pages] == [1, 2]
    assert "First page" in result.extraction_metadata.pages[0].text
    assert "Second page" in result.extraction_metadata.pages[1].text


def test_plain_text_extraction() -> None:
    result = parse("notes.txt", "text/plain", "Café\nNext line".encode())

    assert result.text == "Café\nNext line"
    assert result.page_count is None
    assert result.extraction_metadata.encoding == "utf-8"


def test_markdown_extraction() -> None:
    result = parse("readme.md", "text/markdown", b"# Heading\nBody")

    assert result.file_type == "md"
    assert result.text == "# Heading\nBody"


@pytest.mark.parametrize("content", [b"", b"  \n\t"])
def test_empty_text_files(content: bytes) -> None:
    with pytest.raises(DocumentParseError, match="empty|no text") as exc_info:
        parse("empty.txt", "text/plain", content)

    assert exc_info.value.code == "empty_document"


def test_empty_pdf() -> None:
    with pytest.raises(DocumentParseError) as exc_info:
        parse("blank.pdf", "application/pdf", pdf_bytes(""))

    assert exc_info.value.code == "empty_document"


def test_scanned_pdf_without_text() -> None:
    with pymupdf.open() as document:
        page = document.new_page()
        image = pymupdf.Pixmap(pymupdf.csRGB, 1, 1, bytes([0, 0, 0]), False)
        page.insert_image(pymupdf.Rect(72, 72, 200, 200), pixmap=image)
        content = document.tobytes()

    with pytest.raises(DocumentParseError) as exc_info:
        parse("scan.pdf", "application/pdf", content)

    assert exc_info.value.code == "scanned_pdf"


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [("data.csv", "text/csv"), ("data.pdf", "text/plain"), ("data.txt", "image/png")],
)
def test_unsupported_file_type_or_mime(filename: str, content_type: str) -> None:
    with pytest.raises(DocumentParseError) as exc_info:
        parse(filename, content_type, b"content")

    assert exc_info.value.status_code == 415


def test_malformed_pdf() -> None:
    with pytest.raises(DocumentParseError) as exc_info:
        parse("broken.pdf", "application/pdf", b"%PDF-1.7\nnot a PDF")

    assert exc_info.value.code == "invalid_pdf"


def test_invalid_text_encoding() -> None:
    with pytest.raises(DocumentParseError) as exc_info:
        parse("legacy.txt", "text/plain", b"\xff\xfe")

    assert exc_info.value.code == "invalid_encoding"


def test_oversized_upload() -> None:
    with pytest.raises(DocumentParseError) as exc_info:
        parse("large.txt", "text/plain", b"1234", max_size=3)

    assert exc_info.value.status_code == 413


def test_metadata_and_safe_filename() -> None:
    content = pdf_bytes("Metadata")
    result = parse(r"C:\\uploads\\report.pdf", "application/pdf", content)

    assert result.filename == "report.pdf"
    assert result.document_id == sha256(content).hexdigest()
    assert result.extraction_metadata.pages[0].page_number == 1


def test_document_id_is_stable_for_same_bytes() -> None:
    first = parse("one.txt", "text/plain", b"same bytes")
    second = parse("two.txt", "text/plain", b"same bytes")

    assert first.document_id == second.document_id


def test_api_response_validation() -> None:
    content = pdf_bytes("API sample")
    response = post_file("sample.pdf", "application/pdf", content)

    assert response.status_code == 200
    result = ParsedDocument.model_validate(response.json())
    assert result.document_id == sha256(content).hexdigest()
    assert result.page_count == 1
    assert result.extraction_metadata.pages[0].page_number == 1


@pytest.mark.parametrize(
    ("filename", "content_type", "content"),
    [
        ("notes.txt", "text/plain", b"Plain text sample"),
        ("notes.md", "text/markdown", b"# Markdown sample"),
    ],
)
def test_api_parses_text_samples(filename: str, content_type: str, content: bytes) -> None:
    response = post_file(filename, content_type, content)

    assert response.status_code == 200
    result = ParsedDocument.model_validate(response.json())
    assert result.text == content.decode("utf-8")
    assert result.extraction_metadata.encoding == "utf-8"


def test_api_rejects_malformed_pdf() -> None:
    response = post_file("bad.pdf", "application/pdf", b"%PDF-1.7\nbroken")

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_pdf"


def test_api_rejects_oversized_file() -> None:
    settings = Settings(
        app_env="development", enable_document_parse_endpoint=True, max_upload_size_bytes=3
    )
    response = post_file("large.txt", "text/plain", b"1234", settings)

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "file_too_large"


def test_security_rejects_pdf_mime_and_extension_spoofing() -> None:
    with pytest.raises(DocumentParseError) as exc_info:
        parse("fake.pdf", "application/pdf", b"This is UTF-8 text, not a PDF")

    assert exc_info.value.code == "invalid_pdf"


@pytest.mark.parametrize("filename", ["../../private/notes.txt", r"C:\\private\\notes.txt"])
def test_security_filename_paths_are_not_trusted(filename: str) -> None:
    result = parse(filename, "text/plain", b"safe content")

    assert result.filename == "notes.txt"


@pytest.mark.parametrize("filename", ["a" * 256 + ".txt", "bad\x00name.txt"])
def test_security_invalid_filenames_are_rejected(filename: str) -> None:
    with pytest.raises(DocumentParseError) as exc_info:
        parse(filename, "text/plain", b"content")

    assert exc_info.value.code == "invalid_filename"


def test_security_document_instructions_are_not_executed(tmp_path: Path) -> None:
    target = tmp_path / "executed"
    content = f"__import__('pathlib').Path({str(target)!r}).write_text('bad')".encode()

    result = parse("instructions.txt", "text/plain", content)

    assert result.text == content.decode()
    assert not target.exists()


def test_security_page_limit() -> None:
    with pytest.raises(DocumentParseError) as exc_info:
        parse("many.pdf", "application/pdf", pdf_bytes("one", "two"), max_pages=1)

    assert exc_info.value.code == "too_many_pages"
    assert exc_info.value.status_code == 413


def test_security_api_page_limit() -> None:
    settings = Settings(app_env="development", enable_document_parse_endpoint=True, max_pdf_pages=1)
    response = post_file("many.pdf", "application/pdf", pdf_bytes("one", "two"), settings)

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "too_many_pages"


@pytest.mark.parametrize(
    ("filename", "content_type", "content"),
    [
        ("long.txt", "text/plain", b"1234"),
        ("long.pdf", "application/pdf", pdf_bytes("Long extracted page")),
    ],
)
def test_security_extracted_text_limit(filename: str, content_type: str, content: bytes) -> None:
    with pytest.raises(DocumentParseError) as exc_info:
        parse(filename, content_type, content, max_characters=3)

    assert exc_info.value.code == "extracted_text_too_large"
    assert exc_info.value.status_code == 413


def test_security_document_content_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    marker = "PRIVATE_SYNTHETIC_MARKER_12345"
    response = post_file("private.txt", "text/plain", marker.encode())

    assert response.status_code == 200
    assert marker not in caplog.text


def test_pdf_joined_text_respects_extracted_character_limit() -> None:
    with pymupdf.open() as document:
        for text in ("First", "Second"):
            page = document.new_page()
            page.insert_text((72, 72), text)
        content = document.tobytes()

    with pymupdf.open(stream=content, filetype="pdf") as document:
        page_text_length = sum(len(page.get_text("text")) for page in document)

    with pytest.raises(DocumentParseError) as exc_info:
        parse("joined.pdf", "application/pdf", content, max_characters=page_text_length)

    assert exc_info.value.code == "extracted_text_too_large"
    result = parse("joined.pdf", "application/pdf", content, max_characters=page_text_length + 2)
    assert len(result.text) == page_text_length + 2
