"""Validated extraction of PDF, UTF-8 text, and Markdown uploads."""

from hashlib import sha256
from typing import Literal, cast

import pymupdf
from pydantic import BaseModel, Field

FileType = Literal["pdf", "txt", "md"]

ALLOWED_CONTENT_TYPES: dict[str, set[str]] = {
    "pdf": {"application/pdf"},
    "txt": {"text/plain"},
    "md": {"text/markdown", "text/x-markdown", "text/plain"},
}


class PageText(BaseModel):
    page_number: int
    text: str


class ExtractionMetadata(BaseModel):
    encoding: str | None = None
    pages: list[PageText] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    document_id: str
    filename: str
    file_type: FileType
    page_count: int | None
    text: str
    extraction_metadata: ExtractionMetadata


class DocumentParseError(ValueError):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def validate_upload_metadata(filename: str | None, content_type: str | None) -> tuple[str, FileType]:
    """Validate the declared format and return a safe display filename."""
    safe_name = (filename or "").replace("\\", "/").split("/")[-1]
    if safe_name in {"", ".", ".."} or "." not in safe_name:
        raise DocumentParseError("unsupported_file_type", "A .pdf, .txt, or .md filename is required.", 415)
    if len(safe_name) > 255 or any(ord(character) < 32 or ord(character) == 127 for character in safe_name):
        raise DocumentParseError("invalid_filename", "The upload filename is invalid.", 422)

    extension = safe_name.rsplit(".", 1)[-1].lower()
    if extension not in ALLOWED_CONTENT_TYPES:
        raise DocumentParseError("unsupported_file_type", "Only PDF, TXT, and Markdown files are supported.", 415)

    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if media_type not in ALLOWED_CONTENT_TYPES[extension]:
        raise DocumentParseError("invalid_content_type", "The upload content type does not match its file type.", 415)
    return safe_name, cast(FileType, extension)


def parse_document(
    *,
    filename: str | None,
    content_type: str | None,
    content: bytes,
    max_size_bytes: int,
    max_pdf_pages: int,
    max_extracted_characters: int,
) -> ParsedDocument:
    """Extract a document from bounded in-memory bytes."""
    safe_name, file_type = validate_upload_metadata(filename, content_type)
    if len(content) > max_size_bytes:
        raise DocumentParseError("file_too_large", "The upload exceeds the configured size limit.", 413)
    if not content:
        raise DocumentParseError("empty_document", "The uploaded file is empty.", 422)

    if file_type == "pdf":
        text, page_count, metadata = _parse_pdf(content, max_pdf_pages, max_extracted_characters)
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParseError("invalid_encoding", "Text files must use UTF-8 encoding.", 422) from exc
        if not text.strip():
            raise DocumentParseError("empty_document", "The document has no text content.", 422)
        if len(text) > max_extracted_characters:
            raise DocumentParseError("extracted_text_too_large", "The extracted text exceeds the configured limit.", 413)
        page_count = None
        metadata = ExtractionMetadata(encoding="utf-8")

    return ParsedDocument(
        document_id=sha256(content).hexdigest(),
        filename=safe_name,
        file_type=file_type,
        page_count=page_count,
        text=text,
        extraction_metadata=metadata,
    )


def _parse_pdf(
    content: bytes, max_pdf_pages: int, max_extracted_characters: int
) -> tuple[str, int, ExtractionMetadata]:
    if not content.startswith(b"%PDF-"):
        raise DocumentParseError("invalid_pdf", "The uploaded file is not a valid PDF.", 422)
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise DocumentParseError("invalid_pdf", "Password-protected PDFs are not supported.", 422)
            if document.page_count == 0:
                raise DocumentParseError("empty_document", "The PDF has no pages.", 422)
            if document.page_count > max_pdf_pages:
                raise DocumentParseError("too_many_pages", "The PDF exceeds the configured page limit.", 413)

            pages: list[PageText] = []
            total_characters = 0
            for index, page in enumerate(document):
                page_text = page.get_text("text")
                total_characters += len(page_text) + (2 if index else 0)
                if total_characters > max_extracted_characters:
                    raise DocumentParseError(
                        "extracted_text_too_large", "The extracted text exceeds the configured limit.", 413
                    )
                pages.append(PageText(page_number=index + 1, text=page_text))
            if not any(page.text.strip() for page in pages):
                has_images = any(page.get_image_info() for page in document)
                code = "scanned_pdf" if has_images else "empty_document"
                message = (
                    "The PDF has no extractable text; OCR is not available."
                    if has_images
                    else "The PDF has no text content."
                )
                raise DocumentParseError(code, message, 422)
            return "\n\n".join(page.text for page in pages), document.page_count, ExtractionMetadata(pages=pages)
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise DocumentParseError("invalid_pdf", "The uploaded file is not a valid PDF.", 422) from exc
