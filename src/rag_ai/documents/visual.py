"""Bounded page and embedded-image metadata inspection for PDFs."""

import math
from hashlib import sha256

import pymupdf
from pydantic import BaseModel, Field

from rag_ai.documents.parser import DocumentParseError, validate_upload_metadata

MAX_REPORTED_IMAGE_OCCURRENCES = 100


class ImageBounds(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class VisualPage(BaseModel):
    page_number: int
    width: float = Field(description="Displayed page width in PDF points, after rotation.")
    height: float = Field(description="Displayed page height in PDF points, after rotation.")
    has_text: bool
    image_count: int = Field(description="All detected image occurrences, including off-page ones.")
    image_bounds: list[ImageBounds] = Field(
        description="Visible image boxes clipped to the displayed page; may be fewer than image_count.",
    )


class VisualInspection(BaseModel):
    document_id: str
    filename: str
    page_count: int
    pages: list[VisualPage]


def inspect_pdf(
    *, filename: str | None, content_type: str | None, content: bytes,
    max_size_bytes: int, max_pdf_pages: int,
) -> VisualInspection:
    """Report visible boxes in displayed page points without returning content."""
    safe_name, file_type = validate_upload_metadata(filename, content_type)
    if file_type != "pdf":
        raise DocumentParseError("unsupported_file_type", "Only PDF files can be inspected.", 415)
    if len(content) > max_size_bytes:
        raise DocumentParseError("file_too_large", "The upload exceeds the configured size limit.", 413)
    if not content or not content.startswith(b"%PDF-"):
        raise DocumentParseError("invalid_pdf", "The uploaded file is not a valid PDF.", 422)

    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise DocumentParseError("invalid_pdf", "Password-protected PDFs are not supported.", 422)
            if document.page_count == 0:
                raise DocumentParseError("empty_document", "The PDF has no pages.", 422)
            if document.page_count > max_pdf_pages:
                raise DocumentParseError("too_many_pages", "The PDF exceeds the configured page limit.", 413)

            pages: list[VisualPage] = []
            total_images = 0
            for page_number, page in enumerate(document, 1):
                width, height = float(page.rect.width), float(page.rect.height)
                if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                    raise DocumentParseError("invalid_pdf", "The PDF has invalid page geometry.", 422)
                images = page.get_image_info(hashes=False, xrefs=False)
                total_images += len(images)
                if total_images > MAX_REPORTED_IMAGE_OCCURRENCES:
                    raise DocumentParseError("too_many_images", "The PDF exceeds the image occurrence limit.", 413)
                bounds: list[ImageBounds] = []
                for image in images:
                    x0, y0, x1, y1 = image["bbox"]
                    if (not all(math.isfinite(value) for value in (x0, y0, x1, y1))
                            or not (x0 < x1 and y0 < y1)):
                        raise DocumentParseError("invalid_pdf", "The PDF has invalid image geometry.", 422)
                    # Image metadata uses unrotated coordinates. The API reports
                    # only the visible intersection in displayed page space.
                    displayed = pymupdf.Rect(x0, y0, x1, y1) * page.rotation_matrix
                    if not all(math.isfinite(value) for value in displayed):
                        raise DocumentParseError("invalid_pdf", "The PDF has invalid image geometry.", 422)
                    visible = displayed & page.rect
                    if visible.is_empty:
                        continue  # Count this occurrence, but it has no visible box.
                    if not (0 <= visible.x0 < visible.x1 <= width
                            and 0 <= visible.y0 < visible.y1 <= height):
                        raise DocumentParseError("invalid_pdf", "The PDF has invalid image geometry.", 422)
                    bounds.append(ImageBounds(
                        x0=visible.x0, y0=visible.y0, x1=visible.x1, y1=visible.y1,
                    ))
                pages.append(VisualPage(
                    page_number=page_number, width=width, height=height,
                    has_text=bool(page.get_text("text").strip()),
                    image_count=len(images), image_bounds=bounds,
                ))
            return VisualInspection(
                document_id=sha256(content).hexdigest(), filename=safe_name,
                page_count=document.page_count, pages=pages,
            )
    except DocumentParseError:
        raise
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError,
            ValueError, TypeError, KeyError, OverflowError) as exc:
        raise DocumentParseError("invalid_pdf", "The uploaded file is not a valid PDF.", 422) from exc
