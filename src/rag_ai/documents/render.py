"""Bounded, in-memory PDF rendering in displayed page coordinates."""

import math
from dataclasses import dataclass

import pymupdf

from rag_ai.documents.parser import DocumentParseError, validate_upload_metadata
from rag_ai.documents.visual import ImageBounds

MAX_RENDER_PIXELS = 4_000_000
MAX_RENDER_SIDE = 2048
MAX_RENDER_BYTES = 4_000_000
DEFAULT_RENDER_SCALE = 1.0


@dataclass(frozen=True)
class RenderedPage:
    png: bytes
    region: ImageBounds
    page_number: int
    width: int
    height: int


def render_pdf(
    *, filename: str | None, content_type: str | None, content: bytes,
    page_number: int, region: ImageBounds | None, max_size_bytes: int,
    max_pdf_pages: int, scale: float = DEFAULT_RENDER_SCALE,
    max_pixels: int = MAX_RENDER_PIXELS,
) -> RenderedPage:
    """Render one visible page region; source coordinates use the displayed page."""
    _, file_type = validate_upload_metadata(filename, content_type)
    if file_type != "pdf":
        raise DocumentParseError("unsupported_file_type", "Only PDF files can be rendered.", 415)
    if len(content) > max_size_bytes:
        raise DocumentParseError("file_too_large", "The upload exceeds the configured size limit.", 413)
    if not content.startswith(b"%PDF-"):
        raise DocumentParseError("invalid_pdf", "The uploaded file is not a valid PDF.", 422)
    if not math.isfinite(scale) or not 0.5 <= scale <= 2:
        raise DocumentParseError("invalid_scale", "The requested rendering scale is invalid.", 422)
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise DocumentParseError("invalid_pdf", "Password-protected PDFs are not supported.", 422)
            if document.page_count > max_pdf_pages:
                raise DocumentParseError("too_many_pages", "The PDF exceeds the configured page limit.", 413)
            if not 1 <= page_number <= document.page_count:
                raise DocumentParseError("invalid_page", "The requested page does not exist.", 422)
            page = document[page_number - 1]
            width, height = page.rect.width, page.rect.height
            if not all(math.isfinite(value) and value > 0 for value in (width, height)):
                raise DocumentParseError("invalid_pdf", "The PDF has invalid page geometry.", 422)
            visible = pymupdf.Rect(0, 0, width, height)
            if region is not None:
                values = (region.x0, region.y0, region.x1, region.y1)
                if not all(math.isfinite(value) for value in values) or not (
                    0 <= region.x0 < region.x1 <= width and 0 <= region.y0 < region.y1 <= height
                ):
                    raise DocumentParseError("invalid_region", "The requested region is invalid.", 422)
                visible = pymupdf.Rect(*values)
            pixel_width = math.ceil(visible.width * scale) + 2
            pixel_height = math.ceil(visible.height * scale) + 2
            if (pixel_width > MAX_RENDER_SIDE or pixel_height > MAX_RENDER_SIDE
                    or pixel_width * pixel_height > min(max_pixels, MAX_RENDER_PIXELS)):
                raise DocumentParseError("render_too_large", "The rendered region exceeds the pixel limit.", 413)
            # get_pixmap accepts displayed-page clip coordinates, including
            # on rotated pages. Do not derotate the caller's region here.
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=visible, alpha=False)
            if (pixmap.width > MAX_RENDER_SIDE or pixmap.height > MAX_RENDER_SIDE
                    or pixmap.width * pixmap.height > min(max_pixels, MAX_RENDER_PIXELS)):
                raise DocumentParseError("render_too_large", "The rendered region exceeds the pixel limit.", 413)
            png = pixmap.tobytes("png")
            if len(png) > MAX_RENDER_BYTES:
                raise DocumentParseError("render_too_large", "The rendered image exceeds the byte limit.", 413)
            return RenderedPage(
                png=png, region=ImageBounds(x0=visible.x0, y0=visible.y0, x1=visible.x1, y1=visible.y1),
                page_number=page_number, width=pixmap.width, height=pixmap.height,
            )
    except DocumentParseError:
        raise
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError,
            ValueError, TypeError, IndexError, OverflowError) as exc:
        raise DocumentParseError("invalid_pdf", "The uploaded file is not a valid PDF.", 422) from exc
