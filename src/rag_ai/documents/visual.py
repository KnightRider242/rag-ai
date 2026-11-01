"""Bounded page and embedded-image metadata inspection for PDFs."""

import math
from hashlib import sha256

import pymupdf
from pydantic import BaseModel, Field

from rag_ai.documents.parser import DocumentParseError, validate_upload_metadata

MAX_REPORTED_IMAGE_OCCURRENCES = 100
MAX_REPORTED_DRAWINGS = 1000
MAX_REPORTED_TABLES = 20
MAX_TABLE_CELLS = 200
MAX_TABLE_CHARACTERS = 10000
MAX_INSPECTION_RESPONSE_BYTES = 256000


class ImageBounds(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class VisualTable(BaseModel):
    bounds: ImageBounds
    row_count: int
    column_count: int
    cells: list[list[str | None]] = Field(description="Bounded, directly extracted cell text; null means unavailable.")


class DrawingKinds(BaseModel):
    line_paths: int = 0
    rectangle_paths: int = 0
    curve_paths: int = 0
    filled_paths: int = 0
    stroked_paths: int = 0


class VisualPage(BaseModel):
    page_number: int
    width: float = Field(description="Displayed page width in PDF points, after rotation.")
    height: float = Field(description="Displayed page height in PDF points, after rotation.")
    has_text: bool
    image_count: int = Field(description="All detected image occurrences, including off-page ones.")
    image_bounds: list[ImageBounds] = Field(
        description="Visible image boxes clipped to the displayed page; may be fewer than image_count.",
    )
    drawing_count: int = Field(description="PyMuPDF vector paths, including off-page paths; not a diagram count.")
    drawing_bounds: list[ImageBounds] = Field(description="Visible vector path boxes in displayed page points.")
    drawing_kinds: DrawingKinds
    table_count: int = Field(description="Heuristically detected tables, including off-page tables.")
    tables: list[VisualTable]


class VisualInspection(BaseModel):
    document_id: str
    filename: str
    page_count: int
    pages: list[VisualPage]


def _visible_box(page: pymupdf.Page, source: object, kind: str) -> ImageBounds | None:
    try:
        x0, y0, x1, y1 = source  # type: ignore[misc]
        values = (float(x0), float(y0), float(x1), float(y1))
        if not all(math.isfinite(value) for value in values) or not (values[0] <= values[2] and values[1] <= values[3]):
            raise ValueError("invalid geometry")
        displayed = pymupdf.Rect(values) * page.rotation_matrix
        if not all(math.isfinite(value) for value in displayed):
            raise ValueError("invalid geometry")
        visible = displayed & page.rect
        if visible.is_empty or visible.is_infinite:
            return None
        width, height = page.rect.width, page.rect.height
        if not (0 <= visible.x0 < visible.x1 <= width and 0 <= visible.y0 < visible.y1 <= height):
            raise ValueError("invalid geometry")
        return ImageBounds(x0=visible.x0, y0=visible.y0, x1=visible.x1, y1=visible.y1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DocumentParseError("invalid_pdf", f"The PDF has invalid {kind} geometry.", 422) from exc


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
            total_drawings = 0
            total_tables = 0
            total_characters = 0
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
                    visible = _visible_box(page, image["bbox"], "image")
                    if visible is not None:
                        bounds.append(visible)
                drawings = page.get_cdrawings()
                total_drawings += len(drawings)
                if total_drawings > MAX_REPORTED_DRAWINGS:
                    raise DocumentParseError("too_many_drawings", "The PDF exceeds the vector path limit.", 413)
                drawing_bounds = [box for drawing in drawings
                                  if (box := _visible_box(page, drawing["rect"], "drawing")) is not None]
                kinds = DrawingKinds(
                    line_paths=sum(any(item[0] == "l" for item in path["items"]) for path in drawings),
                    rectangle_paths=sum(any(item[0] == "re" for item in path["items"]) for path in drawings),
                    curve_paths=sum(any(item[0] in {"c", "qu"} for item in path["items"]) for path in drawings),
                    filled_paths=sum("f" in path.get("type", "") for path in drawings),
                    stroked_paths=sum("s" in path.get("type", "") for path in drawings),
                )
                # PyMuPDF's line strategy detects drawn grids; its text strategy
                # provides a heuristic fallback for borderless text tables.
                found = page.find_tables().tables
                if not found:
                    found = page.find_tables(strategy="text").tables
                total_tables += len(found)
                if total_tables > MAX_REPORTED_TABLES:
                    raise DocumentParseError("too_many_tables", "The PDF exceeds the table limit.", 413)
                tables: list[VisualTable] = []
                for table in found:
                    if table.row_count * table.col_count > MAX_TABLE_CELLS:
                        raise DocumentParseError("too_many_cells", "A table exceeds the cell limit.", 413)
                    cells = table.extract()
                    total_characters += sum(len(cell or "") for row in cells for cell in row)
                    if total_characters > MAX_TABLE_CHARACTERS:
                        raise DocumentParseError("table_text_too_large", "Extracted table text exceeds the limit.", 413)
                    box = _visible_box(page, table.bbox, "table")
                    if box is not None:
                        tables.append(VisualTable(bounds=box, row_count=table.row_count,
                                                  column_count=table.col_count, cells=cells))
                pages.append(VisualPage(
                    page_number=page_number, width=width, height=height,
                    has_text=bool(page.get_text("text").strip()),
                    image_count=len(images), image_bounds=bounds,
                    drawing_count=len(drawings), drawing_bounds=drawing_bounds, drawing_kinds=kinds,
                    table_count=len(found), tables=tables,
                ))
            result = VisualInspection(
                document_id=sha256(content).hexdigest(), filename=safe_name,
                page_count=document.page_count, pages=pages,
            )
            if len(result.model_dump_json().encode("utf-8")) > MAX_INSPECTION_RESPONSE_BYTES:
                raise DocumentParseError("inspection_too_large", "Visual metadata exceeds the response limit.", 413)
            return result
    except DocumentParseError:
        raise
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError,
            ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise DocumentParseError("invalid_pdf", "The uploaded file is not a valid PDF.", 422) from exc
