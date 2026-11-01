"""Development-only PDF image metadata endpoint."""

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from rag_ai.core.config import Settings
from rag_ai.documents.parser import DocumentParseError, validate_upload_metadata
from rag_ai.documents.visual import VisualInspection, inspect_pdf
from rag_ai.documents.visual import ImageBounds
from rag_ai.documents.render import render_pdf

router = APIRouter(prefix="/documents", tags=["documents"])


def _region(x0: float | None, y0: float | None, x1: float | None, y1: float | None) -> ImageBounds | None:
    values = (x0, y0, x1, y1)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise DocumentParseError("invalid_region", "All four region coordinates are required.", 422)
    try:
        return ImageBounds(x0=x0, y0=y0, x1=x1, y1=y1)
    except ValidationError as exc:
        raise DocumentParseError("invalid_region", "The requested region is invalid.", 422) from exc


@router.post("/inspect", response_model=VisualInspection)
async def inspect_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF document")],
) -> VisualInspection:
    """Return bounded page and image metadata without storing the upload."""
    settings: Settings = request.app.state.settings
    try:
        _, file_type = validate_upload_metadata(file.filename, file.content_type)
        if file_type != "pdf":
            raise DocumentParseError("unsupported_file_type", "Only PDF files can be inspected.", 415)
        content = await file.read(settings.max_upload_size_bytes + 1)
        return await run_in_threadpool(
            inspect_pdf, filename=file.filename, content_type=file.content_type,
            content=content, max_size_bytes=settings.max_upload_size_bytes,
            max_pdf_pages=settings.max_pdf_pages,
        )
    except DocumentParseError as exc:
        raise HTTPException(
            status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)},
        ) from exc
    finally:
        await file.close()


@router.post("/preview", response_class=Response, responses={200: {"content": {"image/png": {}}}})
async def preview_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF document")],
    page_number: Annotated[int, Form(ge=1)],
    scale: Annotated[float, Form()] = 1.0,
    x0: Annotated[float | None, Form()] = None,
    y0: Annotated[float | None, Form()] = None,
    x1: Annotated[float | None, Form()] = None,
    y1: Annotated[float | None, Form()] = None,
) -> Response:
    """Return a bounded PNG of exactly one PDF page or displayed-page region."""
    settings: Settings = request.app.state.settings
    try:
        region = _region(x0, y0, x1, y1)
        content = await file.read(settings.max_upload_size_bytes + 1)
        rendered = await run_in_threadpool(
            render_pdf, filename=file.filename, content_type=file.content_type,
            content=content, page_number=page_number, region=region,
            max_size_bytes=settings.max_upload_size_bytes, max_pdf_pages=settings.max_pdf_pages,
            scale=scale,
        )
        return Response(content=rendered.png, media_type="image/png")
    except DocumentParseError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc
    finally:
        await file.close()
