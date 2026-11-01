"""Development-only PDF image metadata endpoint."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from rag_ai.core.config import Settings
from rag_ai.documents.parser import DocumentParseError, validate_upload_metadata
from rag_ai.documents.visual import VisualInspection, inspect_pdf

router = APIRouter(prefix="/documents", tags=["documents"])


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
