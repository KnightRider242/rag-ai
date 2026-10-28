"""Development endpoint for document extraction."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from rag_ai.core.config import Settings
from rag_ai.documents.parser import (
    DocumentParseError,
    ParsedDocument,
    parse_document,
    validate_upload_metadata,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/parse", response_model=ParsedDocument)
async def parse_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF, TXT, or Markdown document")],
) -> ParsedDocument:
    """Extract text and metadata without persisting the upload."""
    settings: Settings = request.app.state.settings
    try:
        validate_upload_metadata(file.filename, file.content_type)
        content = await file.read(settings.max_upload_size_bytes + 1)
        return parse_document(
            filename=file.filename,
            content_type=file.content_type,
            content=content,
            max_size_bytes=settings.max_upload_size_bytes,
            max_pdf_pages=settings.max_pdf_pages,
            max_extracted_characters=settings.max_extracted_characters,
        )
    except DocumentParseError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc
    finally:
        await file.close()
