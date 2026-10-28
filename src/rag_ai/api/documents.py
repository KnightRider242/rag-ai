"""Development endpoint for document extraction."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from rag_ai.core.config import Settings
from rag_ai.documents.chunker import ChunkedDocument, ChunkingError, chunk_document
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
    return await _parse_uploaded_file(file, settings)


@router.post("/chunk", response_model=ChunkedDocument)
async def chunk_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF, TXT, or Markdown document")],
) -> ChunkedDocument:
    """Parse and chunk one upload without persisting it."""
    settings: Settings = request.app.state.settings
    document = await _parse_uploaded_file(file, settings)
    try:
        return chunk_document(
            document,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            max_chunks_per_document=settings.max_chunks_per_document,
        )
    except ChunkingError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc


async def _parse_uploaded_file(file: UploadFile, settings: Settings) -> ParsedDocument:
    """Apply the same ingestion validation to both development routes."""
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
