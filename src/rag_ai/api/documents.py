"""Development endpoint for document extraction."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from rag_ai.core.config import Settings
from rag_ai.documents.chunker import ChunkedDocument, ChunkingError, chunk_document
from rag_ai.documents.indexer import IndexedDocument, IndexingError, index_document
from rag_ai.documents.parser import (
    DocumentParseError,
    ParsedDocument,
    parse_document,
    validate_upload_metadata,
)
from rag_ai.embeddings.base import EmbeddingError
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.storage.postgres import (
    ModelDimensionMismatch, ModelRevisionMismatch, PostgresIndexStore, StorageError,
)

router = APIRouter(prefix="/documents", tags=["documents"])
index_router = APIRouter(prefix="/collections", tags=["collections"])


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
    document, _ = await _read_and_parse_uploaded_file(file, settings)
    return document


@index_router.post("/{collection_id}/documents/index", response_model=IndexedDocument)
async def index_upload(
    collection_id: UUID,
    request: Request,
    file: Annotated[UploadFile, File(description="PDF, TXT, or Markdown document")],
) -> IndexedDocument:
    """Index one document locally into a collection; development only."""
    settings: Settings = request.app.state.settings
    if settings.database_url is None:
        await file.close()
        raise HTTPException(status_code=503, detail={"code": "database_unconfigured"})
    document, content = await _read_and_parse_uploaded_file(file, settings)
    try:
        chunked = chunk_document(
            document, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
            max_chunks_per_document=settings.max_chunks_per_document,
        )
        provider = OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url, model_name=settings.embedding_model,
            max_dimensions=settings.max_embedding_dimensions,
        )
        try:
            return await run_in_threadpool(
                index_document, collection_id=collection_id, document=document, original_content=content,
                chunked=chunked, provider=provider,
                store=PostgresIndexStore(settings.database_url.get_secret_value()),
                max_index_chunks=settings.max_index_chunks_per_document,
                max_embedding_elements=settings.max_embedding_elements,
            )
        finally:
            provider.close()
    except ChunkingError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc
    except IndexingError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail={"code": "embedding_unavailable"}) from exc
    except ModelDimensionMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_dimension_mismatch"}) from exc
    except ModelRevisionMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_revision_mismatch"}) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail={"code": "database_unavailable"}) from exc


async def _read_and_parse_uploaded_file(
    file: UploadFile, settings: Settings,
) -> tuple[ParsedDocument, bytes]:
    """Read bounded bytes once and apply the Phase 2 parser."""
    try:
        validate_upload_metadata(file.filename, file.content_type)
        content = await file.read(settings.max_upload_size_bytes + 1)
        document = parse_document(
            filename=file.filename,
            content_type=file.content_type,
            content=content,
            max_size_bytes=settings.max_upload_size_bytes,
            max_pdf_pages=settings.max_pdf_pages,
            max_extracted_characters=settings.max_extracted_characters,
        )
        return document, content
    except DocumentParseError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)}) from exc
    finally:
        await file.close()
