"""Development-only persistence routes for indexed PDF source regions."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Path, Query, Request
from starlette.concurrency import run_in_threadpool

from rag_ai.documents.parser import DocumentParseError
from rag_ai.documents.visual_elements import StoredVisualElement, VisualIndexResult, extract_visual_elements
from rag_ai.storage.visual import (
    PostgresVisualStore, VisualDocumentNotFound, VisualSourceInvalid, VisualStoreError,
)

router = APIRouter(prefix="/collections/{collection_id}/documents/{document_id}/visual-elements",
                   tags=["collections"])


def _store(request: Request) -> PostgresVisualStore:
    url = request.app.state.settings.database_url
    if url is None:
        raise HTTPException(status_code=503, detail={"code": "database_unconfigured"})
    return PostgresVisualStore(url.get_secret_value())


def _index(store: PostgresVisualStore, *, collection_id: UUID, document_id: str,
           max_size_bytes: int, max_pdf_pages: int) -> VisualIndexResult:
    filename, content, page_count = store.load_pdf(
        collection_id=collection_id, document_id=document_id, max_size_bytes=max_size_bytes,
    )
    if page_count > max_pdf_pages:
        raise DocumentParseError("too_many_pages", "The stored PDF exceeds the page limit.", 413)
    candidates = extract_visual_elements(filename=filename, content=content,
                                         max_size_bytes=max_size_bytes, max_pdf_pages=max_pdf_pages,
                                         expected_page_count=page_count)
    return store.replace(collection_id=collection_id, document_id=document_id,
                         source_sha256=document_id, page_count=page_count, candidates=candidates)


def _raise_http(exc: Exception) -> HTTPException:
    if isinstance(exc, VisualDocumentNotFound):
        return HTTPException(status_code=404, detail={"code": "document_not_found"})
    if isinstance(exc, VisualSourceInvalid):
        return HTTPException(status_code=409, detail={"code": "invalid_stored_source"})
    if isinstance(exc, DocumentParseError):
        return HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": str(exc)})
    return HTTPException(status_code=503, detail={"code": "visual_storage_unavailable"})


@router.post("/index", response_model=VisualIndexResult)
async def index_visual_elements(collection_id: UUID,
                                document_id: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")],
                                request: Request) -> VisualIndexResult:
    """Extract a complete bounded set from already persisted PDF bytes."""
    settings = request.app.state.settings
    try:
        return await run_in_threadpool(
            _index, _store(request), collection_id=collection_id, document_id=document_id,
            max_size_bytes=settings.max_upload_size_bytes, max_pdf_pages=settings.max_pdf_pages,
        )
    except (VisualDocumentNotFound, VisualSourceInvalid, VisualStoreError, DocumentParseError) as exc:
        raise _raise_http(exc) from exc


@router.get("", response_model=list[StoredVisualElement])
async def list_visual_elements(collection_id: UUID,
                               document_id: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")],
                               request: Request,
                               limit: Annotated[int, Query(ge=1, le=100)] = 100,
                               offset: Annotated[int, Query(ge=0)] = 0) -> list[StoredVisualElement]:
    """List only persisted provenance and bounded extraction metadata."""
    try:
        return await run_in_threadpool(_store(request).list_elements,
                                       collection_id=collection_id, document_id=document_id,
                                       limit=limit, offset=offset)
    except (VisualDocumentNotFound, VisualStoreError) as exc:
        raise _raise_http(exc) from exc
