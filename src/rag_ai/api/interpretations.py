"""Separately gated development routes for persisted local visual interpretations."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Path, Query, Request
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from rag_ai.api.vision import VisionTask, _http_error
from rag_ai.documents.parser import DocumentParseError
from rag_ai.storage.interpretations import (
    InterpretationConflict, InterpretationNotFound, PostgresInterpretationStore,
    StoredInterpretation,
)
from rag_ai.storage.visual import VisualDocumentNotFound, VisualSourceInvalid, VisualStoreError
from rag_ai.vision.base import VisionError
from rag_ai.vision.persistence import analyze_stored_element

router = APIRouter(prefix="/collections/{collection_id}/documents/{document_id}/visual-elements/{element_id}/interpretations",
                   tags=["collections"])
DocumentId = Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")]
ElementId = Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")]


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: VisionTask
    request_key: UUID | None = None


def _store(request: Request) -> PostgresInterpretationStore:
    url = request.app.state.settings.database_url
    if url is None:
        raise HTTPException(status_code=503, detail={"code": "database_unconfigured"})
    return PostgresInterpretationStore(url.get_secret_value())


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, (VisionError, DocumentParseError)):
        return _http_error(exc)
    if isinstance(exc, (InterpretationNotFound, VisualDocumentNotFound)):
        return HTTPException(status_code=404, detail={"code": "visual_source_not_found"})
    if isinstance(exc, (InterpretationConflict, VisualSourceInvalid)):
        return HTTPException(status_code=409, detail={"code": "visual_source_changed"})
    return HTTPException(status_code=503, detail={"code": "visual_storage_unavailable"})


@router.post("", response_model=StoredInterpretation)
async def analyze_interpretation(collection_id: UUID, document_id: DocumentId,
                                 element_id: ElementId, body: AnalyzeRequest,
                                 request: Request) -> StoredInterpretation:
    try:
        return await run_in_threadpool(
            analyze_stored_element, settings=request.app.state.settings, store=_store(request),
            collection_id=collection_id, document_id=document_id, element_id=element_id,
            task=body.task, request_key=body.request_key,
        )
    except (InterpretationConflict, InterpretationNotFound, VisualDocumentNotFound,
            VisualSourceInvalid, VisualStoreError, VisionError, DocumentParseError) as exc:
        raise _error(exc) from exc


@router.get("", response_model=list[StoredInterpretation])
async def list_interpretations(collection_id: UUID, document_id: DocumentId,
                               element_id: ElementId, request: Request,
                               limit: Annotated[int, Query(ge=1, le=50)] = 50,
                               offset: Annotated[int, Query(ge=0)] = 0) -> list[StoredInterpretation]:
    try:
        return await run_in_threadpool(_store(request).list, collection_id=collection_id,
                                       document_id=document_id, element_id=element_id,
                                       limit=limit, offset=offset)
    except (InterpretationNotFound, VisualStoreError) as exc:
        raise _error(exc) from exc


@router.get("/{interpretation_id}", response_model=StoredInterpretation)
async def get_interpretation(collection_id: UUID, document_id: DocumentId,
                             element_id: ElementId, interpretation_id: UUID,
                             request: Request) -> StoredInterpretation:
    try:
        result = await run_in_threadpool(_store(request).get, collection_id=collection_id,
                                         document_id=document_id, element_id=element_id,
                                         interpretation_id=interpretation_id)
        if result is None:
            raise InterpretationNotFound("Interpretation was not found.")
        return result
    except (InterpretationNotFound, VisualStoreError) as exc:
        raise _error(exc) from exc
