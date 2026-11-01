"""Development-only local visual interpretation indexing and search."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from rag_ai.embeddings.base import EmbeddingError
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.retrieval.models import SearchRequest
from rag_ai.retrieval.visual import (
    VisualIndexResult, VisualRetrievalError, VisualRetrievalService, VisualSearchResponse,
)
from rag_ai.storage.postgres import (
    ModelDimensionMismatch, ModelNameMismatch, ModelRevisionMismatch, StorageError,
)
from rag_ai.storage.visual_retrieval import (
    PostgresVisualRetrievalStore, VisualIndexSourceChanged, VisualInterpretationNotFound,
)

router = APIRouter(prefix="/collections/{collection_id}", tags=["collections"])


class IndexVisualRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    element_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    interpretation_id: UUID


class VisualSearchRequest(SearchRequest):
    model_config = ConfigDict(extra="forbid", strict=True)


def _service(request: Request) -> tuple[VisualRetrievalService, OllamaEmbeddingProvider]:
    settings = request.app.state.settings
    if settings.database_url is None:
        raise HTTPException(status_code=503, detail={"code": "database_unconfigured"})
    provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url, model_name=settings.embedding_model,
        max_dimensions=settings.max_embedding_dimensions,
    )
    return (VisualRetrievalService(
        provider=provider,
        store=PostgresVisualRetrievalStore(settings.database_url.get_secret_value()),
    ), provider)


def _error(exc: Exception) -> HTTPException:
    if isinstance(exc, VisualInterpretationNotFound):
        return HTTPException(status_code=404, detail={"code": "visual_interpretation_not_found"})
    if isinstance(exc, VisualIndexSourceChanged):
        return HTTPException(status_code=409, detail={"code": "visual_source_changed"})
    if isinstance(exc, ModelNameMismatch):
        return HTTPException(status_code=409, detail={"code": "embedding_model_mismatch"})
    if isinstance(exc, ModelRevisionMismatch):
        return HTTPException(status_code=409, detail={"code": "embedding_revision_mismatch"})
    if isinstance(exc, ModelDimensionMismatch):
        return HTTPException(status_code=409, detail={"code": "embedding_dimension_mismatch"})
    if isinstance(exc, (EmbeddingError, VisualRetrievalError)):
        return HTTPException(status_code=502, detail={"code": "embedding_unavailable"})
    return HTTPException(status_code=503, detail={"code": "database_unavailable"})


@router.post("/visual-interpretations/index", response_model=VisualIndexResult)
async def index_visual_interpretation(collection_id: UUID, payload: IndexVisualRequest,
                                      request: Request) -> VisualIndexResult:
    """Index one existing interpretation; never run visual inference."""
    service, provider = _service(request)
    try:
        return await run_in_threadpool(
            service.index, collection_id=collection_id, document_id=payload.document_id,
            element_id=payload.element_id, interpretation_id=payload.interpretation_id,
        )
    except (VisualInterpretationNotFound, VisualIndexSourceChanged, ModelNameMismatch,
            ModelRevisionMismatch, ModelDimensionMismatch, EmbeddingError,
            VisualRetrievalError, StorageError) as exc:
        raise _error(exc) from exc
    finally:
        provider.close()


@router.post("/visual-search", response_model=VisualSearchResponse)
async def search_visual_interpretations(collection_id: UUID, payload: VisualSearchRequest,
                                        request: Request) -> VisualSearchResponse:
    """Search only compatible visual interpretation embeddings in one collection."""
    service, provider = _service(request)
    try:
        return await run_in_threadpool(
            service.search, collection_id=collection_id, query=payload.query,
            top_k=payload.top_k,
        )
    except (ModelNameMismatch, ModelRevisionMismatch, ModelDimensionMismatch,
            EmbeddingError, VisualRetrievalError, StorageError) as exc:
        raise _error(exc) from exc
    finally:
        provider.close()
