"""Development-only combined text and visual retrieval."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from rag_ai.embeddings.base import EmbeddingError
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.retrieval.combined import CombinedRetrievalService
from rag_ai.retrieval.models import CombinedSearchResponse, SearchRequest
from rag_ai.retrieval.visual import VisualRetrievalError
from rag_ai.storage.combined_retrieval import PostgresCombinedRetrievalStore
from rag_ai.storage.postgres import (
    ModelDimensionMismatch, ModelNameMismatch, ModelRevisionMismatch, StorageError,
)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("/{collection_id}/multimodal-search", response_model=CombinedSearchResponse)
async def search_combined(collection_id: UUID, payload: SearchRequest,
                          request: Request) -> CombinedSearchResponse:
    settings = request.app.state.settings
    if settings.database_url is None:
        raise HTTPException(status_code=503, detail={"code": "database_unconfigured"})
    provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url, model_name=settings.embedding_model,
        max_dimensions=settings.max_embedding_dimensions,
    )
    try:
        service = CombinedRetrievalService(
            provider=provider,
            store=PostgresCombinedRetrievalStore(settings.database_url.get_secret_value()),
        )
        return await run_in_threadpool(
            service.search, collection_id=collection_id, query=payload.query,
            top_k=payload.top_k,
        )
    except (EmbeddingError, VisualRetrievalError) as exc:
        raise HTTPException(status_code=502, detail={"code": "embedding_unavailable"}) from exc
    except ModelNameMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_model_mismatch"}) from exc
    except ModelRevisionMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_revision_mismatch"}) from exc
    except ModelDimensionMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_dimension_mismatch"}) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail={"code": "database_unavailable"}) from exc
    finally:
        provider.close()
