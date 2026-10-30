"""Trusted-localhost development search endpoint."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from rag_ai.core.config import Settings
from rag_ai.embeddings.base import EmbeddingError
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.retrieval.models import SearchRequest, SearchResponse
from rag_ai.retrieval.service import RetrievalError, RetrievalService
from rag_ai.storage.postgres import (
    ModelDimensionMismatch, ModelNameMismatch, ModelRevisionMismatch,
    PostgresIndexStore, StorageError,
)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("/{collection_id}/search", response_model=SearchResponse)
async def search_collection(
    collection_id: UUID, payload: SearchRequest, request: Request,
) -> SearchResponse:
    """Embed a question locally and retrieve ranked chunks; development only."""
    settings: Settings = request.app.state.settings
    if settings.database_url is None:
        raise HTTPException(status_code=503, detail={"code": "database_unconfigured"})
    try:
        provider = OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url, model_name=settings.embedding_model,
            max_dimensions=settings.max_embedding_dimensions,
        )
        try:
            service = RetrievalService(
                provider=provider, store=PostgresIndexStore(settings.database_url.get_secret_value()),
            )
            return await run_in_threadpool(
                service.retrieve, collection_id=collection_id,
                query=payload.query, top_k=payload.top_k,
            )
        finally:
            provider.close()
    except (EmbeddingError, RetrievalError) as exc:
        raise HTTPException(status_code=502, detail={"code": "embedding_unavailable"}) from exc
    except ModelNameMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_model_mismatch"}) from exc
    except ModelRevisionMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_revision_mismatch"}) from exc
    except ModelDimensionMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_dimension_mismatch"}) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail={"code": "database_unavailable"}) from exc
