"""Trusted-localhost development answer endpoint."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from rag_ai.core.config import Settings
from rag_ai.embeddings.base import EmbeddingError
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.generation.base import GenerationConfig, GenerationError
from rag_ai.generation.ollama import OllamaGenerationProvider
from rag_ai.rag.context import ContextLimitError, ContextLimits
from rag_ai.rag.models import AnswerRequest, AnswerResponse
from rag_ai.rag.service import CitationError, RAGService
from rag_ai.retrieval.service import RetrievalError, RetrievalService
from rag_ai.storage.postgres import (
    ModelDimensionMismatch, ModelNameMismatch, ModelRevisionMismatch,
    PostgresIndexStore, StorageError,
)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("/{collection_id}/answer", response_model=AnswerResponse)
async def answer_collection(
    collection_id: UUID, payload: AnswerRequest, request: Request,
) -> AnswerResponse:
    """Retrieve and answer locally; registered only for opted-in development."""
    settings: Settings = request.app.state.settings
    if settings.database_url is None:
        raise HTTPException(status_code=503, detail={"code": "database_unconfigured"})
    try:
        embedding = OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url, model_name=settings.embedding_model,
            max_dimensions=settings.max_embedding_dimensions,
        )
        try:
            generation = OllamaGenerationProvider(
                base_url=settings.ollama_base_url, model_name=settings.generation_model,
            )
            try:
                service = RAGService(
                    retriever=RetrievalService(
                        provider=embedding,
                        store=PostgresIndexStore(settings.database_url.get_secret_value()),
                    ),
                    generator=generation,
                    context_limits=ContextLimits(
                        max_chunks=settings.rag_max_chunks,
                        max_chunk_characters=settings.rag_max_chunk_characters,
                        max_total_characters=settings.rag_max_context_characters,
                    ),
                    generation_config=GenerationConfig(
                        num_ctx=settings.generation_num_ctx,
                        num_predict=settings.generation_num_predict,
                        temperature=settings.generation_temperature,
                        max_response_bytes=settings.generation_max_response_bytes,
                    ),
                )
                return await run_in_threadpool(
                    service.answer, collection_id=collection_id,
                    query=payload.query, top_k=payload.top_k,
                )
            finally:
                generation.close()
        finally:
            embedding.close()
    except (EmbeddingError, RetrievalError) as exc:
        raise HTTPException(status_code=502, detail={"code": "embedding_unavailable"}) from exc
    except (GenerationError, CitationError) as exc:
        raise HTTPException(status_code=502, detail={"code": "generation_invalid"}) from exc
    except ContextLimitError as exc:
        raise HTTPException(status_code=413, detail={"code": "evidence_too_large"}) from exc
    except ModelNameMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_model_mismatch"}) from exc
    except ModelRevisionMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_revision_mismatch"}) from exc
    except ModelDimensionMismatch as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_dimension_mismatch"}) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail={"code": "database_unavailable"}) from exc
