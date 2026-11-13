"""Development-only locally generated multimodal answers."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from rag_ai.embeddings.base import EmbeddingError
from rag_ai.embeddings.ollama import OllamaEmbeddingProvider
from rag_ai.generation.base import GenerationConfig, GenerationError
from rag_ai.generation.ollama import OllamaGenerationProvider
from rag_ai.rag.context import ContextLimitError, ContextLimits
from rag_ai.rag.models import AnswerRequest
from rag_ai.rag.multimodal import CitationError, MultimodalAnswerResponse, MultimodalRAGService
from rag_ai.retrieval.combined import CombinedRetrievalService
from rag_ai.retrieval.visual import VisualRetrievalError
from rag_ai.storage.answer_sources import PostgresAnswerSourceResolver
from rag_ai.storage.combined_retrieval import PostgresCombinedRetrievalStore
from rag_ai.storage.postgres import (
    ModelDimensionMismatch, ModelNameMismatch, ModelRevisionMismatch, StorageError,
)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("/{collection_id}/multimodal-answer", response_model=MultimodalAnswerResponse)
async def answer_multimodal(collection_id: UUID, payload: AnswerRequest,
                            request: Request) -> MultimodalAnswerResponse:
    settings = request.app.state.settings
    if settings.database_url is None:
        raise HTTPException(status_code=503, detail={"code": "database_unconfigured"})
    embedding = OllamaEmbeddingProvider(base_url=settings.ollama_base_url,
        model_name=settings.embedding_model, max_dimensions=settings.max_embedding_dimensions)
    try:
        generation = OllamaGenerationProvider(base_url=settings.ollama_base_url,
            model_name=settings.generation_model)
        try:
            service = MultimodalRAGService(
                retriever=CombinedRetrievalService(provider=embedding,
                    store=PostgresCombinedRetrievalStore(settings.database_url.get_secret_value())),
                generator=generation,
                resolver=PostgresAnswerSourceResolver(settings.database_url.get_secret_value()),
                context_limits=ContextLimits(max_chunks=settings.rag_max_chunks,
                    max_chunk_characters=settings.rag_max_chunk_characters,
                    max_total_characters=settings.rag_max_context_characters),
                generation_config=GenerationConfig(num_ctx=settings.generation_num_ctx,
                    num_predict=settings.generation_num_predict,
                    temperature=settings.generation_temperature,
                    max_response_bytes=settings.generation_max_response_bytes),
            )
            return await run_in_threadpool(service.answer, collection_id=collection_id,
                                           query=payload.query, top_k=payload.top_k)
        finally:
            generation.close()
    except (EmbeddingError, VisualRetrievalError) as exc:
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
    finally:
        embedding.close()
