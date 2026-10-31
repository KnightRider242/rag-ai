"""Single-turn retrieval, local generation, and citation resolution."""

from typing import Protocol
from uuid import UUID

from rag_ai.generation.base import GenerationConfig, GenerationProvider
from rag_ai.rag.context import ContextLimits, build_context
from rag_ai.rag.models import AnswerClaim, AnswerRequest, AnswerResponse, Citation
from rag_ai.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from rag_ai.retrieval.models import SearchResponse


class Retriever(Protocol):
    def retrieve(self, *, collection_id: UUID, query: str, top_k: int) -> SearchResponse: ...


class CitationError(ValueError):
    """Generated citation labels do not match the supplied evidence."""


class RAGService:
    def __init__(
        self, *, retriever: Retriever, generator: GenerationProvider,
        context_limits: ContextLimits, generation_config: GenerationConfig,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.context_limits = context_limits
        self.generation_config = generation_config

    def answer(self, *, collection_id: UUID, query: str, top_k: int = 3) -> AnswerResponse:
        """Resolve only model-cited labels that the application assigned this request."""
        request = AnswerRequest(query=query, top_k=top_k)
        retrieved = self.retriever.retrieve(
            collection_id=collection_id, query=request.query, top_k=request.top_k,
        )
        if retrieved.collection_id != collection_id:
            raise CitationError("Retrieved collection does not match the request.")
        if not retrieved.results:
            return AnswerResponse(collection_id=collection_id, status="insufficient_evidence", claims=[])
        context = build_context(retrieved.results, self.context_limits)
        generated = self.generator.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(request.query, context.text),
            config=self.generation_config,
        )
        if generated.status == "insufficient_evidence":
            return AnswerResponse(collection_id=collection_id, status="insufficient_evidence", claims=[])
        claims: list[AnswerClaim] = []
        for claim in generated.claims:
            citations: list[Citation] = []
            for label in dict.fromkeys(claim.source_ids):
                chunk = context.sources.get(label)
                if chunk is None:
                    raise CitationError("Generated citation does not match the supplied evidence.")
                citations.append(Citation(
                    source_id=label, document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id, filename=chunk.filename,
                    file_type=chunk.file_type, page_number=chunk.page_number,
                ))
            claims.append(AnswerClaim(text=claim.text, citations=citations))
        return AnswerResponse(collection_id=collection_id, status="answered", claims=claims)
