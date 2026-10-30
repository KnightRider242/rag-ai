"""Thin LangChain Core adapter for the retrieval service."""

from uuid import UUID

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from rag_ai.retrieval.service import RetrievalService


class RAGAIBaseRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    service: RetrievalService
    collection_id: UUID
    top_k: int = Field(default=5, ge=1, le=20)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        response = self.service.retrieve(collection_id=self.collection_id, query=query, top_k=self.top_k)
        return [
            Document(
                page_content=chunk.text,
                metadata={
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "collection_id": str(self.collection_id),
                    "filename": chunk.filename,
                    "file_type": chunk.file_type,
                    "page_number": chunk.page_number,
                    "distance": chunk.cosine_distance,
                },
            )
            for chunk in response.results
        ]
