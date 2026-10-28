"""Deterministic, page-aware splitting of parsed documents."""

import json
from hashlib import sha256

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from rag_ai.documents.parser import FileType, ParsedDocument


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    file_type: FileType
    page_number: int | None
    chunk_index: int
    text: str


class ChunkedDocument(BaseModel):
    document_id: str
    filename: str
    file_type: FileType
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    chunks: list[DocumentChunk]


class ChunkingError(ValueError):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _chunk_id(
    document_id: str, page_number: int | None, chunk_index: int, text: str,
    chunk_size: int, chunk_overlap: int,
) -> str:
    canonical = json.dumps(
        {
            "version": 1,
            "document_id": document_id,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "text": text,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def chunk_document(
    document: ParsedDocument, *, chunk_size: int = 1000, chunk_overlap: int = 150,
    max_chunks_per_document: int = 5000,
) -> ChunkedDocument:
    """Split a parsed document while preserving its source page metadata."""
    if not 64 <= chunk_size <= 10_000 or not 0 <= chunk_overlap <= chunk_size // 2:
        raise ChunkingError("invalid_chunk_configuration", "Invalid chunk size or overlap.", 422)
    if not 1 <= max_chunks_per_document <= 10_000:
        raise ChunkingError("invalid_chunk_configuration", "Invalid chunk count limit.", 422)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunks: list[DocumentChunk] = []
    if document.file_type == "pdf":
        sources = [
            (page.page_number, page.text)
            for page in sorted(document.extraction_metadata.pages, key=lambda page: page.page_number)
        ]
    else:
        sources = [(None, document.text)]

    for page_number, source_text in sources:
        if not source_text.strip():
            continue
        remaining_chunks = max_chunks_per_document - len(chunks)
        # The splitter can discard whitespace, but it cannot fit more than
        # chunk_size non-whitespace characters in one chunk. Reject inputs
        # that must exceed the remaining budget before materializing a list.
        non_whitespace_count = sum(not char.isspace() for char in source_text)
        if non_whitespace_count > remaining_chunks * chunk_size:
            raise ChunkingError("too_many_chunks", "The document exceeds the chunk count limit.", 413)
        for chunk_text in splitter.split_text(source_text):
            if len(chunks) >= max_chunks_per_document:
                raise ChunkingError("too_many_chunks", "The document exceeds the chunk count limit.", 413)
            chunk_index = len(chunks)
            chunks.append(
                DocumentChunk(
                    chunk_id=_chunk_id(
                        document.document_id, page_number, chunk_index, chunk_text,
                        chunk_size, chunk_overlap,
                    ),
                    document_id=document.document_id,
                    filename=document.filename,
                    file_type=document.file_type,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

    return ChunkedDocument(
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        total_chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=chunks,
    )
