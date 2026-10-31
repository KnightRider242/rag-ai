"""Deterministic, bounded evidence selection for one answer request."""

import json
from dataclasses import dataclass

from pydantic import BaseModel, Field

from rag_ai.retrieval.models import RetrievedChunk


class ContextLimits(BaseModel):
    max_chunks: int = Field(default=5, ge=1, le=5)
    max_chunk_characters: int = Field(default=1200, ge=64, le=2000)
    max_total_characters: int = Field(default=4000, ge=256, le=8000)


class ContextLimitError(ValueError):
    """Retrieved evidence cannot fit the configured prompt budget."""


@dataclass(frozen=True)
class EvidenceContext:
    text: str
    sources: dict[str, RetrievedChunk]


def build_context(chunks: list[RetrievedChunk], limits: ContextLimits) -> EvidenceContext:
    """Keep ranked chunks intact; reject rather than silently truncate evidence."""
    if not 1 <= len(chunks) <= limits.max_chunks:
        raise ContextLimitError("Retrieved evidence count exceeds the configured limit.")
    sections: list[str] = []
    sources: dict[str, RetrievedChunk] = {}
    total = 0
    for index, chunk in enumerate(chunks, 1):
        if not chunk.text.strip() or len(chunk.text) > limits.max_chunk_characters:
            raise ContextLimitError("Retrieved evidence is empty or too large.")
        label = f"S{index}"
        page = str(chunk.page_number) if chunk.page_number is not None else "none"
        section = (
            f"[{label}]\nDocument: {json.dumps(chunk.filename, ensure_ascii=True)}\n"
            f"Page: {page}\nContent (quoted data): {json.dumps(chunk.text, ensure_ascii=True)}"
        )
        total += len(section) + (1 if sections else 0)
        if total > limits.max_total_characters:
            raise ContextLimitError("Retrieved evidence exceeds the configured context limit.")
        sections.append(section)
        sources[label] = chunk
    return EvidenceContext(text="\n".join(sections), sources=sources)
