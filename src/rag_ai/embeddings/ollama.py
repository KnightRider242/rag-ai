"""Local Ollama embedding provider with response validation."""

import math
import re

import httpx
from pydantic import BaseModel, ValidationError

from rag_ai.core.config import validate_local_ollama_url
from rag_ai.embeddings.base import EmbeddingBatch, EmbeddingError


class _OllamaResponse(BaseModel):
    model: str
    embeddings: list[list[float]]


class _OllamaModel(BaseModel):
    name: str
    digest: str


class _OllamaTags(BaseModel):
    models: list[_OllamaModel]


class OllamaEmbeddingProvider:
    def __init__(
        self, *, base_url: str, model_name: str, max_dimensions: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        try:
            validate_local_ollama_url(base_url)
        except ValueError as exc:
            raise EmbeddingError("Local Ollama endpoint configuration is invalid.") from exc
        self.model_name = model_name
        self.max_dimensions = max_dimensions
        self._client = httpx.Client(
            base_url=base_url, timeout=60.0, trust_env=False, follow_redirects=False,
            transport=transport,
        )
        self._model_revision: str | None = None

    def close(self) -> None:
        self._client.close()

    def _read_model_revision(self) -> str:
        try:
            with self._client.stream("GET", "/api/tags") as response:
                response.raise_for_status()
                body = bytearray()
                for part in response.iter_bytes():
                    if len(body) + len(part) > 1_000_000:
                        raise EmbeddingError("Local model list is too large.")
                    body.extend(part)
            tags = _OllamaTags.model_validate_json(body)
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            raise EmbeddingError("Local model identity could not be verified.") from exc
        matches = [model.digest for model in tags.models if model.name == self.model_name]
        if len(matches) != 1 or not re.fullmatch(r"[0-9a-f]{64}", matches[0]):
            raise EmbeddingError("Local model identity could not be verified.")
        return matches[0]

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        if not texts or any(not text.strip() for text in texts):
            raise EmbeddingError("Embedding input must contain nonempty text.")
        revision = self._read_model_revision()
        if self._model_revision is not None and revision != self._model_revision:
            raise EmbeddingError("Local model revision changed during indexing.")
        self._model_revision = revision
        try:
            with self._client.stream(
                "POST", "/api/embed", json={"model": self.model_name, "input": texts, "truncate": False}
            ) as response:
                response.raise_for_status()
                body = bytearray()
                max_response_bytes = self.max_dimensions * len(texts) * 32 + 4096
                for part in response.iter_bytes():
                    if len(body) + len(part) > max_response_bytes:
                        raise EmbeddingError("Local embedding service response is too large.")
                    body.extend(part)
            payload = _OllamaResponse.model_validate_json(body)
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            raise EmbeddingError("Local embedding service failed or returned invalid data.") from exc

        if payload.model != self.model_name or len(payload.embeddings) != len(texts):
            raise EmbeddingError("Local embedding service returned unexpected model or vector count.")
        dimensions = len(payload.embeddings[0])
        if not 1 <= dimensions <= self.max_dimensions:
            raise EmbeddingError("Local embedding service returned an unsupported vector size.")
        if any(len(vector) != dimensions or any(not math.isfinite(value) for value in vector)
               for vector in payload.embeddings):
            raise EmbeddingError("Local embedding service returned invalid vectors.")
        if self._read_model_revision() != revision:
            raise EmbeddingError("Local model revision changed during indexing.")
        return EmbeddingBatch(
            model_name=payload.model, model_revision=revision,
            dimensions=dimensions, vectors=payload.embeddings,
        )
