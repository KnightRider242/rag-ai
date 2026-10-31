"""Loopback-only Ollama JSON-schema generation with bounded responses."""

from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError

from rag_ai.core.config import validate_local_ollama_url
from rag_ai.generation.base import (
    GENERATION_OUTPUT_ADAPTER, GeneratedAnswer, GenerationConfig, GenerationError,
)


class _Message(BaseModel):
    role: Literal["assistant"]
    content: str


class _ChatResponse(BaseModel):
    model: str
    message: _Message
    done: bool
    done_reason: Literal["stop"]


class OllamaGenerationProvider:
    def __init__(
        self, *, base_url: str, model_name: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        try:
            validate_local_ollama_url(base_url)
        except ValueError as exc:
            raise GenerationError("Local Ollama endpoint configuration is invalid.") from exc
        if not model_name.strip() or len(model_name) > 200:
            raise GenerationError("Local generation model configuration is invalid.")
        self.model_name = model_name
        self._client = httpx.Client(
            base_url=base_url, timeout=httpx.Timeout(120.0, connect=5.0),
            trust_env=False, follow_redirects=False, transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def generate(
        self, *, system_prompt: str, user_prompt: str, config: GenerationConfig,
    ) -> GeneratedAnswer:
        """Request one non-streaming JSON-schema response from local Ollama."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": GENERATION_OUTPUT_ADAPTER.json_schema(),
            "stream": False,
            "options": {
                "num_ctx": config.num_ctx,
                "num_predict": config.num_predict,
                "temperature": config.temperature,
            },
        }
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                if 300 <= response.status_code < 400:
                    raise GenerationError("Local generation redirected unexpectedly.")
                response.raise_for_status()
                body = bytearray()
                for part in response.iter_bytes(chunk_size=8192):
                    if len(body) + len(part) > config.max_response_bytes:
                        raise GenerationError("Local generation response exceeded the configured limit.")
                    body.extend(part)
            envelope = _ChatResponse.model_validate_json(body)
            if envelope.model != self.model_name or not envelope.done or not envelope.message.content.strip():
                raise GenerationError("Local generation returned an invalid response.")
            structured = GENERATION_OUTPUT_ADAPTER.validate_json(envelope.message.content)
            return GeneratedAnswer.model_validate(structured.model_dump())
        except GenerationError:
            raise
        except (httpx.HTTPError, ValidationError, ValueError) as exc:
            raise GenerationError("Local generation failed or returned invalid data.") from exc
