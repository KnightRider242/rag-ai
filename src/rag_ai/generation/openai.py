"""Explicit, bounded OpenAI Responses API generation for approved collections."""

from time import monotonic
from typing import Literal

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError

from rag_ai.generation.base import GeneratedAnswer, GenerationConfig, GenerationError

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
SUPPORTED_OPENAI_MODELS = frozenset({"gpt-6-luna"})

# OpenAI strict Structured Outputs require an object root and required properties.
# Pydantic applies the domain's claim/text/count limits after the response arrives.
_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient_evidence"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "source_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "claims"],
    "additionalProperties": False,
}


class _OutputText(BaseModel):
    type: Literal["output_text"]
    text: str


class _OutputMessage(BaseModel):
    type: Literal["message"]
    status: Literal["completed"]
    role: Literal["assistant"]
    content: list[_OutputText] = Field(min_length=1, max_length=1)


class _ResponseEnvelope(BaseModel):
    object: Literal["response"]
    status: Literal["completed"]
    model: str
    error: None = None
    incomplete_details: None = None
    output: list[_OutputMessage] = Field(min_length=1, max_length=1)


class OpenAIGenerationProvider:
    """One fixed HTTPS endpoint; construction never sends a request."""

    def __init__(
        self, *, api_key: SecretStr, model_name: str = "gpt-6-luna",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if model_name not in SUPPORTED_OPENAI_MODELS:
            raise GenerationError("Cloud generation model is not supported.")
        key = api_key.get_secret_value().strip()
        if not key or "\n" in key or "\r" in key:
            raise GenerationError("Cloud generation credential is unavailable.")
        self.model_name = model_name
        self._client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=5.0), verify=True,
            trust_env=False, follow_redirects=False, transport=transport,
            headers={"Authorization": f"Bearer {key}"},
        )

    def close(self) -> None:
        self._client.close()

    def generate(
        self, *, system_prompt: str, user_prompt: str, config: GenerationConfig,
    ) -> GeneratedAnswer:
        """Send only bounded evidence text; never retry or select another provider."""
        payload = {
            "model": self.model_name,
            "input": [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {"format": {
                "type": "json_schema", "name": "rag_answer", "strict": True,
                "schema": _ANSWER_SCHEMA,
            }},
            "reasoning": {"effort": "none"},
            "max_output_tokens": config.num_predict,
            "store": False,
            "stream": False,
            "tools": [],
            "tool_choice": "none",
        }
        deadline = monotonic() + 90.0
        try:
            with self._client.stream("POST", OPENAI_RESPONSES_URL, json=payload) as response:
                if 300 <= response.status_code < 400:
                    raise GenerationError("Cloud generation redirected unexpectedly.")
                if response.status_code in {401, 403}:
                    raise GenerationError("Cloud generation authentication failed.")
                if response.status_code == 429:
                    raise GenerationError("Cloud generation is rate limited.")
                response.raise_for_status()
                body = bytearray()
                for part in response.iter_bytes(chunk_size=8192):
                    if monotonic() > deadline:
                        raise GenerationError("Cloud generation timed out.")
                    if len(body) + len(part) > config.max_response_bytes:
                        raise GenerationError("Cloud generation response exceeded the configured limit.")
                    body.extend(part)
            envelope = _ResponseEnvelope.model_validate_json(body)
            if envelope.model != self.model_name:
                raise GenerationError("Cloud generation returned an unexpected model.")
            text = envelope.output[0].content[0].text
            if not text.strip():
                raise GenerationError("Cloud generation returned no text.")
            return GeneratedAnswer.model_validate_json(text)
        except GenerationError:
            raise
        except (httpx.HTTPError, ValidationError, ValueError):
            raise GenerationError("Cloud generation failed or returned invalid data.") from None
