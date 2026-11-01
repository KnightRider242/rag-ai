"""Bounded Anthropic Messages generation for explicitly approved collections."""

from time import monotonic
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from rag_ai.generation.base import GeneratedAnswer, GenerationConfig, GenerationError

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2026-10-21"
SUPPORTED_ANTHROPIC_MODELS = frozenset({"claude-haiku-5-5"})

# Keep the wire schema within the provider's supported subset. The domain
# model below still enforces claim counts, text lengths, and status consistency.
_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "insufficient_evidence"]},
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text", "source_ids"],
            "additionalProperties": False,
        }},
    },
    "required": ["status", "claims"],
    "additionalProperties": False,
}


class _TextBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str
    citations: None = None


class _MessageEnvelope(BaseModel):
    type: Literal["message"]
    role: Literal["assistant"]
    model: Literal["claude-haiku-5-5"]
    stop_reason: Literal["end_turn"]
    stop_details: None = None
    content: list[dict[str, object]] = Field(min_length=1)


class AnthropicGenerationProvider:
    """One allowlisted Claude model on a fixed HTTPS Messages endpoint."""

    def __init__(
        self, *, api_key: SecretStr, model_name: str = "claude-haiku-5-5",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if model_name not in SUPPORTED_ANTHROPIC_MODELS:
            raise GenerationError("Cloud generation model is not supported.")
        key = api_key.get_secret_value().strip()
        if not key or "\n" in key or "\r" in key:
            raise GenerationError("Cloud generation credential is unavailable.")
        self.model_name = model_name
        self._client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=5.0),
            verify=True, trust_env=False, follow_redirects=False, transport=transport,
            headers={
                "x-api-key": key,
                "anthropic-version": ANTHROPIC_API_VERSION,
                "content-type": "application/json",
            },
        )

    def close(self) -> None:
        self._client.close()

    def generate(
        self, *, system_prompt: str, user_prompt: str, config: GenerationConfig,
    ) -> GeneratedAnswer:
        """Accept one completed text answer; ignore thinking without exposing it."""
        payload = {
            "model": self.model_name,
            "max_tokens": config.num_predict,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "output_config": {"format": {
                "type": "json_schema", "schema": _ANSWER_SCHEMA,
            }},
            "stream": False,
        }
        deadline = monotonic() + 90.0
        try:
            with self._client.stream("POST", ANTHROPIC_MESSAGES_URL, json=payload) as response:
                if 300 <= response.status_code < 400:
                    raise GenerationError("Cloud generation redirected unexpectedly.")
                if response.status_code in {401, 403}:
                    raise GenerationError("Cloud generation authentication failed.")
                if response.status_code == 429:
                    raise GenerationError("Cloud generation is rate limited.")
                response.raise_for_status()
                body = bytearray()
                for part in response.iter_bytes():
                    if monotonic() > deadline:
                        raise GenerationError("Cloud generation timed out.")
                    if len(body) + len(part) > config.max_response_bytes:
                        raise GenerationError("Cloud generation response exceeded the configured limit.")
                    body.extend(part)
            envelope = _MessageEnvelope.model_validate_json(body)
            if envelope.model != self.model_name:
                raise GenerationError("Cloud generation returned an unexpected model.")
            answers: list[str] = []
            for block in envelope.content:
                if block.get("type") in {"thinking", "redacted_thinking"}:
                    continue
                if block.get("type") != "text":
                    raise GenerationError("Cloud generation returned unexpected content.")
                answers.append(_TextBlock.model_validate(block).text)
            if len(answers) != 1 or not answers[0].strip():
                raise GenerationError("Cloud generation returned no unambiguous text answer.")
            return GeneratedAnswer.model_validate_json(answers[0])
        except GenerationError:
            raise
        except (httpx.HTTPError, ValidationError, ValueError):
            raise GenerationError("Cloud generation failed or returned invalid data.") from None
