"""Bounded Gemini Developer API generation for explicitly approved collections."""

import re
from time import monotonic
from typing import Literal

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError

from rag_ai.generation.base import GeneratedAnswer, GenerationConfig, GenerationError

GEMINI_API_ORIGIN = "https://generativelanguage.googleapis.com"
SUPPORTED_GEMINI_MODELS = frozenset({"gemini-3.5-flash-lite"})

# Use only the JSON Schema subset needed by this answer contract. Application
# validation remains authoritative even when the provider accepts this schema.
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


class _TextPart(BaseModel):
    model_config = {"extra": "forbid"}

    text: str


class _ModelContent(BaseModel):
    role: Literal["model"]
    parts: list[_TextPart] = Field(min_length=1, max_length=1)


class _Candidate(BaseModel):
    finishReason: Literal["STOP"]
    content: _ModelContent


class _PromptFeedback(BaseModel):
    blockReason: Literal["BLOCK_REASON_UNSPECIFIED"] | None = None


class _GenerateContentResponse(BaseModel):
    candidates: list[_Candidate] = Field(min_length=1, max_length=1)
    modelVersion: str
    promptFeedback: _PromptFeedback | None = None


def _compatible_model_version(requested: str, returned: str) -> bool:
    """Accept the alias or a numeric version suffix of the same model family."""
    return returned == requested or bool(re.fullmatch(
        re.escape(requested) + r"-(?:[0-9]{3}|[0-9]{4}-[0-9]{2}-[0-9]{2})",
        returned,
    ))


class GeminiGenerationProvider:
    """One allowlisted model on Google's fixed HTTPS Developer API host."""

    def __init__(
        self, *, api_key: SecretStr, model_name: str = "gemini-3.5-flash-lite",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if model_name not in SUPPORTED_GEMINI_MODELS:
            raise GenerationError("Cloud generation model is not supported.")
        key = api_key.get_secret_value().strip()
        if not key or "\n" in key or "\r" in key:
            raise GenerationError("Cloud generation credential is unavailable.")
        self.model_name = model_name
        self._client = httpx.Client(
            base_url=GEMINI_API_ORIGIN,
            timeout=httpx.Timeout(60.0, connect=5.0),
            verify=True, trust_env=False, follow_redirects=False,
            transport=transport, headers={"x-goog-api-key": key},
        )

    def close(self) -> None:
        self._client.close()

    def generate(
        self, *, system_prompt: str, user_prompt: str, config: GenerationConfig,
    ) -> GeneratedAnswer:
        """Request one non-streaming text candidate; never retry or fall back."""
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "candidateCount": 1,
                "maxOutputTokens": config.num_predict,
                "responseFormat": {"text": {
                    "mimeType": "APPLICATION_JSON", "schema": _ANSWER_SCHEMA,
                }},
            },
            "store": False,
        }
        deadline = monotonic() + 90.0
        try:
            with self._client.stream(
                "POST", f"/v1beta/models/{self.model_name}:generateContent", json=payload,
            ) as response:
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
            envelope = _GenerateContentResponse.model_validate_json(body)
            if not _compatible_model_version(self.model_name, envelope.modelVersion):
                raise GenerationError("Cloud generation returned an unexpected model.")
            text = envelope.candidates[0].content.parts[0].text
            if not text.strip():
                raise GenerationError("Cloud generation returned no text.")
            return GeneratedAnswer.model_validate_json(text)
        except GenerationError:
            raise
        except (httpx.HTTPError, ValidationError, ValueError):
            raise GenerationError("Cloud generation failed or returned invalid data.") from None
