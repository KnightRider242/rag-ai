"""Gemini generateContent contract and failure tests using synthetic HTTP responses."""

import json
import logging
from copy import deepcopy

import httpx
import pytest
from pydantic import SecretStr

from rag_ai.generation.base import GenerationConfig, GenerationError
from rag_ai.generation.gemini import GeminiGenerationProvider

ANSWERED = '{"status":"answered","claims":[{"text":"The relay disconnects power.","source_ids":["S1"]}]}'
ABSTAINED = '{"status":"insufficient_evidence","claims":[]}'


def _envelope(text: str = ANSWERED) -> dict:
    return {
        "modelVersion": "gemini-3.5-flash-lite",
        "candidates": [{
            "finishReason": "STOP",
            "content": {"role": "model", "parts": [{"text": text}]},
        }],
    }


def _provider(handler) -> GeminiGenerationProvider:
    return GeminiGenerationProvider(
        api_key=SecretStr("synthetic-test-key"), model_name="gemini-3.5-flash-lite",
        transport=httpx.MockTransport(handler),
    )


def _generate(provider: GeminiGenerationProvider):
    return provider.generate(
        system_prompt="trusted synthetic instructions",
        user_prompt="private synthetic evidence",
        config=GenerationConfig(),
    )


def test_gemini_request_contract_answer_and_cleanup() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-3.5-flash-lite:generateContent"
        )
        assert request.headers["x-goog-api-key"] == "synthetic-test-key"
        assert "synthetic-test-key" not in str(request.url)
        payload = json.loads(request.content)
        assert payload["systemInstruction"] == {"parts": [{"text": "trusted synthetic instructions"}]}
        assert payload["contents"] == [{"role": "user", "parts": [{"text": "private synthetic evidence"}]}]
        assert payload["store"] is False
        assert "tools" not in payload and "toolConfig" not in payload
        assert "temperature" not in payload["generationConfig"]
        assert "thinkingConfig" not in payload["generationConfig"]
        assert "num_ctx" not in payload["generationConfig"]
        assert payload["generationConfig"]["candidateCount"] == 1
        assert payload["generationConfig"]["maxOutputTokens"] == 512
        fmt = payload["generationConfig"]["responseFormat"]["text"]
        assert fmt["mimeType"] == "APPLICATION_JSON"
        assert fmt["schema"]["type"] == "object"
        assert set(fmt["schema"]["required"]) == {"status", "claims"}
        return httpx.Response(200, json=_envelope())

    provider = _provider(handle)
    try:
        result = _generate(provider)
        assert result.status == "answered" and result.claims[0].source_ids == ["S1"]
    finally:
        provider.close()
    assert provider._client.is_closed


def test_gemini_abstention() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_envelope(ABSTAINED)))
    try:
        assert _generate(provider).status == "insufficient_evidence"
    finally:
        provider.close()


@pytest.mark.parametrize("version", [
    "gemini-3.5-flash-lite-001", "gemini-3.5-flash-lite-2026-07-01",
])
def test_supported_version_suffix(version: str) -> None:
    envelope = _envelope()
    envelope["modelVersion"] = version
    provider = _provider(lambda request: httpx.Response(200, json=envelope))
    try:
        assert _generate(provider).status == "answered"
    finally:
        provider.close()


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(candidates=[]),
    lambda x: x["candidates"].append(deepcopy(x["candidates"][0])),
    lambda x: x.update(promptFeedback={"blockReason": "SAFETY"}),
    lambda x: x["candidates"][0].update(finishReason="SAFETY"),
    lambda x: x["candidates"][0].update(finishReason="MAX_TOKENS"),
    lambda x: x["candidates"][0].update(finishReason="RECITATION"),
    lambda x: x["candidates"][0].update(finishReason="OTHER"),
    lambda x: x["candidates"][0].pop("finishReason"),
    lambda x: x["candidates"][0].pop("content"),
    lambda x: x["candidates"][0]["content"].update(role="user"),
    lambda x: x["candidates"][0]["content"].pop("role"),
    lambda x: x["candidates"][0]["content"].update(parts=[]),
    lambda x: x["candidates"][0]["content"]["parts"].append({"text": ANSWERED}),
    lambda x: x["candidates"][0]["content"]["parts"][0].update(functionCall={"name": "tool"}),
    lambda x: x["candidates"][0]["content"]["parts"][0].update(text=""),
    lambda x: x.update(modelVersion="gemini-3.5-flash"),
    lambda x: x.update(modelVersion="gemini-3.5-flash-lite-other"),
    lambda x: x.pop("modelVersion"),
])
def test_invalid_gemini_envelope_rejected(mutation) -> None:
    payload = _envelope()
    mutation(payload)
    provider = _provider(lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(GenerationError) as error:
            _generate(provider)
        assert "private" not in str(error.value)
    finally:
        provider.close()


@pytest.mark.parametrize("text", [
    "not json", "{}", '{"status":"answered","claims":[]}',
    '{"status":"insufficient_evidence","claims":[{"text":"Wrong","source_ids":["S1"]}]}',
    '{"status":"answered","claims":[{"text":"A","source_ids":[]}]}',
    json.dumps({"status": "answered", "claims": [{"text": "x" * 601, "source_ids": ["S1"]}]}),
])
def test_invalid_structured_answer_rejected(text: str) -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_envelope(text)))
    try:
        with pytest.raises(GenerationError):
            _generate(provider)
    finally:
        provider.close()


@pytest.mark.parametrize("response", [
    httpx.Response(400, text="private parameter detail"),
    httpx.Response(401, text="private credential detail"),
    httpx.Response(403, text="private credential detail"),
    httpx.Response(429, text="private quota detail"),
    httpx.Response(500, text="private server detail"),
    httpx.Response(503, text="private server detail"),
    httpx.Response(302, headers={"location": "https://other.example/"}),
])
def test_http_failures_are_sanitized(response: httpx.Response) -> None:
    provider = _provider(lambda request: response)
    try:
        with pytest.raises(GenerationError) as error:
            _generate(provider)
        assert "private" not in str(error.value)
    finally:
        provider.close()


@pytest.mark.parametrize("failure", [
    httpx.ConnectError("private key detail"), httpx.ReadTimeout("private prompt detail"),
])
def test_network_failures_are_sanitized(failure: Exception) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise failure

    provider = _provider(fail)
    try:
        with pytest.raises(GenerationError) as error:
            _generate(provider)
        assert "private" not in str(error.value)
    finally:
        provider.close()


def test_oversized_body_and_sensitive_logging(caplog: pytest.LogCaptureFixture) -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_envelope("private document" * 1000)))
    try:
        with caplog.at_level(logging.INFO), pytest.raises(GenerationError) as error:
            provider.generate(
                system_prompt="safe", user_prompt="private synthetic evidence",
                config=GenerationConfig(max_response_bytes=4096),
            )
        assert "private" not in str(error.value) and "private" not in caplog.text
    finally:
        provider.close()


def test_bad_model_and_missing_key_rejected() -> None:
    with pytest.raises(GenerationError):
        GeminiGenerationProvider(api_key=SecretStr("synthetic"), model_name="other")
    with pytest.raises(GenerationError):
        GeminiGenerationProvider(api_key=SecretStr(""))
