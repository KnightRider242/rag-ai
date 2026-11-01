"""Mocked Responses API transport; these tests never contact OpenAI."""

import json
import logging
from copy import deepcopy

import httpx
import pytest
from pydantic import SecretStr

from rag_ai.generation.base import GenerationConfig, GenerationError
from rag_ai.generation.openai import OpenAIGenerationProvider

ANSWERED = '{"status":"answered","claims":[{"text":"The relay disconnects power.","source_ids":["S1"]}]}'
ABSTAINED = '{"status":"insufficient_evidence","claims":[]}'


def _envelope(text: str = ANSWERED) -> dict:
    return {
        "object": "response", "status": "completed", "model": "gpt-6-luna",
        "error": None, "incomplete_details": None,
        "output": [{
            "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }],
    }


def _provider(handler) -> OpenAIGenerationProvider:
    return OpenAIGenerationProvider(
        api_key=SecretStr("synthetic-test-key"), model_name="gpt-6-luna",
        transport=httpx.MockTransport(handler),
    )


def _generate(provider: OpenAIGenerationProvider):
    return provider.generate(
        system_prompt="trusted instructions", user_prompt="private synthetic evidence",
        config=GenerationConfig(),
    )


def test_request_contract_success_and_cleanup() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == "Bearer synthetic-test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-6-luna"
        assert payload["store"] is False and payload["stream"] is False
        assert payload["tools"] == [] and payload["tool_choice"] == "none"
        assert payload["reasoning"] == {"effort": "none"}
        assert payload["max_output_tokens"] == 512
        assert "temperature" not in payload and "num_ctx" not in payload
        assert payload["input"] == [
            {"role": "developer", "content": "trusted instructions"},
            {"role": "user", "content": "private synthetic evidence"},
        ]
        fmt = payload["text"]["format"]
        assert fmt["type"] == "json_schema" and fmt["strict"] is True
        schema = fmt["schema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema["required"]) == {"status", "claims"}
        claim_schema = schema["properties"]["claims"]["items"]
        assert claim_schema["additionalProperties"] is False
        assert set(claim_schema["required"]) == {"text", "source_ids"}
        return httpx.Response(200, json=_envelope())

    provider = _provider(handle)
    try:
        assert _generate(provider).claims[0].source_ids == ["S1"]
    finally:
        provider.close()
    assert provider._client.is_closed


def test_structured_abstention() -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_envelope(ABSTAINED)))
    try:
        result = _generate(provider)
        assert result.status == "insufficient_evidence" and result.claims == []
    finally:
        provider.close()


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
    lambda x: x.update(status="failed"),
    lambda x: x.update(status="cancelled"),
    lambda x: x.update(model="wrong-model"),
    lambda x: x.update(output=[]),
    lambda x: x["output"].append(deepcopy(x["output"][0])),
    lambda x: x["output"][0].update(type="function_call"),
    lambda x: x["output"][0].update(status="incomplete"),
    lambda x: x["output"][0].update(role="user"),
    lambda x: x["output"][0]["content"].append({"type": "output_text", "text": ANSWERED}),
    lambda x: x["output"][0].update(content=[]),
    lambda x: x["output"][0]["content"][0].update(type="refusal", refusal="private refusal"),
    lambda x: x["output"][0]["content"][0].update(text=""),
    lambda x: x.update(error={"message": "private provider detail"}),
])
def test_invalid_response_envelope_rejected(mutation) -> None:
    payload = _envelope()
    mutation(payload)
    provider = _provider(lambda request: httpx.Response(200, json=payload))
    try:
        with pytest.raises(GenerationError) as error:
            _generate(provider)
        assert "private" not in str(error.value)
    finally:
        provider.close()


@pytest.mark.parametrize("content", [
    "not json", "{}", '{"status":"invented","claims":[]}',
    '{"status":"answered","claims":[]}',
    '{"status":"insufficient_evidence","claims":[{"text":"Wrong","source_ids":["S1"]}]}',
    '{"status":"answered","claims":[{"text":"A","source_ids":[]}]}',
    json.dumps({"status": "answered", "claims": [{"text": "x" * 601, "source_ids": ["S1"]}]}),
])
def test_invalid_structured_output_rejected(content: str) -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_envelope(content)))
    try:
        with pytest.raises(GenerationError):
            _generate(provider)
    finally:
        provider.close()


def test_invalid_response_json_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, text="not json"))
    try:
        with pytest.raises(GenerationError):
            _generate(provider)
    finally:
        provider.close()


@pytest.mark.parametrize("response", [
    httpx.Response(401, text="private key detail"),
    httpx.Response(429, text="private rate detail"),
    httpx.Response(503, text="private provider detail"),
    httpx.Response(302, headers={"location": "https://other.example/"}),
])
def test_http_failure_is_controlled(response: httpx.Response) -> None:
    provider = _provider(lambda request: response)
    try:
        with pytest.raises(GenerationError) as error:
            _generate(provider)
        assert "private" not in str(error.value)
    finally:
        provider.close()


@pytest.mark.parametrize("failure", [httpx.ConnectError("private key detail"),
                                     httpx.ReadTimeout("private prompt detail")])
def test_network_failure_is_controlled(failure: Exception) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise failure

    provider = _provider(fail)
    try:
        with pytest.raises(GenerationError) as error:
            _generate(provider)
        assert "private" not in str(error.value)
    finally:
        provider.close()


def test_response_limit_and_no_sensitive_logging(caplog: pytest.LogCaptureFixture) -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_envelope("private document" * 1000)))
    try:
        with caplog.at_level(logging.INFO), pytest.raises(GenerationError) as error:
            provider.generate(system_prompt="safe", user_prompt="private synthetic evidence",
                              config=GenerationConfig(max_response_bytes=4096))
        assert "private" not in str(error.value) and "private" not in caplog.text
    finally:
        provider.close()


def test_unsupported_model_and_missing_key_rejected() -> None:
    with pytest.raises(GenerationError):
        OpenAIGenerationProvider(api_key=SecretStr("synthetic"), model_name="other")
    with pytest.raises(GenerationError):
        OpenAIGenerationProvider(api_key=SecretStr(""))
