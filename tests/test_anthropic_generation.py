"""Synthetic Anthropic Messages API tests; no live cloud traffic."""

import json
import logging
from copy import deepcopy

import httpx
import pytest
from pydantic import SecretStr

from rag_ai.generation.anthropic import ANTHROPIC_API_VERSION, AnthropicGenerationProvider
from rag_ai.generation.base import GenerationConfig, GenerationError

ANSWERED = '{"status":"answered","claims":[{"text":"The relay disconnects power.","source_ids":["S1"]}]}'
ABSTAINED = '{"status":"insufficient_evidence","claims":[]}'


def _envelope(text: str = ANSWERED) -> dict:
    return {
        "id": "msg_synthetic", "type": "message", "role": "assistant",
        "model": "claude-haiku-5-5", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
    }


def _provider(handler) -> AnthropicGenerationProvider:
    return AnthropicGenerationProvider(
        api_key=SecretStr("synthetic-test-key"), model_name="claude-haiku-5-5",
        transport=httpx.MockTransport(handler),
    )


def _generate(provider: AnthropicGenerationProvider):
    return provider.generate(
        system_prompt="trusted synthetic instructions",
        user_prompt="private synthetic evidence", config=GenerationConfig(),
    )


def test_messages_request_contract_answer_and_cleanup() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "synthetic-test-key"
        assert request.headers["anthropic-version"] == ANTHROPIC_API_VERSION
        assert request.headers["content-type"] == "application/json"
        assert "synthetic-test-key" not in str(request.url)
        payload = json.loads(request.content)
        assert payload["model"] == "claude-haiku-5-5"
        assert payload["max_tokens"] == 512 and payload["stream"] is False
        assert payload["system"] == "trusted synthetic instructions"
        assert payload["messages"] == [{"role": "user", "content": "private synthetic evidence"}]
        assert "tools" not in payload and "tool_choice" not in payload
        assert "temperature" not in payload and "top_p" not in payload and "top_k" not in payload
        assert "thinking" not in payload and "num_ctx" not in payload
        fmt = payload["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["type"] == "object"
        assert fmt["schema"]["additionalProperties"] is False
        assert set(fmt["schema"]["required"]) == {"status", "claims"}
        claim_schema = fmt["schema"]["properties"]["claims"]["items"]
        assert claim_schema["additionalProperties"] is False
        assert set(claim_schema["required"]) == {"text", "source_ids"}
        return httpx.Response(200, json=_envelope())

    provider = _provider(handle)
    try:
        result = _generate(provider)
        assert result.status == "answered" and result.claims[0].source_ids == ["S1"]
    finally:
        provider.close()
    assert provider._client.is_closed


@pytest.mark.parametrize("include_null", [False, True])
def test_success_accepts_absent_or_null_stop_details(include_null: bool) -> None:
    payload = _envelope()
    if include_null:
        payload["stop_details"] = None
    provider = _provider(lambda request: httpx.Response(200, json=payload))
    try:
        assert _generate(provider).status == "answered"
    finally:
        provider.close()


@pytest.mark.parametrize("stop_reason,details", [
    ("end_turn", {"type": "refusal", "explanation": "private synthetic refusal"}),
    ("end_turn", {"type": "unexpected", "explanation": "private synthetic detail"}),
    ("refusal", {"type": "refusal", "explanation": "private synthetic refusal"}),
])
def test_non_null_stop_details_fail_closed_and_sanitized(
    stop_reason: str, details: dict[str, str], caplog: pytest.LogCaptureFixture,
) -> None:
    payload = _envelope()
    payload["stop_reason"] = stop_reason
    payload["stop_details"] = details
    provider = _provider(lambda request: httpx.Response(200, json=payload))
    try:
        with caplog.at_level(logging.INFO), pytest.raises(GenerationError) as error:
            _generate(provider)
        assert "private synthetic" not in str(error.value)
        assert "private synthetic" not in caplog.text
    finally:
        provider.close()


def test_abstention_and_thinking_blocks_are_not_answer_text() -> None:
    payload = _envelope(ABSTAINED)
    payload["content"] = [
        {"type": "thinking", "thinking": "private synthetic reasoning", "signature": "synthetic"},
        {"type": "redacted_thinking", "data": "synthetic"},
        payload["content"][0],
    ]
    provider = _provider(lambda request: httpx.Response(200, json=payload))
    try:
        result = _generate(provider)
        assert result.status == "insufficient_evidence" and result.claims == []
    finally:
        provider.close()


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(type="other"),
    lambda x: x.update(role="user"),
    lambda x: x.update(model="claude-sonnet-5-5"),
    lambda x: x.update(stop_reason="max_tokens"),
    lambda x: x.update(stop_reason="refusal"),
    lambda x: x.update(stop_reason="tool_use"),
    lambda x: x.update(stop_reason="pause_turn"),
    lambda x: x.pop("stop_reason"),
    lambda x: x.update(content=[]),
    lambda x: x.update(content=[{"type": "thinking", "thinking": "private"}]),
    lambda x: x["content"].append(deepcopy(x["content"][0])),
    lambda x: x["content"].append({"type": "tool_use", "name": "web_search"}),
    lambda x: x["content"][0].update(type="tool_use"),
    lambda x: x["content"][0].update(text=""),
    lambda x: x["content"][0].update(citations=[{"type": "native"}]),
])
def test_invalid_message_envelope_rejected(mutation) -> None:
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


def test_malformed_json_envelope_rejected() -> None:
    provider = _provider(lambda request: httpx.Response(200, text="not json"))
    try:
        with pytest.raises(GenerationError):
            _generate(provider)
    finally:
        provider.close()


@pytest.mark.parametrize("response", [
    httpx.Response(400, text="private parameter detail"),
    httpx.Response(401, text="private key detail"),
    httpx.Response(403, text="private key detail"),
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


def test_response_limit_and_logging(caplog: pytest.LogCaptureFixture) -> None:
    provider = _provider(lambda request: httpx.Response(200, json=_envelope("private document" * 1000)))
    try:
        with caplog.at_level(logging.INFO), pytest.raises(GenerationError) as error:
            provider.generate(
                system_prompt="trusted", user_prompt="private synthetic evidence",
                config=GenerationConfig(max_response_bytes=4096),
            )
        assert "private" not in str(error.value) and "private" not in caplog.text
    finally:
        provider.close()


def test_model_allowlist_missing_key_and_no_network_on_construction() -> None:
    with pytest.raises(GenerationError):
        AnthropicGenerationProvider(api_key=SecretStr("synthetic"), model_name="other")
    with pytest.raises(GenerationError):
        AnthropicGenerationProvider(api_key=SecretStr(""))
    calls = []
    provider = _provider(lambda request: calls.append(request))
    provider.close()
    assert calls == []
