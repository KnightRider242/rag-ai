"""Synthetic local generation transport and schema security tests."""

import json
import logging

import httpx
import pytest

from rag_ai.generation.base import GenerationConfig, GenerationError
from rag_ai.generation.ollama import OllamaGenerationProvider


def _response(
    content: str, *, model: str = "qwen3:4b-instruct", done: bool = True,
    done_reason: str | None = "stop", role: str = "assistant",
) -> httpx.Response:
    return httpx.Response(200, json={
        "model": model, "done": done, "done_reason": done_reason,
        "message": {"role": role, "content": content},
    })


def _provider(handler: httpx.MockTransport) -> OllamaGenerationProvider:
    return OllamaGenerationProvider(
        base_url="http://127.0.0.1:11434", model_name="qwen3:4b-instruct", transport=handler,
    )


def test_structured_generation_request_and_cleanup() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert payload["model"] == "qwen3:4b-instruct"
        assert payload["stream"] is False
        assert payload["options"] == {"num_ctx": 4096, "num_predict": 512, "temperature": 0}
        assert payload["messages"][0] == {"role": "system", "content": "system instructions"}
        assert len(payload["format"]["oneOf"]) == 2
        return _response('{"status":"answered","claims":[{"text":"Synthetic answer","source_ids":["S1"]}]}')

    provider = _provider(httpx.MockTransport(handle))
    result = provider.generate(
        system_prompt="system instructions", user_prompt="synthetic evidence",
        config=GenerationConfig(),
    )
    assert result.claims[0].source_ids == ["S1"]
    provider.close()
    assert provider._client.is_closed


@pytest.mark.parametrize("content", [
    "not json", "", "{}",
    '{"status":"invented","claims":[]}',
    '{"status":"answered","claims":[]}',
    '{"status":"insufficient_evidence","claims":[{"text":"Wrong","source_ids":["S1"]}]}',
    '{"status":"answered","claims":[{"text":"A","source_ids":[],"extra":true}]}',
    '{"status":"answered","claims":[{"text":" ","source_ids":["S1"]}]}',
])
def test_invalid_structured_output_is_rejected(content: str) -> None:
    provider = _provider(httpx.MockTransport(lambda request: _response(content)))
    try:
        with pytest.raises(GenerationError):
            provider.generate(system_prompt="safe", user_prompt="synthetic", config=GenerationConfig())
    finally:
        provider.close()


@pytest.mark.parametrize("response", [
    _response('{"status":"insufficient_evidence","claims":[]}', model="wrong-model"),
    _response('{"status":"insufficient_evidence","claims":[]}', done=False),
    httpx.Response(503, text="private synthetic prompt"),
    httpx.Response(302, headers={"location": "https://external.example/api/chat"}),
])
def test_wrong_model_http_error_and_redirect_are_controlled(response: httpx.Response) -> None:
    provider = _provider(httpx.MockTransport(lambda request: response))
    try:
        with pytest.raises(GenerationError) as error:
            provider.generate(system_prompt="safe", user_prompt="private synthetic prompt",
                              config=GenerationConfig())
        assert "private synthetic prompt" not in str(error.value)
    finally:
        provider.close()


@pytest.mark.parametrize("done_reason", ["length", "unexpected", "", None])
def test_non_normal_termination_is_rejected(done_reason: str | None) -> None:
    content = '{"status":"answered","claims":[{"text":"Synthetic answer","source_ids":["S1"]}]}'
    provider = _provider(httpx.MockTransport(
        lambda request: _response(content, done_reason=done_reason),
    ))
    try:
        with pytest.raises(GenerationError) as error:
            provider.generate(system_prompt="safe", user_prompt="private synthetic prompt",
                              config=GenerationConfig())
        assert "private synthetic prompt" not in str(error.value)
        assert content not in str(error.value)
    finally:
        provider.close()


def test_missing_termination_reason_is_rejected() -> None:
    payload = _response('{"status":"insufficient_evidence","claims":[]}').json()
    del payload["done_reason"]
    provider = _provider(httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    try:
        with pytest.raises(GenerationError):
            provider.generate(system_prompt="safe", user_prompt="synthetic", config=GenerationConfig())
    finally:
        provider.close()


@pytest.mark.parametrize("role", ["user", "system", "", None])
def test_non_assistant_role_is_rejected(role: str | None) -> None:
    payload = _response('{"status":"insufficient_evidence","claims":[]}').json()
    payload["message"]["role"] = role
    provider = _provider(httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    try:
        with pytest.raises(GenerationError):
            provider.generate(system_prompt="safe", user_prompt="synthetic", config=GenerationConfig())
    finally:
        provider.close()


def test_missing_assistant_role_is_rejected() -> None:
    payload = _response('{"status":"insufficient_evidence","claims":[]}').json()
    del payload["message"]["role"]
    provider = _provider(httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    try:
        with pytest.raises(GenerationError):
            provider.generate(system_prompt="safe", user_prompt="synthetic", config=GenerationConfig())
    finally:
        provider.close()


@pytest.mark.parametrize("error", [httpx.ConnectError("private synthetic prompt"),
                                      httpx.ReadTimeout("private synthetic prompt")])
def test_network_failure_and_timeout_are_sanitized(error: Exception) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise error

    provider = _provider(httpx.MockTransport(fail))
    try:
        with pytest.raises(GenerationError) as caught:
            provider.generate(system_prompt="safe", user_prompt="private synthetic prompt",
                              config=GenerationConfig())
        assert "private synthetic prompt" not in str(caught.value)
    finally:
        provider.close()


def test_oversized_response_and_no_sensitive_logging(caplog: pytest.LogCaptureFixture) -> None:
    provider = _provider(httpx.MockTransport(lambda request: _response("private synthetic document" * 300)))
    try:
        with caplog.at_level(logging.INFO), pytest.raises(GenerationError) as error:
            provider.generate(system_prompt="safe", user_prompt="private synthetic prompt",
                              config=GenerationConfig(max_response_bytes=4096))
        assert "private synthetic" not in str(error.value)
        assert "private synthetic" not in caplog.text
    finally:
        provider.close()


@pytest.mark.parametrize("url", [
    "https://api.example.com:443", "http://example.com:11434",
    "http://user:pass@127.0.0.1:11434", "http://127.0.0.1:11434/redirect",
])
def test_constructor_rejects_nonlocal_or_credentialed_urls(url: str) -> None:
    with pytest.raises(GenerationError):
        OllamaGenerationProvider(base_url=url, model_name="qwen3:4b-instruct")
