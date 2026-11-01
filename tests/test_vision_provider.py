"""Mocked Ollama vision boundary; no real model or network required."""

import json

import httpx
import pytest

from rag_ai.vision.base import VisionError
from rag_ai.vision.ollama import OllamaVisionProvider


def body(*, content: str = '{"description":"A relay is shown.","observations":["relay"],"limitations":[]}',
         done_reason: str = "stop", role: str = "assistant", model: str = "synthetic-vision") -> dict:
    return {"model": model, "message": {"role": role, "content": content},
            "done": True, "done_reason": done_reason}


def provider(handler):
    return OllamaVisionProvider(base_url="http://127.0.0.1:11434", model_name="synthetic-vision",
                                transport=httpx.MockTransport(handler))


def test_valid_structured_vision_request() -> None:
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=body())
    model = provider(handler)
    try:
        result = model.analyze(image=b"synthetic-image", task_prompt="Describe this figure")
        assert result.description == "A relay is shown."
        request = seen[0]
        assert str(request.url) == "http://127.0.0.1:11434/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "synthetic-vision"
        assert payload["messages"][1]["images"] == ["c3ludGhldGljLWltYWdl"]
        assert payload["stream"] is False and "tools" not in payload
        assert payload["options"] == {"num_ctx": 4096, "num_predict": 512, "temperature": 0}
        assert "description" in payload["format"]["properties"]
    finally:
        model.close()
    assert model._client.is_closed


@pytest.mark.parametrize("result", [
    body(done_reason="length"), body(role="user"), body(model="other"),
    body(content="not json"), body(content='{"description":" ","observations":[],"limitations":[]}'),
    body(content='{"description":"ok","observations":[],"limitations":[],"source_id":"invented"}'),
    {"model": "synthetic-vision", "message": {"role": "assistant", "content": ""}, "done": False},
])
def test_invalid_responses_fail_closed(result: dict) -> None:
    model = provider(lambda request: httpx.Response(200, json=result))
    try:
        with pytest.raises(VisionError):
            model.analyze(image=b"image", task_prompt="Describe")
    finally:
        model.close()


@pytest.mark.parametrize("reason,category", [
    ("length", "generation_limit"), ("unexpected", "invalid_envelope"),
    (None, "invalid_envelope"),
])
def test_completion_reason_is_strict_even_with_valid_structured_output(
    reason: str | None, category: str,
) -> None:
    result = body()
    if reason is None:
        result.pop("done_reason")
    else:
        result["done_reason"] = reason
    model = provider(lambda request: httpx.Response(200, json=result))
    try:
        with pytest.raises(VisionError) as error:
            model.analyze(image=b"image", task_prompt="Describe")
        assert error.value.category == category
    finally:
        model.close()


@pytest.mark.parametrize("change", [
    {"done": False}, {"done": None}, {"done_reason": None},
    {"message": {"role": "user", "content": "{}"}},
])
def test_invalid_completion_envelope_is_categorized(change: dict) -> None:
    result = body()
    result.update(change)
    model = provider(lambda request: httpx.Response(200, json=result))
    try:
        with pytest.raises(VisionError) as error:
            model.analyze(image=b"image", task_prompt="Describe")
        assert error.value.category == "invalid_envelope"
    finally:
        model.close()


@pytest.mark.parametrize("done_value", [False, 1, 1.0, "true", None, "missing"])
def test_done_requires_actual_json_true(done_value: object) -> None:
    result = body()
    if done_value == "missing":
        result.pop("done")
    else:
        result["done"] = done_value
    model = provider(lambda request: httpx.Response(200, json=result))
    try:
        with pytest.raises(VisionError) as error:
            model.analyze(image=b"image", task_prompt="Describe")
        assert error.value.category == "invalid_envelope"
    finally:
        model.close()


@pytest.mark.parametrize("content", [
    '{"description":', "not-json", "{}",
    json.dumps({"description": "x" * 1201, "observations": [], "limitations": []}),
    json.dumps({"description": "ok", "observations": ["x"] * 9, "limitations": []}),
    json.dumps({"description": "ok", "observations": [], "limitations": ["x"] * 6}),
    json.dumps({"description": "ok", "observations": [9], "limitations": []}),
    json.dumps({"description": "ok", "observations": [" "], "limitations": []}),
    json.dumps({"description": " ", "observations": [], "limitations": []}),
    json.dumps({"description": "ok", "observations": [], "limitations": [], "extra": "bad"}),
])
def test_invalid_structured_output_is_categorized(content: str) -> None:
    model = provider(lambda request: httpx.Response(200, json=body(content=content)))
    try:
        with pytest.raises(VisionError) as error:
            model.analyze(image=b"image", task_prompt="Describe")
        assert error.value.category == "invalid_output"
    finally:
        model.close()


def test_invalid_response_encoding_is_categorized() -> None:
    model = provider(lambda request: httpx.Response(200, content=b"\xff\xfe"))
    try:
        with pytest.raises(VisionError) as error:
            model.analyze(image=b"image", task_prompt="Describe")
        assert error.value.category == "invalid_envelope"
    finally:
        model.close()


@pytest.mark.parametrize("status", [302, 400, 401, 404, 429, 500, 503])
def test_http_failures_are_controlled(status: int) -> None:
    model = provider(lambda request: httpx.Response(
        status, headers={"location": "https://example.com"}, text="private provider body",
    ))
    try:
        with pytest.raises(VisionError) as error:
            model.analyze(image=b"image", task_prompt="Describe")
        assert error.value.category == ("transport" if status == 302 else "http_status")
        assert "private" not in str(error.value)
    finally:
        model.close()


def test_oversize_and_sensitive_output_are_not_exposed() -> None:
    model = OllamaVisionProvider(base_url="http://127.0.0.1:11434", model_name="synthetic-vision",
                                 max_response_bytes=4096, transport=httpx.MockTransport(
                                     lambda request: httpx.Response(200, text="private-document" * 1000)))
    try:
        with pytest.raises(VisionError) as error:
            model.analyze(image=b"image", task_prompt="Describe")
        assert "private-document" not in str(error.value)
        assert error.value.category == "resource_limit"
    finally:
        model.close()


def test_external_endpoint_is_rejected() -> None:
    with pytest.raises(VisionError):
        OllamaVisionProvider(base_url="https://external.example", model_name="synthetic-vision")


def test_unexpected_tool_call_is_rejected() -> None:
    result = body()
    result["message"]["tool_calls"] = [{"function": {"name": "unsafe"}}]
    model = provider(lambda request: httpx.Response(200, json=result))
    try:
        with pytest.raises(VisionError):
            model.analyze(image=b"image", task_prompt="Describe")
    finally:
        model.close()


@pytest.mark.parametrize("failure,category", [
    (httpx.ReadTimeout("private timeout"), "timeout"),
    (httpx.ConnectTimeout("private timeout"), "timeout"),
    (httpx.ConnectError("private host"), "transport"),
])
def test_transport_failures_are_sanitized(failure: Exception, category: str) -> None:
    def handler(request):
        raise failure
    model = provider(handler)
    try:
        with pytest.raises(VisionError) as error:
            model.analyze(image=b"image", task_prompt="Describe")
        assert "private" not in str(error.value)
        assert error.value.category == category
    finally:
        model.close()


@pytest.mark.parametrize("image,prompt", [(b"", "Describe"), (b"image", " "),
                                          (b"image", "x" * 1001)])
def test_invalid_input_rejected_before_transport(image: bytes, prompt: str) -> None:
    model = provider(lambda request: pytest.fail("transport must not run"))
    try:
        with pytest.raises(VisionError) as error:
            model.analyze(image=image, task_prompt=prompt)
        assert error.value.category == "resource_limit"
    finally:
        model.close()


def test_verified_digest_wraps_inference() -> None:
    calls = []
    digest = "a" * 64
    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "synthetic-vision", "digest": digest}]})
        return httpx.Response(200, json=body())
    model = provider(handler)
    try:
        output, revision = model.analyze_verified(image=b"image", task_prompt="Describe")
        assert output.description == "A relay is shown."
        assert revision == digest
        assert calls == ["/api/tags", "/api/chat", "/api/tags"]
    finally:
        model.close()


@pytest.mark.parametrize("tags", [
    {}, {"models": []}, {"models": [{"name": "other", "digest": "a" * 64}]},
    {"models": [{"name": "synthetic-vision", "digest": "invalid"}]},
    {"models": [{"name": "synthetic-vision", "digest": "a" * 64}] * 2},
    {"models": "wrong"},
])
def test_unverifiable_digest_fails_before_chat(tags: dict) -> None:
    model = provider(lambda request: httpx.Response(200, json=tags) if request.url.path == "/api/tags"
                     else pytest.fail("chat must not run"))
    try:
        with pytest.raises(VisionError):
            model.analyze_verified(image=b"image", task_prompt="Describe")
    finally:
        model.close()


def test_digest_change_after_chat_is_rejected() -> None:
    checks = 0
    def handler(request):
        nonlocal checks
        if request.url.path == "/api/tags":
            checks += 1
            return httpx.Response(200, json={"models": [
                {"name": "synthetic-vision", "digest": ("a" if checks == 1 else "b") * 64}]})
        return httpx.Response(200, json=body())
    model = provider(handler)
    try:
        with pytest.raises(VisionError, match="changed"):
            model.analyze_verified(image=b"image", task_prompt="Describe")
    finally:
        model.close()
