"""Five-page, service-free visual API workflow with application-owned provenance."""

import asyncio
import json
import logging
from hashlib import sha256

import httpx
import pymupdf
import pytest
from pydantic import ValidationError

from rag_ai.core.config import Settings
from rag_ai.main import create_app
from rag_ai.vision.base import VisionOutput
from rag_ai.vision.base import VisionError
from rag_ai.vision.ollama import OllamaVisionProvider
from rag_ai.vision.evidence import safe_image_upload
from rag_ai.documents.parser import DocumentParseError


class MockVision:
    model_name = "synthetic-vision"
    calls = 0
    closed = False

    def analyze(self, *, image: bytes, task_prompt: str) -> VisionOutput:
        self.calls += 1
        assert image.startswith(b"\x89PNG") or image.startswith(b"\xff\xd8")
        assert "source_id" not in task_prompt
        return VisionOutput(description="A synthetic component is visible.",
                            observations=["component"], limitations=["Synthetic evaluation only"])

    def close(self) -> None:
        self.closed = True


def fixture_pdf() -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, 20, 20, bytes([0, 0, 0] * 400), False)
    with pymupdf.open() as doc:
        doc.new_page().insert_text((50, 50), "Synthetic engineering text")
        doc.new_page().insert_image(pymupdf.Rect(50, 50, 150, 150), pixmap=pix)
        page = doc.new_page()
        page.draw_rect(pymupdf.Rect(50, 50, 150, 150), color=(0, 0, 0))
        page.draw_line((50, 50), (150, 150), color=(0, 0, 0))
        page = doc.new_page()
        for y in (50, 80, 110):
            page.draw_line((50, y), (250, y))
        for x in (50, 150, 250):
            page.draw_line((x, 50), (x, 110))
        for value, point in (("Relay", (60, 70)), ("Open", (160, 70)),
                             ("Pump", (60, 100)), ("On", (160, 100))):
            page.insert_text(point, value)
        doc.new_page().insert_image(pymupdf.Rect(50, 50, 150, 150), pixmap=pix)
        return doc.tobytes()


def settings(**overrides):
    return Settings(app_env="development", enable_document_visual_inspection_endpoint=True,
                    enable_document_visual_analysis_endpoint=True, vision_model="synthetic-vision",
                    _env_file=None, **overrides)


def request(method: str, url: str, *, settings_obj: Settings, **kwargs) -> httpx.Response:
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(settings_obj)),
                                     base_url="http://test") as client:
            return await client.request(method, url, **kwargs)
    return asyncio.run(send())


def test_five_page_pdf_api_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_ai.api import vision
    models = []
    def construct(_settings):
        model = MockVision()
        models.append(model)
        return model
    monkeypatch.setattr(vision, "_provider", construct)
    content = fixture_pdf()
    config = settings()
    inspected = request("POST", "/documents/inspect", settings_obj=config,
                        files={"file": ("synthetic.pdf", content, "application/pdf")})
    assert inspected.status_code == 200
    pages = inspected.json()["pages"]
    assert len(pages) == 5
    assert pages[0]["has_text"] is True
    assert pages[1]["image_count"] == 1
    assert pages[2]["drawing_count"] >= 1
    assert pages[3]["table_count"] >= 1
    assert pages[3]["tables"][0]["cells"][0] == ["Relay", "Open"]
    assert pages[4]["has_text"] is False and pages[4]["image_count"] == 1
    preview = request("POST", "/documents/preview", settings_obj=config,
                      data={"page_number": "3"}, files={"file": ("synthetic.pdf", content, "application/pdf")})
    assert preview.status_code == 200 and preview.content.startswith(b"\x89PNG")
    crop = request("POST", "/documents/preview", settings_obj=config,
                   data={"page_number": "3", "x0": "40", "y0": "40", "x1": "160", "y1": "160"},
                   files={"file": ("synthetic.pdf", content, "application/pdf")})
    assert crop.status_code == 200 and len(crop.content) < len(preview.content)
    analyzed = request("POST", "/documents/analyze-visual", settings_obj=config,
                       data={"page_number": "3", "task": "describe_diagram",
                             "x0": "40", "y0": "40", "x1": "160", "y1": "160"},
                       files={"file": ("synthetic.pdf", content, "application/pdf")})
    assert analyzed.status_code == 200
    evidence = analyzed.json()
    assert evidence["document_id"] == sha256(content).hexdigest()
    assert evidence["page_number"] == 3 and evidence["filename"] == "synthetic.pdf"
    assert evidence["region"] == {"x0": 40, "y0": 40, "x1": 160, "y1": 160}
    assert evidence["source_id"] != evidence["document_id"]
    assert evidence["extraction_method"] == "local_ollama_vision"
    for page_number, task in [(4, "interpret_table"), (5, "examine_scanned_page")]:
        response = request("POST", "/documents/analyze-visual", settings_obj=config,
                           data={"page_number": str(page_number), "task": task},
                           files={"file": ("synthetic.pdf", content, "application/pdf")})
        assert response.status_code == 200
        assert response.json()["page_number"] == page_number
        assert response.json()["source_id"] != evidence["source_id"]
    assert len(models) == 3 and all(model.closed and model.calls == 1 for model in models)
    assert "Synthetic engineering text" not in analyzed.text


@pytest.mark.parametrize("extension,media_type,format_name", [
    ("png", "image/png", "png"), ("jpg", "image/jpeg", "jpeg"),
])
def test_standalone_images(monkeypatch: pytest.MonkeyPatch, extension: str,
                           media_type: str, format_name: str) -> None:
    from rag_ai.api import vision
    monkeypatch.setattr(vision, "_provider", lambda _settings: MockVision())
    pix = pymupdf.Pixmap(pymupdf.csRGB, 20, 10, bytes([0, 0, 0] * 200), False)
    content = pix.tobytes(format_name)
    response = request("POST", "/images/analyze", settings_obj=settings(),
                       data={"task": "describe_figure"},
                       files={"file": (f"synthetic.{extension}", content, media_type)})
    assert response.status_code == 200
    result = response.json()
    assert result["source_id"] == sha256(content).hexdigest()
    assert result["page_number"] is None
    assert result["region"] == {"x0": 0, "y0": 0, "x1": 20, "y1": 10}
    assert result["source_type"] == ("jpeg" if extension == "jpg" else "png")


def test_spoofed_or_oversized_image_rejected_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_ai.api import vision
    monkeypatch.setattr(vision, "_provider", lambda _settings: pytest.fail("provider must not be built"))
    with pytest.raises(DocumentParseError):
        safe_image_upload("fake.png", "image/png", b"not an image", max_pixels=100, max_size_bytes=1000)
    pix = pymupdf.Pixmap(pymupdf.csRGB, 20, 20, bytes([0, 0, 0] * 400), False)
    with pytest.raises(DocumentParseError):
        safe_image_upload("spoof.png", "image/png", pix.tobytes("jpeg"),
                          max_pixels=1000, max_size_bytes=1000)
    with pytest.raises(DocumentParseError):
        safe_image_upload("large.png", "image/png", pix.tobytes("png"),
                          max_pixels=100, max_size_bytes=1000)
    response = request("POST", "/images/analyze", settings_obj=settings(),
                       files={"file": ("fake.png", b"not an image", "image/png")})
    assert response.status_code == 422


@pytest.mark.parametrize("format_name,extension,media_type", [
    ("png", "png", "image/png"), ("jpeg", "jpg", "image/jpeg"),
])
def test_truncated_image_is_controlled_and_never_reaches_provider(
    monkeypatch: pytest.MonkeyPatch, format_name: str, extension: str, media_type: str,
) -> None:
    from rag_ai.api import vision
    monkeypatch.setattr(vision, "_provider", lambda _settings: pytest.fail("provider must not be built"))
    pix = pymupdf.Pixmap(pymupdf.csRGB, 20, 20, bytes([0, 0, 0] * 400), False)
    truncated = pix.tobytes(format_name)[:-1]
    with pytest.raises(DocumentParseError) as error:
        safe_image_upload(f"synthetic.{extension}", media_type, truncated,
                          max_pixels=1000, max_size_bytes=1000)
    assert error.value.code == "invalid_image"
    response = request("POST", "/images/analyze", settings_obj=settings(),
                       files={"file": (f"synthetic.{extension}", truncated, media_type)})
    assert response.status_code == 422
    assert "invalid_image" in response.text


def test_routes_disabled_by_default_and_production_rejects_opt_in() -> None:
    app = create_app(Settings(app_env="development", _env_file=None))
    assert "/documents/preview" not in app.openapi()["paths"]
    assert "/images/analyze" not in app.openapi()["paths"]
    with pytest.raises(ValidationError):
        Settings(app_env="production", enable_document_visual_analysis_endpoint=True,
                 vision_model="synthetic-vision", _env_file=None)


def test_missing_model_and_invalid_region_fail_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_ai.api import vision
    with pytest.raises(ValidationError):
        Settings(app_env="development", enable_document_visual_analysis_endpoint=True,
                 vision_model="", _env_file=None)
    monkeypatch.setattr(vision, "_provider", lambda _settings: pytest.fail("provider must not be built"))
    response = request("POST", "/documents/analyze-visual", settings_obj=settings(),
                       data={"page_number": "1", "x0": "0", "y0": "0", "x1": "9999", "y1": "50"},
                       files={"file": ("synthetic.pdf", fixture_pdf(), "application/pdf")})
    assert response.status_code == 422


@pytest.mark.parametrize("region", [
    {"x0": "1", "y0": "1", "x1": "1", "y1": "10"},
    {"x0": "nan", "y0": "1", "x1": "10", "y1": "10"},
    {"x0": "1", "y0": "-inf", "x1": "10", "y1": "10"},
])
def test_invalid_pdf_regions_fail_before_provider(
    monkeypatch: pytest.MonkeyPatch, region: dict[str, str],
) -> None:
    from rag_ai.api import vision
    monkeypatch.setattr(vision, "_provider", lambda _settings: pytest.fail("provider must not be built"))
    response = request("POST", "/documents/analyze-visual", settings_obj=settings(),
                       data={"page_number": "1", **region},
                       files={"file": ("synthetic.pdf", fixture_pdf(), "application/pdf")})
    assert response.status_code == 422


@pytest.mark.parametrize("content,media_type,status", [
    (b"not a PDF", "application/pdf", 422),
    (b"%PDF-1.7\ninvalid", "application/pdf", 422),
    (b"", "application/pdf", 422),
    (b"synthetic", "image/png", 415),
])
def test_bad_pdf_uploads_fail_before_provider(
    monkeypatch: pytest.MonkeyPatch, content: bytes, media_type: str, status: int,
) -> None:
    from rag_ai.api import vision
    monkeypatch.setattr(vision, "_provider", lambda _settings: pytest.fail("provider must not be built"))
    response = request("POST", "/documents/analyze-visual", settings_obj=settings(),
                       data={"page_number": "1"},
                       files={"file": ("synthetic.pdf", content, media_type)})
    assert response.status_code == status


def test_protected_pdf_fails_before_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_ai.api import vision
    monkeypatch.setattr(vision, "_provider", lambda _settings: pytest.fail("provider must not be built"))
    with pymupdf.open() as document:
        document.new_page()
        content = document.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,
                                   owner_pw="synthetic-owner", user_pw="synthetic-user")
    response = request("POST", "/documents/analyze-visual", settings_obj=settings(),
                       data={"page_number": "1"},
                       files={"file": ("synthetic.pdf", content, "application/pdf")})
    assert response.status_code == 422


def test_provider_failure_is_sanitized_and_client_closes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    from rag_ai.api import vision
    class Broken(MockVision):
        def analyze(self, *, image: bytes, task_prompt: str) -> VisionOutput:
            raise VisionError("private provider detail")
    model = Broken()
    monkeypatch.setattr(vision, "_provider", lambda _settings: model)
    pix = pymupdf.Pixmap(pymupdf.csRGB, 2, 2, bytes([0, 0, 0] * 4), False)
    with caplog.at_level(logging.WARNING, logger="rag_ai.api.vision"):
        response = request("POST", "/images/analyze", settings_obj=settings(),
                           files={"file": ("synthetic.png", pix.tobytes("png"), "image/png")})
    assert response.status_code == 502 and "private provider detail" not in response.text
    assert "private provider detail" not in caplog.text
    assert "invalid_envelope" in caplog.text
    assert model.closed


def test_generation_limit_has_safe_api_diagnostics(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    from rag_ai.api import vision
    content = json.dumps({"description": "Synthetic result", "observations": [], "limitations": []})
    envelope = {"model": "synthetic-vision", "message": {"role": "assistant", "content": content},
                "done": True, "done_reason": "length"}
    model = OllamaVisionProvider(base_url="http://127.0.0.1:11434", model_name="synthetic-vision",
                                 transport=httpx.MockTransport(lambda request: httpx.Response(200, json=envelope)))
    monkeypatch.setattr(vision, "_provider", lambda _settings: model)
    pix = pymupdf.Pixmap(pymupdf.csRGB, 2, 2, bytes([0, 0, 0] * 4), False)
    with caplog.at_level(logging.WARNING, logger="rag_ai.api.vision"):
        response = request("POST", "/images/analyze", settings_obj=settings(),
                           files={"file": ("synthetic.png", pix.tobytes("png"), "image/png")})
    assert response.status_code == 502
    assert response.json() == {"detail": {"code": "vision_failed", "message": "Local visual analysis failed."}}
    assert "generation_limit" in caplog.text
    assert content not in response.text + caplog.text
    assert model._client.is_closed


@pytest.mark.parametrize("failure,category", [
    ("numeric_done", "invalid_envelope"),
    ("http_400", "http_status"),
    ("http_500", "http_status"),
    ("connection_refused", "transport"),
    ("connection_timeout", "timeout"),
    ("read_timeout", "timeout"),
    ("redirect", "transport"),
])
def test_provider_boundary_failures_keep_public_error_sanitized(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    failure: str, category: str,
) -> None:
    from rag_ai.api import vision

    def handler(_request: httpx.Request) -> httpx.Response:
        if failure == "numeric_done":
            return httpx.Response(200, json={
                "model": "synthetic-vision", "message": {"role": "assistant", "content":
                    '{"description":"private document content","observations":[],"limitations":[]}'},
                "done": 1, "done_reason": "stop",
            })
        if failure == "http_400":
            return httpx.Response(400, text="private document content and credentials")
        if failure == "http_500":
            return httpx.Response(500, text="private document content and credentials")
        if failure == "redirect":
            return httpx.Response(302, headers={"location": "https://private.example"})
        if failure == "connection_refused":
            raise httpx.ConnectError("private connection refused")
        if failure == "connection_timeout":
            raise httpx.ConnectTimeout("private connection timeout")
        raise httpx.ReadTimeout("private read timeout")

    model = OllamaVisionProvider(base_url="http://127.0.0.1:11434", model_name="synthetic-vision",
                                 transport=httpx.MockTransport(handler))
    monkeypatch.setattr(vision, "_provider", lambda _settings: model)
    pix = pymupdf.Pixmap(pymupdf.csRGB, 2, 2, bytes([0, 0, 0] * 4), False)
    with caplog.at_level(logging.WARNING, logger="rag_ai.api.vision"):
        response = request("POST", "/images/analyze", settings_obj=settings(),
                           files={"file": ("synthetic.png", pix.tobytes("png"), "image/png")})
    assert response.status_code == 502
    assert response.json() == {"detail": {"code": "vision_failed", "message": "Local visual analysis failed."}}
    assert category in caplog.text
    assert "private" not in response.text + caplog.text
    assert model._client.is_closed


@pytest.mark.parametrize("data,expected", [
    ({"page_number": "99"}, 422),
    ({"page_number": "1", "scale": "20"}, 422),
    ({"page_number": "1", "x0": "0"}, 422),
])
def test_preview_invalid_inputs(data: dict[str, str], expected: int) -> None:
    response = request("POST", "/documents/preview", settings_obj=settings(), data=data,
                       files={"file": ("synthetic.pdf", fixture_pdf(), "application/pdf")})
    assert response.status_code == expected


def test_rotated_preview_and_visual_analysis_receive_intended_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_ai.api import vision

    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=400)
        page.draw_rect(pymupdf.Rect(40, 50, 90, 100), color=None, fill=(1, 0, 0))
        page.draw_rect(pymupdf.Rect(180, 250, 230, 300), color=None, fill=(0, 0, 1))
        page.set_rotation(90)
        red = pymupdf.Rect(50, 60, 80, 90) * page.rotation_matrix
        blue = pymupdf.Rect(190, 260, 220, 290) * page.rotation_matrix
        content = doc.tobytes()

    class CaptureVision(MockVision):
        received: bytes | None = None

        def analyze(self, *, image: bytes, task_prompt: str) -> VisionOutput:
            self.received = image
            return super().analyze(image=image, task_prompt=task_prompt)

    model = CaptureVision()
    monkeypatch.setattr(vision, "_provider", lambda _settings: model)

    def region_data(rect: pymupdf.Rect) -> dict[str, str]:
        return {"page_number": "1", "x0": str(rect.x0), "y0": str(rect.y0),
                "x1": str(rect.x1), "y1": str(rect.y1)}

    for rect, expected in ((red, (255, 0, 0)), (blue, (0, 0, 255))):
        preview = request("POST", "/documents/preview", settings_obj=settings(),
                          data=region_data(rect),
                          files={"file": ("colored.pdf", content, "application/pdf")})
        assert preview.status_code == 200
        pixmap = pymupdf.Pixmap(preview.content)
        assert pixmap.pixel(pixmap.width // 2, pixmap.height // 2) == expected

    analyzed = request("POST", "/documents/analyze-visual", settings_obj=settings(),
                       data={**region_data(red), "task": "describe_figure"},
                       files={"file": ("colored.pdf", content, "application/pdf")})
    assert analyzed.status_code == 200
    assert model.received is not None
    pixels = pymupdf.Pixmap(model.received)
    assert pixels.pixel(pixels.width // 2, pixels.height // 2) == (255, 0, 0)
    assert model.closed
    assert analyzed.json()["document_id"] == sha256(content).hexdigest()
    assert analyzed.json()["region"] == {"x0": red.x0, "y0": red.y0,
                                          "x1": red.x1, "y1": red.y1}
