"""Development-only local visual analysis routes."""

import logging
from hashlib import sha256
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from rag_ai.api.visual import _region
from rag_ai.core.config import Settings
from rag_ai.documents.parser import DocumentParseError, validate_upload_metadata
from rag_ai.documents.render import render_pdf
from rag_ai.vision.base import VisionError, VisionProvider
from rag_ai.vision.evidence import VisualEvidence, evidence_for_image, evidence_for_pdf, safe_image_upload
from rag_ai.vision.ollama import OllamaVisionProvider

pdf_router = APIRouter(prefix="/documents", tags=["documents"])
image_router = APIRouter(prefix="/images", tags=["images"])
logger = logging.getLogger(__name__)

VisionTask = Literal["describe_figure", "describe_diagram", "summarize_page", "interpret_table",
                     "examine_scanned_page", "extract_visible_labels"]
TASK_PROMPTS: dict[str, str] = {
    "describe_figure": "Describe the visible figure and note uncertainty.",
    "describe_diagram": "Describe the visible diagram and its labels without inventing connections.",
    "summarize_page": "Summarize the supplied page image's visible content.",
    "interpret_table": "Describe visible table values; state when a cell is unreadable.",
    "examine_scanned_page": "Describe the scanned page's visible content; do not claim OCR certainty.",
    "extract_visible_labels": "List visible labels as tentative observations; do not guess missing text.",
}


def _provider(settings: Settings) -> VisionProvider:
    return OllamaVisionProvider(base_url=settings.ollama_base_url, model_name=settings.vision_model,
                                max_response_bytes=settings.vision_max_response_bytes)


def _analyze_pdf(settings: Settings, *, filename: str | None, content_type: str | None,
                 content: bytes, page_number: int, region_values: tuple[float | None, ...],
                 task: VisionTask) -> VisualEvidence:
    region = _region(*region_values)
    safe_name, _ = validate_upload_metadata(filename, content_type)
    rendered = render_pdf(filename=filename, content_type=content_type, content=content,
                          page_number=page_number, region=region,
                          max_size_bytes=settings.max_upload_size_bytes,
                          max_pdf_pages=settings.max_pdf_pages,
                          max_pixels=settings.vision_max_image_pixels)
    provider = _provider(settings)
    try:
        output = provider.analyze(image=rendered.png, task_prompt=TASK_PROMPTS[task])
        return evidence_for_pdf(document_id=sha256(content).hexdigest(), filename=safe_name,
                                page_number=page_number, region=rendered.region,
                                task=task, model_name=provider.model_name, output=output)
    finally:
        provider.close()


def _analyze_image(settings: Settings, *, filename: str | None, content_type: str | None,
                   content: bytes, task: VisionTask) -> VisualEvidence:
    safe_name, image_type, width, height = safe_image_upload(
        filename, content_type, content, max_pixels=settings.vision_max_image_pixels,
        max_size_bytes=settings.max_upload_size_bytes,
    )
    provider = _provider(settings)
    try:
        output = provider.analyze(image=content, task_prompt=TASK_PROMPTS[task])
        return evidence_for_image(content=content, filename=safe_name, image_type=image_type,
                                  width=width, height=height, task=task,
                                  model_name=provider.model_name, output=output)
    finally:
        provider.close()


def _http_error(exc: DocumentParseError | VisionError) -> HTTPException:
    if isinstance(exc, DocumentParseError):
        return HTTPException(status_code=exc.status_code,
                             detail={"code": exc.code, "message": str(exc)})
    # Log only the fixed internal category; model output, source text, and
    # transport exception details must never enter logs or the HTTP response.
    logger.warning("Local visual analysis failed: %s", exc.category)
    return HTTPException(status_code=502,
                         detail={"code": "vision_failed", "message": "Local visual analysis failed."})


@pdf_router.post("/analyze-visual", response_model=VisualEvidence)
async def analyze_pdf_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="PDF document")],
    page_number: Annotated[int, Form(ge=1)],
    task: Annotated[VisionTask, Form()] = "summarize_page",
    x0: Annotated[float | None, Form()] = None,
    y0: Annotated[float | None, Form()] = None,
    x1: Annotated[float | None, Form()] = None,
    y1: Annotated[float | None, Form()] = None,
) -> VisualEvidence:
    settings: Settings = request.app.state.settings
    try:
        content = await file.read(settings.max_upload_size_bytes + 1)
        return await run_in_threadpool(_analyze_pdf, settings, filename=file.filename,
                                       content_type=file.content_type, content=content,
                                       page_number=page_number, region_values=(x0, y0, x1, y1), task=task)
    except (DocumentParseError, VisionError) as exc:
        raise _http_error(exc) from exc
    finally:
        await file.close()


@image_router.post("/analyze", response_model=VisualEvidence)
async def analyze_image_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="PNG or JPEG image")],
    task: Annotated[VisionTask, Form()] = "describe_figure",
) -> VisualEvidence:
    settings: Settings = request.app.state.settings
    try:
        content = await file.read(min(settings.max_upload_size_bytes, 4_000_000) + 1)
        return await run_in_threadpool(_analyze_image, settings, filename=file.filename,
                                       content_type=file.content_type, content=content, task=task)
    except (DocumentParseError, VisionError) as exc:
        raise _http_error(exc) from exc
    finally:
        await file.close()
