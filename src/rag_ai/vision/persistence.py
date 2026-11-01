"""Render one stored visual source and persist a local model interpretation."""

from decimal import Decimal
from hashlib import sha256
from uuid import UUID

import pymupdf

from rag_ai.api.vision import TASK_PROMPTS, VisionTask
from rag_ai.core.config import Settings
from rag_ai.documents.render import render_pdf
from rag_ai.documents.visual import ImageBounds
from rag_ai.storage.interpretations import (
    InterpretationConflict, PostgresInterpretationStore, StoredInterpretation,
)
from rag_ai.storage.visual import PostgresVisualStore
from rag_ai.vision.ollama import OllamaVisionProvider

PROMPT_VERSION = "stored_visual_v1"
RENDER_SCALE = 1.0


def analyze_stored_element(*, settings: Settings, store: PostgresInterpretationStore,
                           collection_id: UUID, document_id: str, element_id: str,
                           task: VisionTask, request_key: UUID | None) -> StoredInterpretation:
    """Keep rendering and inference outside the final database transaction."""
    if request_key is not None:
        prior = store.get(collection_id=collection_id, document_id=document_id,
                          element_id=element_id, request_key=request_key)
        if prior is not None:
            if prior.analysis_task != task or prior.prompt_version != PROMPT_VERSION:
                raise InterpretationConflict("Request key was used for a different task.")
            return prior
    element = store.load_element(collection_id=collection_id, document_id=document_id,
                                 element_id=element_id)
    visual_store = PostgresVisualStore(store.dsn)
    filename, content, page_count = visual_store.load_pdf(
        collection_id=collection_id, document_id=document_id,
        max_size_bytes=settings.max_upload_size_bytes,
    )
    if (page_count > settings.max_pdf_pages or element.page_number > page_count
            or element.source_sha256 != document_id):
        raise InterpretationConflict("Stored visual source is inconsistent.")
    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            page = pdf[element.page_number - 1]
            actual = (Decimal(str(page.rect.width)).quantize(Decimal("0.0001")),
                      Decimal(str(page.rect.height)).quantize(Decimal("0.0001")))
            if (actual != (element.page_width, element.page_height)
                    or page.rotation != element.page_rotation):
                raise InterpretationConflict("Stored visual geometry changed.")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, IndexError, ValueError) as exc:
        raise InterpretationConflict("Stored visual geometry is invalid.") from exc
    region = ImageBounds(x0=float(element.x0), y0=float(element.y0),
                         x1=float(element.x1), y1=float(element.y1))
    rendered = render_pdf(filename=filename, content_type="application/pdf", content=content,
                          page_number=element.page_number, region=region,
                          max_size_bytes=settings.max_upload_size_bytes,
                          max_pdf_pages=settings.max_pdf_pages,
                          scale=RENDER_SCALE, max_pixels=settings.vision_max_image_pixels)
    provider = OllamaVisionProvider(base_url=settings.ollama_base_url, model_name=settings.vision_model,
                                    max_response_bytes=settings.vision_max_response_bytes)
    try:
        output, digest = provider.analyze_verified(
            image=rendered.png, task_prompt=TASK_PROMPTS[task],
        )
    finally:
        provider.close()
    return store.insert(element=element, source_sha256=document_id, page_count=page_count,
                        request_key=request_key, task=task, prompt_version=PROMPT_VERSION,
                        model_name=provider.model_name, model_digest=digest,
                        render_scale=RENDER_SCALE,
                        rendered_image_sha256=sha256(rendered.png).hexdigest(), output=output)
