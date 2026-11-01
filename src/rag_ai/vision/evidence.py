"""Application-owned provenance for local visual observations."""

from hashlib import sha256
from typing import Literal

import pymupdf
from pydantic import BaseModel, Field

from rag_ai.documents.parser import DocumentParseError
from rag_ai.documents.visual import ImageBounds
from rag_ai.vision.base import VisionOutput

MAX_VISION_IMAGE_BYTES = 4_000_000


def _image_dimensions(content: bytes, kind: str) -> tuple[int, int]:
    # Read dimensions before native pixel decoding, so a small compressed
    # image cannot request an unbounded allocation through its header.
    if kind == "png":
        if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n" or content[12:16] != b"IHDR":
            raise ValueError("invalid image header")
        if not content.endswith(b"\x00\x00\x00\x00IEND\xaeB\x60\x82"):
            raise ValueError("incomplete png image")
        return int.from_bytes(content[16:20], "big"), int.from_bytes(content[20:24], "big")
    if not content.startswith(b"\xff\xd8") or not content.endswith(b"\xff\xd9"):
        raise ValueError("invalid image header")
    position = 2
    while position + 4 <= len(content):
        if content[position] != 0xff:
            raise ValueError("invalid jpeg marker")
        while position < len(content) and content[position] == 0xff:
            position += 1
        if position >= len(content):
            raise ValueError("invalid jpeg marker")
        marker = content[position]
        position += 1
        if marker in {0x01, 0xd8, 0xd9}:
            continue
        if marker == 0xda:
            break
        segment = int.from_bytes(content[position:position + 2], "big")
        if segment < 2 or position + segment > len(content):
            raise ValueError("invalid jpeg segment")
        if marker in {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}:
            if segment < 7:
                raise ValueError("invalid jpeg frame")
            return (int.from_bytes(content[position + 5:position + 7], "big"),
                    int.from_bytes(content[position + 3:position + 5], "big"))
        position += segment
    raise ValueError("jpeg dimensions unavailable")


class VisualEvidence(BaseModel):
    source_id: str
    document_id: str
    filename: str
    source_type: Literal["pdf", "png", "jpeg"]
    page_number: int | None
    region: ImageBounds | None = Field(description="Displayed PDF points or native image pixels.")
    visual_element_type: Literal["page_region", "standalone_image"]
    extraction_method: Literal["local_ollama_vision"] = "local_ollama_vision"
    evidence_kind: Literal["model_generated_interpretation"] = "model_generated_interpretation"
    analysis_task: str
    model_name: str
    description: str
    observations: list[str]
    limitations: list[str]


def safe_image_upload(filename: str | None, content_type: str | None, content: bytes,
                      *, max_pixels: int, max_size_bytes: int) -> tuple[str, str, int, int]:
    """Decode bounded PNG/JPEG bytes after checking claimed and actual type."""
    safe_name = (filename or "").replace("\\", "/").split("/")[-1]
    if (not safe_name or len(safe_name) > 255 or any(ord(c) < 32 or ord(c) == 127 for c in safe_name)):
        raise DocumentParseError("invalid_filename", "The image filename is invalid.", 422)
    extension = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    expected = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(extension)
    if expected is None or (content_type or "").split(";", 1)[0].lower() != expected:
        raise DocumentParseError("unsupported_file_type", "Only PNG and JPEG uploads are supported.", 415)
    if not content or len(content) > min(max_size_bytes, MAX_VISION_IMAGE_BYTES):
        raise DocumentParseError("file_too_large", "The image exceeds the byte limit or is empty.", 413)
    try:
        kind = "jpeg" if extension in {"jpg", "jpeg"} else "png"
        width, height = _image_dimensions(content, kind)
        if width <= 0 or height <= 0 or width > 2048 or height > 2048 or width * height > max_pixels:
            raise DocumentParseError("image_too_large", "The image exceeds the pixel limit.", 413)
        try:
            pixmap = pymupdf.Pixmap(content)
        except Exception as exc:
            # Native MuPDF decode errors do not consistently subclass its
            # public FileDataError; keep this untrusted-input boundary safe.
            raise DocumentParseError("invalid_image", "The uploaded image is invalid.", 422) from exc
        if pixmap.width != width or pixmap.height != height:
            raise DocumentParseError("invalid_image", "The image geometry is invalid.", 422)
        return safe_name, kind, width, height
    except DocumentParseError:
        raise
    except (RuntimeError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise DocumentParseError("invalid_image", "The uploaded image is invalid.", 422) from exc


def evidence_for_pdf(*, document_id: str, filename: str, page_number: int,
                     region: ImageBounds, task: str, model_name: str,
                     output: VisionOutput) -> VisualEvidence:
    region_key = f"{region.x0}:{region.y0}:{region.x1}:{region.y1}"
    source_id = sha256(f"{document_id}:{page_number}:{region_key}".encode()).hexdigest()
    return VisualEvidence(source_id=source_id, document_id=document_id, filename=filename,
                          source_type="pdf", page_number=page_number, region=region,
                          visual_element_type="page_region", analysis_task=task,
                          model_name=model_name, **output.model_dump())


def evidence_for_image(*, content: bytes, filename: str, image_type: str,
                       width: int, height: int, task: str, model_name: str,
                       output: VisionOutput) -> VisualEvidence:
    document_id = sha256(content).hexdigest()
    return VisualEvidence(source_id=document_id, document_id=document_id, filename=filename,
                          source_type=image_type, page_number=None,
                          region=ImageBounds(x0=0, y0=0, x1=width, y1=height),
                          visual_element_type="standalone_image", analysis_task=task,
                          model_name=model_name, **output.model_dump())
