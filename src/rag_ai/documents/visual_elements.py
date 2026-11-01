"""Deterministic, bounded source-region records derived from Phase 8 inspection."""

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from hashlib import sha256
import json
from typing import Literal
from uuid import UUID
from datetime import datetime

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rag_ai.documents.parser import DocumentParseError
from rag_ai.documents.visual import ImageBounds, inspect_pdf

EXTRACTOR_NAME = "pymupdf_visual_inspection"
EXTRACTOR_VERSION = "1"
COORDINATE_SYSTEM = "displayed_pdf_points"
MAX_VISUAL_ELEMENTS = 1120
MAX_METADATA_BYTES = 16000
_QUANTUM = Decimal("0.0001")


def _coordinate(value: float | Decimal) -> Decimal:
    try:
        result = Decimal(str(value))
        if not result.is_finite():
            raise ValueError("Coordinate must be finite.")
        return result.quantize(_QUANTUM, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Invalid visual coordinate.") from exc


class VisualElementCandidate(BaseModel):
    """One visible source occurrence; metadata is extraction, never model output."""

    model_config = ConfigDict(extra="forbid", strict=True)

    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    element_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(ge=1)
    element_type: Literal["raster_occurrence", "table", "vector_path"]
    occurrence_index: int = Field(ge=0)
    extractor_name: str = Field(min_length=1, max_length=80)
    extractor_version: str = Field(min_length=1, max_length=80)
    coordinate_system: Literal["displayed_pdf_points"] = COORDINATE_SYSTEM
    page_width: Decimal
    page_height: Decimal
    page_rotation: Literal[0, 90, 180, 270]
    x0: Decimal
    y0: Decimal
    x1: Decimal
    y1: Decimal
    extraction_metadata: dict[str, object]

    @field_validator("page_width", "page_height", "x0", "y0", "x1", "y1", mode="before")
    @classmethod
    def normalize_coordinate(cls, value: object) -> Decimal:
        if not isinstance(value, (float, int, str, Decimal)) or isinstance(value, bool):
            raise ValueError("Invalid visual coordinate.")
        return _coordinate(value)

    @model_validator(mode="after")
    def validate_geometry_and_metadata(self) -> "VisualElementCandidate":
        if not (self.page_width > 0 and self.page_height > 0
                and 0 <= self.x0 < self.x1 <= self.page_width
                and 0 <= self.y0 < self.y1 <= self.page_height):
            raise ValueError("Visual bounds must be inside the displayed page.")
        encoded = json.dumps(self.extraction_metadata, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_METADATA_BYTES:
            raise ValueError("Visual extraction metadata exceeds its limit.")
        return self


class StoredVisualElement(VisualElementCandidate):
    collection_id: UUID
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    updated_at: datetime


class VisualIndexResult(BaseModel):
    collection_id: UUID
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    extractor_name: str
    extractor_version: str
    element_count: int = Field(ge=0, le=MAX_VISUAL_ELEMENTS)


def element_identity(candidate: VisualElementCandidate) -> str:
    """Hash exactly the normalized coordinates persisted in NUMERIC(12,4)."""
    values = [candidate.source_sha256, candidate.extractor_name, candidate.extractor_version,
              candidate.page_number, candidate.element_type, candidate.occurrence_index,
              *[format(getattr(candidate, field), ".4f") for field in ("x0", "y0", "x1", "y1")]]
    return sha256(json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def extract_visual_elements(*, filename: str, content: bytes, max_size_bytes: int,
                            max_pdf_pages: int, extractor_version: str = EXTRACTOR_VERSION,
                            expected_page_count: int | None = None,
                            ) -> list[VisualElementCandidate]:
    """Adapt the existing inspector; visible occurrence order is deterministic per page/type."""
    if not extractor_version or len(extractor_version) > 80:
        raise ValueError("Invalid extractor version.")
    inspected = inspect_pdf(filename=filename, content_type="application/pdf", content=content,
                            max_size_bytes=max_size_bytes, max_pdf_pages=max_pdf_pages)
    if expected_page_count is not None and inspected.page_count != expected_page_count:
        raise DocumentParseError("invalid_stored_source", "Stored PDF page metadata is inconsistent.", 409)
    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            rotations = [page.rotation for page in pdf]
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as exc:
        raise DocumentParseError("invalid_pdf", "The stored PDF is invalid.", 422) from exc
    candidates: list[VisualElementCandidate] = []
    for page in inspected.pages:
        groups: tuple[tuple[str, list[tuple[ImageBounds, dict[str, object]]]], ...] = (
            ("raster_occurrence", [(box, {"method": "pymupdf_image_occurrence"})
                                   for box in page.image_bounds]),
            ("table", [(table.bounds, {"method": "pymupdf_find_tables_heuristic",
                                        "row_count": table.row_count, "column_count": table.column_count,
                                        "cells": table.cells}) for table in page.tables]),
            ("vector_path", [(box, {"method": "pymupdf_vector_path"})
                             for box in page.drawing_bounds]),
        )
        for element_type, entries in groups:
            # Same-box occurrences retain separate ordinal identities. Sorting
            # makes IDs independent of the order PyMuPDF reports distinct boxes.
            entries.sort(key=lambda item: (item[0].x0, item[0].y0, item[0].x1, item[0].y1,
                                           json.dumps(item[1], sort_keys=True)))
            for occurrence_index, (box, metadata) in enumerate(entries):
                data = dict(source_sha256=inspected.document_id, element_id="0" * 64,
                            page_number=page.page_number, element_type=element_type,
                            occurrence_index=occurrence_index, extractor_name=EXTRACTOR_NAME,
                            extractor_version=extractor_version, page_width=page.width,
                            page_height=page.height, page_rotation=rotations[page.page_number - 1],
                            x0=box.x0, y0=box.y0, x1=box.x1, y1=box.y1,
                            extraction_metadata=metadata)
                try:
                    candidate = VisualElementCandidate.model_validate(data)
                except (ValidationError, TypeError, ValueError) as exc:
                    raise DocumentParseError("invalid_pdf", "The stored PDF has invalid visual geometry.", 422) from exc
                candidates.append(candidate.model_copy(update={"element_id": element_identity(candidate)}))
                if len(candidates) > MAX_VISUAL_ELEMENTS:
                    raise DocumentParseError("too_many_visual_elements", "The PDF exceeds the visual element limit.", 413)
    return candidates
