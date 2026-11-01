"""Synthetic source occurrence, identity, and route-boundary checks."""

from hashlib import sha256
from uuid import uuid4

import pymupdf
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rag_ai.core.config import Settings
from rag_ai.documents.visual_elements import (
    VisualElementCandidate, element_identity, extract_visual_elements,
)
from rag_ai.documents.visual import ImageBounds, VisualInspection, VisualPage, VisualTable, DrawingKinds
from rag_ai.main import create_app


def _pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page(width=300, height=200)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), False)
    pix.clear_with(255)
    for _ in range(2):
        page.insert_image(pymupdf.Rect(30, 40, 70, 80), pixmap=pix)
    page.draw_rect(pymupdf.Rect(90, 40, 130, 80))
    page.set_rotation(90)
    content = pdf.tobytes()
    pdf.close()
    return content


def _extract(content: bytes, version: str = "1") -> list[VisualElementCandidate]:
    return extract_visual_elements(filename="synthetic.pdf", content=content,
                                   max_size_bytes=100_000, max_pdf_pages=3,
                                   extractor_version=version)


def test_repeated_rasters_and_rotation_have_distinct_stable_ids() -> None:
    content = _pdf()
    first = _extract(content)
    second = _extract(content)
    assert [item.element_id for item in first] == [item.element_id for item in second]
    rasters = [item for item in first if item.element_type == "raster_occurrence"]
    assert len(rasters) == 2
    assert {item.occurrence_index for item in rasters} == {0, 1}
    assert len({item.element_id for item in rasters}) == 2
    assert len({(item.x0, item.y0, item.x1, item.y1) for item in rasters}) == 1
    assert all(item.page_rotation == 90 and item.page_width == 200 and item.page_height == 300
               for item in first)
    assert any(item.element_type == "vector_path" for item in first)
    changed = _extract(content, "2")
    assert [item.element_id for item in first] != [item.element_id for item in changed]


def test_normalized_geometry_is_identity_geometry() -> None:
    data = dict(source_sha256=sha256(b"source").hexdigest(), element_id="0" * 64,
                page_number=1, element_type="raster_occurrence", occurrence_index=0,
                extractor_name="test", extractor_version="1", page_width="100",
                page_height="100", page_rotation=0, x0="1.00001", y0="2",
                x1="5.00001", y1="6", extraction_metadata={"method": "synthetic"})
    first = VisualElementCandidate.model_validate(data)
    second = VisualElementCandidate.model_validate({**data, "x0": "1.00002", "x1": "5.00002"})
    assert first.x0 == second.x0
    assert element_identity(first) == element_identity(second)
    for bad in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(ValidationError):
            VisualElementCandidate.model_validate({**data, "x0": bad})
    for override in ({"x1": "0"}, {"page_width": "0"}, {"page_rotation": 45},
                     {"source_sha256": "forged"}, {"x1": "101"}):
        with pytest.raises(ValidationError):
            VisualElementCandidate.model_validate({**data, **override})


def test_adapter_keeps_overlapping_types_and_only_visible_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    content = _pdf()
    document_id = sha256(content).hexdigest()
    box = ImageBounds(x0=10, y0=20, x1=30, y1=40)
    inspection = VisualInspection(document_id=document_id, filename="synthetic.pdf", page_count=1,
        pages=[VisualPage(page_number=1, width=200, height=300, has_text=False,
                          image_count=3, image_bounds=[box, box], drawing_count=2,
                          drawing_bounds=[box], drawing_kinds=DrawingKinds(),
                          table_count=1, tables=[VisualTable(bounds=box, row_count=1,
                                                               column_count=1, cells=[["synthetic"]])])])
    monkeypatch.setattr("rag_ai.documents.visual_elements.inspect_pdf", lambda **kwargs: inspection)
    records = _extract(content)
    assert len(records) == 4  # Third off-page image has no visible box.
    assert {record.element_type for record in records} == {"raster_occurrence", "table", "vector_path"}
    assert len({record.element_id for record in records}) == 4
    assert next(record for record in records if record.element_type == "table").extraction_metadata["cells"] == [["synthetic"]]


def test_visual_storage_routes_are_separately_gated() -> None:
    path = f"/collections/{uuid4()}/documents/{'a' * 64}/visual-elements"
    for environment in ("production", "test", "development"):
        app = create_app(Settings(app_env=environment, _env_file=None))
        with TestClient(app) as client:
            assert client.get(path).status_code == 404
            assert client.post(path + "/index").status_code == 404
    with pytest.raises(ValidationError, match="Visual element storage cannot be enabled in production"):
        Settings(app_env="production", enable_visual_element_storage_endpoint=True, _env_file=None)
    app = create_app(Settings(app_env="development", enable_visual_element_storage_endpoint=True,
                              _env_file=None))
    with TestClient(app) as client:
        assert client.get(path).status_code == 503
        assert client.post(path + "/index").status_code == 503
