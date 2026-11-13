"""Synthetic vector, table, and preview checks without local services."""

import pymupdf
import pytest
from types import SimpleNamespace

from rag_ai.documents.parser import DocumentParseError
from rag_ai.documents.render import render_pdf
from rag_ai.documents.visual import ImageBounds, inspect_pdf


def inspect(content: bytes):
    return inspect_pdf(filename="synthetic.pdf", content_type="application/pdf", content=content,
                       max_size_bytes=1_000_000, max_pdf_pages=10)


def pdf(*, rotation: int = 0, table: bool = False, off_page: bool = False) -> bytes:
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=300)
        if table:
            for y in (50, 80, 110):
                page.draw_line((50, y), (250, y))
            for x in (50, 150, 250):
                page.draw_line((x, 50), (x, 110))
            for text, point in (("A", (60, 70)), ("B", (160, 70)), ("C", (60, 100))):
                page.insert_text(point, text)
        else:
            page.draw_rect(pymupdf.Rect(340 if off_page else 30, 30,
                                        380 if off_page else 70, 70), color=(0, 0, 0))
        page.set_rotation(rotation)
        return doc.tobytes()


def colored_pdf(rotation: int) -> tuple[bytes, pymupdf.Rect, pymupdf.Rect]:
    """Return distinguishable regions in displayed coordinates."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=400)
        red = pymupdf.Rect(40, 50, 90, 100)
        blue = pymupdf.Rect(180, 250, 230, 300)
        page.draw_rect(red, color=None, fill=(1, 0, 0))
        page.draw_rect(blue, color=None, fill=(0, 0, 1))
        page.set_rotation(rotation)
        red_inner = pymupdf.Rect(50, 60, 80, 90) * page.rotation_matrix
        blue_inner = pymupdf.Rect(190, 260, 220, 290) * page.rotation_matrix
        return doc.tobytes(), red_inner, blue_inner


def render(content: bytes, region: pymupdf.Rect | None = None, **limits):
    bounds = ImageBounds(x0=region.x0, y0=region.y0, x1=region.x1, y1=region.y1) if region else None
    return render_pdf(filename="synthetic.pdf", content_type="application/pdf", content=content,
                      page_number=1, region=bounds, max_size_bytes=1_000_000,
                      max_pdf_pages=10, **limits)


def center_pixel(png: bytes) -> tuple[int, ...]:
    pixmap = pymupdf.Pixmap(png)
    return pixmap.pixel(pixmap.width // 2, pixmap.height // 2)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_displayed_region_crops_actual_colored_pixels(rotation: int) -> None:
    content, red, blue = colored_pdf(rotation)
    red_crop = render(content, red)
    blue_crop = render(content, blue)
    assert center_pixel(red_crop.png) == (255, 0, 0)
    assert center_pixel(blue_crop.png) == (0, 0, 255)
    assert red_crop.region == ImageBounds(x0=red.x0, y0=red.y0, x1=red.x1, y1=red.y1)

    full = render(content)
    full_pixmap = pymupdf.Pixmap(full.png)
    assert full_pixmap.pixel(int((red.x0 + red.x1) / 2), int((red.y0 + red.y1) / 2)) == (255, 0, 0)
    assert full_pixmap.pixel(int((blue.x0 + blue.x1) / 2), int((blue.y0 + blue.y1) / 2)) == (0, 0, 255)


def test_colored_region_still_obeys_region_and_pixel_limits() -> None:
    content, red, _ = colored_pdf(90)
    with pytest.raises(DocumentParseError) as invalid:
        render(content, pymupdf.Rect(-1, red.y0, red.x1, red.y1))
    assert invalid.value.code == "invalid_region"
    with pytest.raises(DocumentParseError) as large:
        render(content, red, max_pixels=100)
    assert large.value.code == "render_too_large"


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_vector_bounds_follow_displayed_rotation(rotation: int) -> None:
    page = inspect(pdf(rotation=rotation)).pages[0]
    assert page.drawing_count == 1
    assert len(page.drawing_bounds) == 1
    box = page.drawing_bounds[0]
    assert 0 <= box.x0 < box.x1 <= page.width
    assert 0 <= box.y0 < box.y1 <= page.height


def test_off_page_vector_counts_without_bounds() -> None:
    page = inspect(pdf(off_page=True)).pages[0]
    assert page.drawing_count == 1
    assert page.drawing_bounds == []


def test_clipped_vector_path_stays_within_displayed_page() -> None:
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=300)
        page.draw_rect(pymupdf.Rect(-20, 30, 50, 90))
        result = inspect(doc.tobytes()).pages[0]
    assert result.drawing_count == 1
    assert result.drawing_bounds[0].x0 == 0


def test_image_based_table_is_not_fabricated_as_cells() -> None:
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=300)
        pix = pymupdf.Pixmap(pymupdf.csRGB, 10, 10, bytes([0, 0, 0] * 100), False)
        page.insert_image(pymupdf.Rect(30, 30, 200, 200), pixmap=pix)
        result = inspect(doc.tobytes()).pages[0]
    assert result.image_count == 1
    assert result.table_count == 0
    assert result.tables == []


def test_bordered_table_extracts_only_present_cells() -> None:
    page = inspect(pdf(table=True)).pages[0]
    assert page.table_count == 1
    assert page.tables[0].row_count == 2
    assert page.tables[0].column_count == 2
    assert page.tables[0].cells == [["A", "B"], ["C", ""]]


def test_rotated_table_bounds_remain_on_displayed_page() -> None:
    page = inspect(pdf(table=True, rotation=90)).pages[0]
    assert page.table_count == 1
    box = page.tables[0].bounds
    assert 0 <= box.x0 < box.x1 <= page.width
    assert 0 <= box.y0 < box.y1 <= page.height


def test_borderless_table_is_heuristic_and_keeps_empty_cells() -> None:
    with pymupdf.open() as doc:
        page = doc.new_page()
        for point, value in [((50, 50), "Name"), ((200, 50), "State"),
                             ((50, 80), "Relay"), ((200, 80), "Open"),
                             ((50, 110), "Pump"), ((200, 110), "On")]:
            page.insert_text(point, value)
        result = inspect(doc.tobytes()).pages[0]
    assert result.table_count >= 1
    cells = result.tables[0].cells
    assert ["Relay", "Open"] in cells
    assert any(not any(cell for cell in row) for row in cells)


def test_excessive_paths_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    drawing = {"rect": (0, 0, 10, 10), "items": [("re",)], "type": "s"}
    monkeypatch.setattr(pymupdf.Page, "get_cdrawings", lambda self: [drawing] * 1001)
    with pytest.raises(DocumentParseError, match="vector path limit"):
        inspect(pdf())


def test_invalid_drawing_geometry_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pymupdf.Page, "get_cdrawings", lambda self: [
        {"rect": (float("nan"), 0, 10, 10), "items": [], "type": "s"}])
    with pytest.raises(DocumentParseError, match="invalid drawing geometry"):
        inspect(pdf())


def test_excessive_tables_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pymupdf.Page, "find_tables", lambda self, **kwargs:
                        SimpleNamespace(tables=[object()] * 21))
    with pytest.raises(DocumentParseError, match="table limit"):
        inspect(pdf())


def test_excessive_table_cells_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    table = SimpleNamespace(row_count=201, col_count=1)
    monkeypatch.setattr(pymupdf.Page, "find_tables", lambda self, **kwargs:
                        SimpleNamespace(tables=[table]))
    with pytest.raises(DocumentParseError, match="cell limit"):
        inspect(pdf())


def test_oversized_text_fallback_is_skipped_but_small_table_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = SimpleNamespace(row_count=52, col_count=9,
                                extract=lambda: (_ for _ in ()).throw(AssertionError("must not extract")))
    small = SimpleNamespace(row_count=1, col_count=2, bbox=(30, 30, 90, 50),
                            extract=lambda: [["A", "B"]])
    monkeypatch.setattr(pymupdf.Page, "find_tables", lambda self, **kwargs:
                        SimpleNamespace(tables=[oversized, small] if kwargs.get("strategy") == "text" else []))
    page = inspect(pdf()).pages[0]
    assert page.table_count == 2
    assert page.skipped_oversized_text_tables == 1
    assert len(page.tables) == 1
    assert page.tables[0].cells == [["A", "B"]]
    assert page.drawing_count == 1


def test_table_text_budget_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_ai.documents.visual as visual
    monkeypatch.setattr(visual, "MAX_TABLE_CHARACTERS", 1)
    with pytest.raises(DocumentParseError, match="table text exceeds"):
        inspect(pdf(table=True))


def test_tables_remain_on_their_source_pages() -> None:
    with pymupdf.open() as doc:
        doc.new_page()
        page = doc.new_page()
        for y in (50, 80, 110):
            page.draw_line((50, y), (250, y))
        for x in (50, 150, 250):
            page.draw_line((x, 50), (x, 110))
        for text, point in (("A", (60, 70)), ("B", (160, 70)),
                            ("C", (60, 100)), ("D", (160, 100))):
            page.insert_text(point, text)
        result = inspect(doc.tobytes())
    assert [page.table_count for page in result.pages] == [0, 1]
    assert result.pages[1].page_number == 2


def test_preview_is_one_page_png_and_region_is_bounded() -> None:
    content = pdf(rotation=90)
    result = render_pdf(filename="synthetic.pdf", content_type="application/pdf", content=content,
                        page_number=1, region=None, max_size_bytes=1_000_000, max_pdf_pages=10)
    assert result.png.startswith(b"\x89PNG")
    assert result.page_number == 1
    assert (result.width, result.height) == (300, 300)
    cropped = render_pdf(filename="synthetic.pdf", content_type="application/pdf", content=content,
                         page_number=1, region=type(result.region)(x0=0, y0=0, x1=100, y1=100),
                         max_size_bytes=1_000_000, max_pdf_pages=10)
    assert cropped.width == 100 and cropped.height == 100
    assert len(cropped.png) < len(result.png)


@pytest.mark.parametrize("page_number,scale", [(2, 1), (1, 10), (1, float("nan"))])
def test_preview_rejects_invalid_work(page_number: int, scale: float) -> None:
    with pytest.raises(DocumentParseError):
        render_pdf(filename="synthetic.pdf", content_type="application/pdf", content=pdf(),
                   page_number=page_number, region=None, scale=scale,
                   max_size_bytes=1_000_000, max_pdf_pages=10)


def test_preview_rejects_large_page_before_rasterizing() -> None:
    with pymupdf.open() as doc:
        doc.new_page(width=10000, height=10000)
        content = doc.tobytes()
    with pytest.raises(DocumentParseError, match="pixel limit"):
        render_pdf(filename="synthetic.pdf", content_type="application/pdf", content=content,
                   page_number=1, region=None, max_size_bytes=1_000_000, max_pdf_pages=10)
