"""Synthetic PDF image-inspection and development-gate tests."""

import asyncio
from hashlib import sha256

import httpx
import pymupdf
import pytest
from pydantic import ValidationError

from rag_ai.core.config import Settings
from rag_ai.documents.parser import DocumentParseError
from rag_ai.documents.visual import MAX_REPORTED_IMAGE_OCCURRENCES, VisualInspection, inspect_pdf
from rag_ai.main import create_app


def pdf_bytes(*, page_count: int = 1, image_page: int | None = None,
              image_occurrences: int = 1, text: bool = True,
              image_rect: tuple[int, int, int, int] = (80, 100, 160, 180),
              rotation: int = 0) -> bytes:
    with pymupdf.open() as document:
        pixel = pymupdf.Pixmap(pymupdf.csRGB, 1, 1, bytes([0, 0, 0]), False)
        for page_number in range(1, page_count + 1):
            page = document.new_page(width=300, height=400)
            if text:
                page.insert_text((72, 72), f"Synthetic page {page_number}")
            if image_page == page_number:
                for _ in range(image_occurrences):
                    page.insert_image(pymupdf.Rect(*image_rect), pixmap=pixel)
            if rotation:
                page.set_rotation(rotation)
        return document.tobytes()


def inspect(content: bytes, *, filename: str = "sample.pdf", max_size: int = 1_000_000,
            max_pages: int = 500) -> VisualInspection:
    return inspect_pdf(
        filename=filename, content_type="application/pdf", content=content,
        max_size_bytes=max_size, max_pdf_pages=max_pages,
    )


def post_file(content: bytes, settings: Settings, filename: str = "sample.pdf") -> httpx.Response:
    async def request() -> httpx.Response:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(settings)), base_url="http://test",
        ) as client:
            return await client.post(
                "/documents/inspect", files={"file": (filename, content, "application/pdf")},
            )

    return asyncio.run(request())


def test_text_only_pdf_has_no_images() -> None:
    content = pdf_bytes()
    result = inspect(content, filename="../sample.pdf")
    assert result.filename == "sample.pdf"
    assert result.document_id == sha256(content).hexdigest()
    assert result.page_count == 1
    assert result.pages[0].page_number == 1
    assert (result.pages[0].width, result.pages[0].height) == (300, 400)
    assert result.pages[0].has_text is True
    assert result.pages[0].image_count == 0
    assert result.pages[0].image_bounds == []


def test_image_occurrence_has_valid_bounds() -> None:
    page = inspect(pdf_bytes(image_page=1)).pages[0]
    assert page.image_count == 1
    box = page.image_bounds[0]
    assert (box.x0, box.y0, box.x1, box.y1) == (80, 100, 160, 180)
    assert 0 <= box.x0 < box.x1 <= page.width
    assert 0 <= box.y0 < box.y1 <= page.height


@pytest.mark.parametrize("invalid_box", [
    (float("nan"), 100, 160, 180),
    (160, 100, 80, 180),
])
def test_invalid_image_bounds_are_rejected(
    invalid_box: tuple[float, float, float, float], monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pymupdf.Page, "get_image_info",
        lambda self, **kwargs: [{"bbox": invalid_box}],
    )
    with pytest.raises(DocumentParseError) as error:
        inspect(pdf_bytes())
    assert error.value.code == "invalid_pdf"


def test_partially_off_page_image_is_clipped() -> None:
    page = inspect(pdf_bytes(image_page=1, image_rect=(-20, 100, 80, 180))).pages[0]
    assert page.image_count == 1
    assert len(page.image_bounds) == 1
    box = page.image_bounds[0]
    assert (box.x0, box.y0, box.x1, box.y1) == (0, 100, 80, 180)
    assert 0 <= box.x0 < box.x1 <= page.width
    assert 0 <= box.y0 < box.y1 <= page.height


def test_completely_off_page_image_is_counted_without_a_box() -> None:
    page = inspect(pdf_bytes(image_page=1, image_rect=(340, 100, 420, 180))).pages[0]
    assert page.image_count == 1
    assert page.image_bounds == []


@pytest.mark.parametrize("rotation,expected_size,expected_box", [
    (90, (400, 300), (220, 80, 300, 160)),
    (180, (300, 400), (140, 220, 220, 300)),
])
def test_rotated_page_uses_displayed_coordinates(
    rotation: int, expected_size: tuple[int, int], expected_box: tuple[int, int, int, int],
) -> None:
    page = inspect(pdf_bytes(image_page=1, rotation=rotation)).pages[0]
    assert (page.width, page.height) == expected_size
    assert page.image_count == 1
    box = page.image_bounds[0]
    assert (box.x0, box.y0, box.x1, box.y1) == expected_box
    assert 0 <= box.x0 < box.x1 <= page.width
    assert 0 <= box.y0 < box.y1 <= page.height


def test_two_pages_keep_image_on_its_source_page() -> None:
    result = inspect(pdf_bytes(page_count=2, image_page=2))
    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert [page.image_count for page in result.pages] == [0, 1]


def test_image_only_pdf_is_inspectable() -> None:
    result = inspect(pdf_bytes(image_page=1, text=False))
    assert result.pages[0].has_text is False
    assert result.pages[0].image_count == 1


@pytest.mark.parametrize("content", [b"not a PDF", b"%PDF-1.7\ninvalid", b""])
def test_malformed_pdf_is_rejected(content: bytes) -> None:
    with pytest.raises(DocumentParseError) as error:
        inspect(content)
    assert error.value.code == "invalid_pdf"


def test_oversized_pdf_is_rejected() -> None:
    content = pdf_bytes()
    with pytest.raises(DocumentParseError) as error:
        inspect(content, max_size=len(content) - 1)
    assert error.value.status_code == 413


def test_page_limit_is_enforced() -> None:
    with pytest.raises(DocumentParseError) as error:
        inspect(pdf_bytes(page_count=2), max_pages=1)
    assert error.value.code == "too_many_pages"


def test_unsupported_upload_is_rejected() -> None:
    with pytest.raises(DocumentParseError) as error:
        inspect(pdf_bytes(), filename="sample.txt")
    assert error.value.status_code == 415


def test_password_protected_pdf_is_rejected() -> None:
    with pymupdf.open() as document:
        document.new_page()
        content = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="synthetic-owner", user_pw="synthetic-user",
        )
    with pytest.raises(DocumentParseError) as error:
        inspect(content)
    assert error.value.code == "invalid_pdf"


def test_image_occurrence_limit_is_enforced() -> None:
    content = pdf_bytes(image_page=1, image_occurrences=MAX_REPORTED_IMAGE_OCCURRENCES + 1)
    with pytest.raises(DocumentParseError) as error:
        inspect(content)
    assert error.value.code == "too_many_images"
    assert error.value.status_code == 413


def test_disabled_endpoint_returns_404() -> None:
    settings = Settings(app_env="development", enable_document_visual_inspection_endpoint=False,
                        _env_file=None)
    assert "/documents/inspect" not in create_app(settings).openapi()["paths"]
    assert post_file(pdf_bytes(), settings).status_code == 404


def test_endpoint_is_unavailable_in_test_environment() -> None:
    settings = Settings(app_env="test", enable_document_visual_inspection_endpoint=True,
                        _env_file=None)
    assert "/documents/inspect" not in create_app(settings).openapi()["paths"]


def test_development_opt_in_returns_metadata_only() -> None:
    settings = Settings(app_env="development", enable_document_visual_inspection_endpoint=True,
                        _env_file=None)
    response = post_file(pdf_bytes(image_page=1), settings)
    assert response.status_code == 200
    assert response.json()["pages"][0]["image_count"] == 1
    assert "Synthetic page" not in response.text
    assert "image_bytes" not in response.text


def test_production_opt_in_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Document visual inspection cannot be enabled in production"):
        Settings(app_env="production", enable_document_visual_inspection_endpoint=True,
                 _env_file=None)
