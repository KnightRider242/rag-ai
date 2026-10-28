"""Security checks for development-only route registration."""

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from rag_ai.core.config import Settings
from rag_ai.main import create_app


def request_parse(settings: Settings) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(settings))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/documents/parse", files={"file": ("sample.txt", b"sample", "text/plain")}
            )

    return asyncio.run(request())


def test_parse_endpoint_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENABLE_DOCUMENT_PARSE_ENDPOINT", raising=False)
    settings = Settings(_env_file=None)
    app = create_app(settings)

    assert settings.app_env == "production"
    assert settings.enable_document_parse_endpoint is False
    assert "/documents/parse" not in app.openapi()["paths"]
    assert request_parse(settings).status_code == 404


def test_parse_endpoint_enabled_explicitly_in_development() -> None:
    settings = Settings(app_env="development", enable_document_parse_endpoint=True)
    app = create_app(settings)

    assert "/documents/parse" in app.openapi()["paths"]
    assert request_parse(settings).status_code == 200


def test_parse_endpoint_unavailable_in_test_environment() -> None:
    settings = Settings(app_env="test", enable_document_parse_endpoint=True)
    app = create_app(settings)

    assert "/documents/parse" not in app.openapi()["paths"]
    assert request_parse(settings).status_code == 404


def test_parse_endpoint_unavailable_in_production() -> None:
    settings = Settings(app_env="production", enable_document_parse_endpoint=False)
    app = create_app(settings)

    assert "/documents/parse" not in app.openapi()["paths"]
    assert request_parse(settings).status_code == 404


def test_production_configuration_cannot_enable_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("ENABLE_DOCUMENT_PARSE_ENDPOINT", "true")

    with pytest.raises(ValidationError, match="Document parsing cannot be enabled in production"):
        Settings(_env_file=None)
