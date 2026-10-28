"""API smoke tests."""

import asyncio

import httpx

from rag_ai.main import app


def get_response(path: str) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(request())


def test_health_returns_ok() -> None:
    response = get_response("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_documents_health() -> None:
    response = get_response("/openapi.json")

    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
