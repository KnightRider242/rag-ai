"""Opt-in, synthetic PDF-to-cited-answer test through the local FastAPI routes."""

import asyncio
import json
import os
from collections.abc import Mapping
from time import perf_counter
from uuid import uuid4

import httpx
import pymupdf
import pytest
from psycopg import ProgrammingError
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from rag_ai.core.config import Settings
from rag_ai.main import create_app
from rag_ai.storage.migrate import MigrationRunner


_UNSAFE_PG_ENV = frozenset({
    "PGHOST", "PGHOSTADDR", "PGDATABASE", "PGPORT", "PGSERVICE", "PGSERVICEFILE",
    "PGSYSCONFDIR", "PGOPTIONS", "PGLOADBALANCEHOSTS", "PGTARGETSESSIONATTRS",
})


@pytest.fixture
def e2e_database_url(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> str:
    """Verify the opt-in destination before the shared fixture creates a schema."""
    if os.getenv("RAG_AI_RUN_OLLAMA_TESTS") != "1":
        pytest.skip("Set RAG_AI_RUN_OLLAMA_TESTS=1 for local Ollama integration.")
    base_url = os.getenv("RAG_AI_TEST_DATABASE_URL")
    if not base_url:
        pytest.skip("Set RAG_AI_TEST_DATABASE_URL for isolated PostgreSQL tests.")
    try:
        safe_url = _validated_test_conninfo(base_url, os.environ)
    except ValueError as exc:
        pytest.fail(str(exc))
    # The shared fixture reads this environment value before its first connect
    # and captures the same DSN for creation, migration, and teardown.
    monkeypatch.setenv("RAG_AI_TEST_DATABASE_URL", safe_url)
    return request.getfixturevalue("isolated_database_url")


def _validated_test_conninfo(base_url: str, environment: Mapping[str, str]) -> str:
    """Pin the exact test destination before the schema fixture can connect."""
    if any(name in environment for name in _UNSAFE_PG_ENV):
        raise ValueError("PostgreSQL routing environment settings are not allowed for this test.")
    try:
        destination = conninfo_to_dict(base_url)
    except (ProgrammingError, ValueError):
        raise ValueError("The dedicated test database URL is invalid.") from None
    if destination.get("service"):
        raise ValueError("PostgreSQL service configuration is not allowed for this test.")
    host = destination.get("host")
    hostaddr = destination.get("hostaddr")
    if (destination.get("dbname") != "rag_ai_retrieval_test"
            or host not in {"localhost", "127.0.0.1", "::1"}
            or hostaddr not in {None, "127.0.0.1", "::1"}
            or (host in {"127.0.0.1", "::1"} and hostaddr not in {None, host})):
        raise ValueError("Local end-to-end tests require the dedicated loopback test database.")
    try:
        port = int(destination.get("port") or 5432)
    except (TypeError, ValueError) as exc:
        raise ValueError("The dedicated test database port is invalid.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("The dedicated test database port is invalid.")
    destination["port"] = str(port)
    destination["hostaddr"] = "::1" if host == "::1" or hostaddr == "::1" else "127.0.0.1"
    try:
        return make_conninfo("", **destination)
    except ProgrammingError:
        raise ValueError("The dedicated test database URL is invalid.") from None


@pytest.mark.parametrize("bad_url,pg_env", [
    ("postgresql://localhost:5432/rag_ai", {}),
    ("postgresql://db.example:5432/rag_ai_retrieval_test", {}),
    ("postgresql://user:synthetic-secret@db.example:5432/rag_ai_retrieval_test", {}),
    ("postgresql://localhost:5432/rag_ai_retrieval_test?hostaddr=8.8.8.8", {}),
    ("postgresql://127.0.0.1:5432/rag_ai_retrieval_test?hostaddr=::1", {}),
    ("postgresql://localhost:5432/rag_ai_retrieval_test?service=other", {}),
    ("postgresql://localhost:5432/rag_ai_retrieval_test", {"PGHOSTADDR": "8.8.8.8"}),
    ("postgresql://localhost:5432/rag_ai_retrieval_test", {"PGHOST": "db.example"}),
    ("postgresql://localhost:5432/rag_ai_retrieval_test", {"PGDATABASE": "rag_ai"}),
    ("postgresql://localhost:5432/rag_ai_retrieval_test", {"PGSERVICE": "other"}),
    ("postgresql://localhost:5432/rag_ai_retrieval_test", {"PGSERVICEFILE": "other.conf"}),
    ("postgresql://localhost:5432/rag_ai_retrieval_test", {"PGPORT": "6543"}),
])
def test_e2e_database_preflight_rejects_unsafe_destination(
    bad_url: str, pg_env: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("psycopg.connect", lambda *args, **kwargs: calls.append("connect"))
    with pytest.raises(ValueError) as error:
        _validated_test_conninfo(bad_url, pg_env)
    assert not calls
    assert "password" not in str(error.value).lower()
    assert "synthetic-secret" not in str(error.value)
    assert "db.example" not in str(error.value)


@pytest.mark.parametrize("url,expected_addr", [
    ("postgresql://127.0.0.1:5432/rag_ai_retrieval_test", "127.0.0.1"),
    ("postgresql://localhost:5432/rag_ai_retrieval_test", "127.0.0.1"),
    ("postgresql://[::1]:5432/rag_ai_retrieval_test", "::1"),
])
def test_e2e_database_preflight_pins_loopback_without_connecting(
    url: str, expected_addr: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("psycopg.connect", lambda *args, **kwargs: calls.append("connect"))
    safe_url = _validated_test_conninfo(url, {})
    destination = conninfo_to_dict(safe_url)
    assert destination["dbname"] == "rag_ai_retrieval_test"
    assert destination["hostaddr"] == expected_addr
    assert destination["port"] == "5432"
    assert not calls


def _synthetic_pdf() -> bytes:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 72, 520, 250),
            "The emergency-stop button does not directly disconnect motor power. "
            "Pressing it opens the safety relay. The safety relay is the component "
            "that disconnects electrical power to the drive motor during an emergency "
            "stop. A normal software stop does not open the safety relay.",
            fontsize=10,
        )
        page = pdf.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 72, 520, 250),
            "The battery coolant pump circulates coolant through cold plates. "
            "The radiator releases heat from the coolant to the surrounding air.",
            fontsize=10,
        )
        return pdf.tobytes()


def test_local_pdf_rag_end_to_end(e2e_database_url: str) -> None:
    """Exercise real PDF parsing, local models, pgvector, and cited API answers."""
    MigrationRunner(e2e_database_url).upgrade()
    collection_id, other_collection_id = uuid4(), uuid4()
    settings = Settings(
        app_env="development", enable_document_parse_endpoint=True,
        enable_document_index_endpoint=True, enable_document_retrieval_endpoint=True,
        enable_rag_answer_endpoint=True, database_url=e2e_database_url,
        ollama_base_url=os.getenv("RAG_AI_TEST_OLLAMA_URL", "http://127.0.0.1:11434"),
        embedding_model="embeddinggemma:300m", generation_model="qwen3:4b-instruct",
        _env_file=None,
    )
    filename = "synthetic_engineering_manual.pdf"
    pdf_bytes = _synthetic_pdf()
    timings: dict[str, float] = {}

    async def exercise() -> dict[str, object]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(settings)),
            base_url="http://test",
            timeout=120.0,
        ) as client:
            def pdf_upload() -> dict[str, tuple[str, bytes, str]]:
                return {"file": (filename, pdf_bytes, "application/pdf")}

            parsed_response = await client.post("/documents/parse", files=pdf_upload())
            assert parsed_response.status_code == 200, parsed_response.status_code
            parsed = parsed_response.json()
            assert parsed["filename"] == filename and parsed["file_type"] == "pdf"
            assert parsed["page_count"] == 2
            pages = parsed["extraction_metadata"]["pages"]
            assert [page["page_number"] for page in pages] == [1, 2]
            assert "safety relay" in pages[0]["text"]
            assert "coolant pump" in pages[1]["text"]

            chunk_response = await client.post("/documents/chunk", files=pdf_upload())
            assert chunk_response.status_code == 200, chunk_response.status_code
            chunked = chunk_response.json()
            assert chunked["document_id"] == parsed["document_id"]
            assert chunked["total_chunks"] >= 2
            assert {chunk["page_number"] for chunk in chunked["chunks"]} == {1, 2}
            source_chunks = {chunk["chunk_id"]: chunk for chunk in chunked["chunks"]}

            start = perf_counter()
            indexed_response = await client.post(
                f"/collections/{collection_id}/documents/index", files=pdf_upload(),
            )
            timings["index_seconds"] = round(perf_counter() - start, 3)
            assert indexed_response.status_code == 200, indexed_response.status_code
            indexed = indexed_response.json()
            assert indexed["collection_id"] == str(collection_id)
            assert indexed["document_id"] == parsed["document_id"]
            assert indexed["filename"] == filename and indexed["file_type"] == "pdf"
            assert indexed["total_chunks"] == chunked["total_chunks"]
            assert indexed["embedding_model"] == "embeddinggemma:300m"
            assert indexed["embedding_model_revision"]
            assert 1 <= indexed["embedding_dimensions"] <= settings.max_embedding_dimensions
            assert "safety relay" not in indexed_response.text
            assert "vectors" not in indexed_response.text

            warranty = await client.post(
                f"/collections/{other_collection_id}/documents/index",
                files={"file": ("synthetic_warranty.txt", b"The warranty period is two years.", "text/plain")},
            )
            assert warranty.status_code == 200, warranty.status_code
            other_document_id = warranty.json()["document_id"]

            emergency_query = "What disconnects electrical power to the drive motor during an emergency stop?"
            start = perf_counter()
            search_response = await client.post(
                f"/collections/{collection_id}/search", json={"query": emergency_query, "top_k": 2},
            )
            timings["search_seconds"] = round(perf_counter() - start, 3)
            assert search_response.status_code == 200, search_response.status_code
            search = search_response.json()
            assert search["collection_id"] == str(collection_id) and search["results"]
            assert search["results"][0]["page_number"] == 1
            assert all(result["document_id"] == indexed["document_id"]
                       and result["filename"] == filename
                       and result["chunk_id"] in source_chunks
                       and result["page_number"] == source_chunks[result["chunk_id"]]["page_number"]
                       and "vector" not in result for result in search["results"])

            async def answer(label: str, query: str) -> dict[str, object]:
                start = perf_counter()
                response = await client.post(
                    f"/collections/{collection_id}/answer", json={"query": query, "top_k": 1},
                )
                timings[f"{label}_answer_seconds"] = round(perf_counter() - start, 3)
                assert response.status_code == 200, response.status_code
                return response.json()

            emergency = await answer("emergency", emergency_query)
            cooling = await answer("cooling", "What component circulates coolant through the battery cold plates?")
            unsupported = await answer("unsupported", "How long is the warranty period?")

            for response, page_number, terms in (
                (emergency, 1, ("relay", "power")),
                (cooling, 2, ("coolant", "pump")),
            ):
                assert response["collection_id"] == str(collection_id)
                assert response["status"] == "answered" and response["claims"]
                claim_text = " ".join(claim["text"].lower() for claim in response["claims"])
                missing_terms = [term for term in terms if term not in claim_text]
                assert not missing_terms, (page_number, missing_terms, claim_text[:160])
                for claim in response["claims"]:
                    assert claim["citations"]
                    for citation in claim["citations"]:
                        assert citation["source_id"] == "S1"
                        assert citation["document_id"] == indexed["document_id"]
                        assert citation["chunk_id"] in source_chunks
                        assert citation["filename"] == filename
                        assert citation["file_type"] == "pdf"
                        assert citation["page_number"] == page_number
                        assert source_chunks[citation["chunk_id"]]["page_number"] == page_number

            assert unsupported["status"] == "insufficient_evidence"
            assert unsupported["claims"] == []
            assert unsupported["collection_id"] == str(collection_id)

            primary_warranty_search = await client.post(
                f"/collections/{collection_id}/search",
                json={"query": "How long is the warranty period?", "top_k": 5},
            )
            other_warranty_search = await client.post(
                f"/collections/{other_collection_id}/search",
                json={"query": "How long is the warranty period?", "top_k": 5},
            )
            assert primary_warranty_search.status_code == 200
            assert other_warranty_search.status_code == 200
            assert all(result["document_id"] != other_document_id
                       for result in primary_warranty_search.json()["results"])
            assert any(result["document_id"] == other_document_id
                       for result in other_warranty_search.json()["results"])
            return {
                "pages": parsed["page_count"], "chunks": indexed["total_chunks"],
                "embedding_model": indexed["embedding_model"],
                "embedding_dimensions": indexed["embedding_dimensions"],
                "generation_model": settings.generation_model,
                "search_first_page": search["results"][0]["page_number"],
                "answer_statuses": [emergency["status"], cooling["status"], unsupported["status"]],
                "citation_pages": [
                    emergency["claims"][0]["citations"][0]["page_number"],
                    cooling["claims"][0]["citations"][0]["page_number"],
                ],
            }

    started = perf_counter()
    summary = asyncio.run(exercise())
    timings["total_seconds"] = round(perf_counter() - started, 3)
    print("Local synthetic PDF RAG verification:", json.dumps(summary), json.dumps(timings))
