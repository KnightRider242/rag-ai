# RAG-AI

RAG-AI is an enterprise knowledge assistant under development. This repository provides a Python API, document extraction and chunking, and development-only local indexing with Ollama embeddings in PostgreSQL/pgvector. Retrieval, answer generation, and the React client are planned for later milestones.

## Requirements

- Python 3.12 or newer
- Docker with Compose (for the local database)

## Local setup

1. Create a local environment file and set a unique development password:

   ```bash
   cp .env.example .env
   ```

   Update both `POSTGRES_PASSWORD` and the password in `DATABASE_URL` so they match. Keep `.env` local; it is ignored by Git.

2. Create a virtual environment and install the backend with test dependencies (macOS/Linux example):

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e '.[dev]'
   ```

   If `python3.12` is unavailable, use another installed Python version 3.12 or newer.

3. Start PostgreSQL:

   ```bash
   docker compose up -d postgres
   docker compose ps
   ```

   The database is bound to `127.0.0.1` on `POSTGRES_PORT` (default `5432`). The `vector` extension is enabled when the database volume is first created. For a fresh database, run `python -m rag_ai.storage upgrade` after PostgreSQL is healthy. For an existing unversioned Phase 4 database, first follow the [migration and backup instructions](docs/DATABASE_MIGRATIONS.md). Migrations never run automatically at API startup.

4. Run the API:

   ```bash
   uvicorn rag_ai.main:app --reload --host 127.0.0.1
   ```

   Visit `http://127.0.0.1:8000/health` for liveness and `http://127.0.0.1:8000/docs` for API documentation.

The Python backend targets Windows, macOS, and Linux. On Windows PowerShell, use `Copy-Item .env.example .env`, `py -3.12 -m venv .venv`, `.venv\Scripts\Activate.ps1`, and `python -m pip install -e '.[dev]'` for steps 1–2; then use the same `docker compose` and `uvicorn` commands. Use an installed Python version 3.12 or newer. These are setup instructions, not a claim that the full test suite has been run on every platform.

## Document parsing

The parse route is **disabled by default**. To use it for local development, set both values in `.env`:

```dotenv
APP_ENV=development
ENABLE_DOCUMENT_PARSE_ENDPOINT=true
```

`APP_ENV` accepts `development`, `test`, or `production` and defaults to `production`; the enable flag defaults to `false`. The route is registered only when both development settings above are present. Enabling it in production causes application startup to fail with a configuration error. The route has no authentication: use it only on a trusted machine through `127.0.0.1`, and do not expose it on a public or shared network.

`POST /documents/parse` accepts one multipart field named `file`. It supports `.pdf` (`application/pdf`), `.txt` (`text/plain`), and `.md` (`text/markdown`, `text/x-markdown`, or `text/plain`). Text and Markdown must be UTF-8. For example:

```bash
curl -F 'file=@notes.md;type=text/markdown' http://127.0.0.1:8000/documents/parse
```

The JSON response includes a SHA-256 `document_id` based on the uploaded bytes, a safe filename, file type, extracted `text`, and `extraction_metadata`. PDFs also return `page_count` and one-based page numbers with text for each page. TXT and Markdown return `page_count: null` and `encoding: "utf-8"`.

The endpoint rejects unsupported formats and content types (HTTP 415), oversized files or extracted text (413), and empty, invalid, or non-UTF-8 content (422). Image-only PDFs return a `scanned_pdf` error; OCR is not included. `MAX_UPLOAD_SIZE_BYTES` defaults to 10 MiB, `MAX_PDF_PAGES` to 500, and `MAX_EXTRACTED_CHARACTERS` to 2 million. All are configurable in `.env`. The application reads at most one byte beyond the file limit, closes the upload after processing, and does not persist document contents. The multipart framework may temporarily spool uploads to disk before the route checks their size. Any future deployment that accepts documents needs authentication, authorization, and an ingress request-size limit. See the [security test plan](docs/SECURITY_TEST_PLAN.md).

## Document chunking

`POST /documents/chunk` accepts the same multipart `file` field and applies the same parsing and security checks as `/documents/parse`. It has the same development-only gate: set `APP_ENV=development` and `ENABLE_DOCUMENT_PARSE_ENDPOINT=true`, then run the API bound to `127.0.0.1` on a trusted machine. The endpoint is absent by default and in production.

```bash
curl -F 'file=@notes.md;type=text/markdown' http://127.0.0.1:8000/documents/chunk
```

The server uses LangChain's `RecursiveCharacterTextSplitter` with character counts (`len`). PDF pages are split separately, empty pages are skipped, and each chunk retains its one-based source page number. TXT and Markdown chunks have `page_number: null`. The response contains `document_id`, filename, file type, `total_chunks`, `chunk_size`, `chunk_overlap`, and a `chunks` list. Every chunk contains its SHA-256 `chunk_id`, document ID, filename, file type, page number, zero-based `chunk_index`, and text. Chunk IDs hash a canonical representation of the document ID, page, index, text, and chunking configuration, so repeating the same input and configuration yields the same IDs.

| Setting | Default | Valid range |
| --- | ---: | --- |
| `CHUNK_SIZE` | 1000 | 64–10,000 characters |
| `CHUNK_OVERLAP` | 150 | 0 to at most half of `CHUNK_SIZE` |
| `MAX_CHUNKS_PER_DOCUMENT` | 5000 | 1–10,000 chunks |

The server controls these settings through the environment; clients cannot override them per request. Actual overlap may vary because the splitter chooses text boundaries. Chunking returns HTTP 413 when the document exceeds the chunk count limit. A preflight check rejects sources whose non-whitespace text cannot fit within the remaining chunk budget. Chunks are returned in memory and are not stored or sent to an external service. LangChain still materializes a list for one source page or text document before the final count check; the extracted-character limit, minimum chunk size, and overlap cap reduce this remaining CPU and memory risk.

## Local document indexing

Phase 4 adds **development-only** `POST /collections/{collection_id}/documents/index`. Install [Ollama](https://ollama.com/download) natively on Windows, macOS, or Linux, run `ollama pull embeddinggemma:300m`, and keep its HTTP service on the local machine. PostgreSQL with pgvector runs through the existing Docker Compose service. Set these values in the local, Git-ignored `.env` file:

```dotenv
APP_ENV=development
ENABLE_DOCUMENT_PARSE_ENDPOINT=true
ENABLE_DOCUMENT_INDEX_ENDPOINT=true
OLLAMA_BASE_URL=http://127.0.0.1:11434
EMBEDDING_MODEL=embeddinggemma:300m
```

The indexing flag defaults to `false` and also requires the parse flag. The route is absent outside development; enabling it in production fails configuration validation. `OLLAMA_BASE_URL` accepts only a loopback HTTP endpoint, including when the provider is constructed directly. Redirects are rejected and no cloud fallback is present. Start Uvicorn with `--host 127.0.0.1` as shown above, use the route only on a trusted machine, and never expose it publicly. The route has **no authentication or collection authorization**. Route gating does not enforce the server bind address; an application-level peer or Host-header check would not reliably isolate it from a local reverse proxy.

After starting PostgreSQL and running the appropriate [explicit migration command](docs/DATABASE_MIGRATIONS.md), submit a synthetic or otherwise authorized document to a UUID collection:

```bash
curl -F 'file=@notes.md;type=text/markdown' \
  http://127.0.0.1:8000/collections/00000000-0000-4000-8000-000000000001/documents/index
```

The endpoint reuses Phase 2 validation and Phase 3 chunking, obtains local embeddings in batches of at most 16 chunks, and atomically stores the original bytes, document metadata, chunks, vectors, model name, Ollama model digest, and **actual returned vector dimensions**. The initial local `embeddinggemma:300m` test returned 768 dimensions; other models are not assumed to use 768. After the first batch reveals the actual dimension, the application checks `chunk count × dimensions` before generating the remaining embeddings. Batches are staged in a connection-local PostgreSQL temporary table and the final document replacement uses one short transaction. A failed batch or final write leaves the prior permanent index unchanged. This keeps Python vector memory bounded, at the cost of holding a database connection while Ollama processes the document. A collection cannot mix dimensions or digests under the same model name. Re-indexing the same document replaces its chunks and vectors in that collection. The response contains identifiers, chunk count, and model metadata, without document text or vectors. Semantic retrieval and citations are not implemented.

`MAX_INDEX_CHUNKS_PER_DOCUMENT` defaults to 500 (maximum 5000), `MAX_EMBEDDING_DIMENSIONS` defaults to 4096 (maximum 16,000), and `MAX_EMBEDDING_ELEMENTS` defaults to 1,000,000 (maximum 2,000,000). Exceeding the element budget returns HTTP 413 after the first dimension probe. These settings, the existing upload/extraction/chunk limits, and bounded Ollama responses constrain work. Original bytes, chunks, and vectors remain until explicitly deleted; there is no automatic expiry. The storage layer provides collection-scoped transactional document deletion, with no unauthenticated deletion route. See the [lifecycle policy and migration commands](docs/DATABASE_MIGRATIONS.md). Narrower database privileges, backup retention controls, and authenticated collection access are still required before deployment.

## Tests

```bash
python -m pytest
```

Tests exercise parsing and chunking through the HTTP API without requiring a running database. `/health` reports API process liveness; it does not assert database readiness.

The default suite skips the real database/Ollama tests. To run them, start the Compose database, set `RAG_AI_TEST_DATABASE_URL` to a **dedicated local test database URL** with the `vector` extension installed and schema-creation permission, and set `RAG_AI_RUN_OLLAMA_TESTS=1` with the model available locally. Then run `python -m pytest -q`. Tests use synthetic documents in disposable, randomly named schemas and remove those schemas afterward. The local end-to-end check verifies parse → chunk → embed → pgvector persistence and the measured 768 dimensions. Never use a database containing real documents for destructive integration tests. Windows and macOS runtime checks remain outstanding.

## Configuration

The application reads environment variables and optionally `.env` from the working directory. `APP_NAME` changes the OpenAPI title. `APP_ENV`, `ENABLE_DOCUMENT_PARSE_ENDPOINT`, and `ENABLE_DOCUMENT_INDEX_ENDPOINT` control development route registration. Upload, extraction, chunking, and embedding limits are described above and in `.env.example`. `DATABASE_URL` is used by the development indexing path and explicit migration commands. Docker Compose reads `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and optionally `POSTGRES_PORT` from `.env`. Use a managed secret store for deployed credentials; do not commit credentials or environment files.

## Roadmap

Phases 1–4 provide the backend foundation, document ingestion, chunking, and local Ollama/pgvector indexing. Local inference can use a CPU; [GPU acceleration](https://docs.ollama.com/gpu) depends on the host and Ollama support. Later phases plan semantic retrieval, cited text RAG, explicit cloud-provider choice, multimodal evidence, LangGraph workflows, MCP tools, a React frontend, and deployment hardening. These later capabilities are not implemented. See the [full roadmap](docs/ROADMAP.md) and [architecture decisions](docs/ARCHITECTURE_DECISIONS.md).

## Layout

```text
src/rag_ai/api/       HTTP routes and response models
src/rag_ai/core/      Typed application configuration
src/rag_ai/documents/ Validated extraction and deterministic chunking
src/rag_ai/embeddings/ Local Ollama provider and embedding boundary
src/rag_ai/storage/   Collection-scoped PostgreSQL/pgvector indexing
src/rag_ai/main.py    FastAPI application factory and entry point
tests/               API tests
docker/postgres/      Database initialization script
compose.yaml          Local PostgreSQL with pgvector
```

The repository retains its [Apache 2.0 license](LICENSE).

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) (MIT) powers the backend API; [PyMuPDF](https://pymupdf.readthedocs.io/) (AGPL or commercial license) extracts PDF text.
- [LangChain text splitters](https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters) (MIT) provides `RecursiveCharacterTextSplitter` for chunking. Other LangChain features are not implemented.
- [PostgreSQL](https://www.postgresql.org/) and [pgvector](https://github.com/pgvector/pgvector) (PostgreSQL License) store local document vectors. Vector search is planned, not implemented.
- [Ollama](https://ollama.com/) provides local embeddings for Phase 4; [Psycopg](https://www.psycopg.org/psycopg3/) connects the Python backend to PostgreSQL.
- [Python](https://www.python.org/) (PSF License) and [Pydantic](https://docs.pydantic.dev/) (MIT) support the backend and typed configuration.
- [OpenAI Codex](https://openai.com/codex/) assisted with development and review. This acknowledgement does not imply endorsement by OpenAI or any other project listed here.

Each third-party project remains the work of its respective authors and is governed by its own license. Review PyMuPDF's license terms before commercial distribution or deployment.
