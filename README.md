# RAG-AI

RAG-AI is an enterprise knowledge assistant under development. This repository provides a Python API, in-memory document text extraction, and a local PostgreSQL database with pgvector. Retrieval, LLM integration, and the React client are planned for later milestones.

## Requirements

- Python 3.12 or newer
- Docker with Compose (for the local database)

## Local setup

1. Create a local environment file and set a unique development password:

   ```bash
   cp .env.example .env
   ```

   Update both `POSTGRES_PASSWORD` and the password in `DATABASE_URL` so they match. Keep `.env` local; it is ignored by Git.

2. Create a virtual environment and install the backend with test dependencies:

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

   The database is bound to `127.0.0.1` on `POSTGRES_PORT` (default `5432`). The `vector` extension is enabled when the database volume is first created. The API does not yet connect to the database.

4. Run the API:

   ```bash
   uvicorn rag_ai.main:app --reload --host 127.0.0.1
   ```

   Visit `http://127.0.0.1:8000/health` for liveness and `http://127.0.0.1:8000/docs` for API documentation.

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

## Tests

```bash
python -m pytest
```

Tests exercise parsing and chunking through the HTTP API without requiring a running database. `/health` reports API process liveness; it does not assert database readiness.

## Configuration

The application reads environment variables and optionally `.env` from the working directory. `APP_NAME` changes the OpenAPI title. `APP_ENV` and `ENABLE_DOCUMENT_PARSE_ENDPOINT` control development route registration. Upload, extraction, and chunking limits are described above and in `.env.example`. `DATABASE_URL` is reserved for the database integration milestone. Docker Compose reads `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and optionally `POSTGRES_PORT` from `.env`. Use a managed secret store for deployed credentials; do not commit credentials or environment files.

## Layout

```text
src/rag_ai/api/       HTTP routes and response models
src/rag_ai/core/      Typed application configuration
src/rag_ai/documents/ Validated extraction and deterministic chunking
src/rag_ai/main.py    FastAPI application factory and entry point
tests/               API tests
docker/postgres/      Database initialization script
compose.yaml          Local PostgreSQL with pgvector
```

The repository retains its [Apache 2.0 license](LICENSE).

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) (MIT) powers the backend API; [PyMuPDF](https://pymupdf.readthedocs.io/) (AGPL or commercial license) extracts PDF text.
- [LangChain text splitters](https://github.com/langchain-ai/langchain/tree/master/libs/text-splitters) (MIT) provides `RecursiveCharacterTextSplitter` for chunking. Other LangChain features are not implemented.
- [PostgreSQL](https://www.postgresql.org/) and [pgvector](https://github.com/pgvector/pgvector) (PostgreSQL License) provide the local database foundation. Vector search is planned, not implemented.
- [Python](https://www.python.org/) (PSF License) and [Pydantic](https://docs.pydantic.dev/) (MIT) support the backend and typed configuration.
- [OpenAI Codex](https://openai.com/codex/) assisted with development and review. This acknowledgement does not imply endorsement by OpenAI or any other project listed here.

Each third-party project remains the work of its respective authors and is governed by its own license. Review PyMuPDF's license terms before commercial distribution or deployment.
