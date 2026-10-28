# RAG-AI

RAG-AI is an enterprise knowledge assistant under development. This repository currently provides the Python API foundation and a local PostgreSQL database with pgvector. Retrieval, LLM integration, and the React client are planned for later milestones.

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
   uvicorn rag_ai.main:app --reload
   ```

   Visit `http://127.0.0.1:8000/health` for liveness and `http://127.0.0.1:8000/docs` for API documentation.

## Tests

```bash
python -m pytest
```

Tests exercise the HTTP API without requiring a running database. `/health` reports API process liveness; it does not assert database readiness.

## Configuration

The application reads environment variables and optionally `.env` from the working directory. `APP_NAME` changes the OpenAPI title. `DATABASE_URL` is reserved for the database integration milestone. Docker Compose reads `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and optionally `POSTGRES_PORT` from `.env`. Use a managed secret store for deployed credentials; do not commit credentials or environment files.

## Layout

```text
src/rag_ai/api/       HTTP routes and response models
src/rag_ai/core/      Typed application configuration
src/rag_ai/main.py    FastAPI application factory and entry point
tests/               API tests
docker/postgres/      Database initialization script
compose.yaml          Local PostgreSQL with pgvector
```

The repository retains its [Apache 2.0 license](LICENSE).
