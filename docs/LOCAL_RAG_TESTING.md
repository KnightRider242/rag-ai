# Local RAG Testing — Manual and Automated Verification

## Overview

RAG-AI currently accepts PDF, TXT, and Markdown for development-only parsing and chunking. The local path is **upload → extract text → split into chunks → Ollama embeddings → PostgreSQL/pgvector → semantic search → Ollama answer → source citations**. FastAPI's Swagger page is the interactive testing interface; a React frontend is planned, not implemented. These routes are for a trusted machine bound to localhost, not public deployment.

The E2E regression is merged on stable `main`. This guide separates the user's **manual Swagger observations** from the prior **Codex automated test runs**. The examples use synthetic engineering text, not a real manual or safety specification.

## Prerequisites and safe startup

Use Python 3.12+, the project's existing virtual environment, Docker Compose with PostgreSQL/pgvector, and a native local Ollama installation. Confirm that `embeddinggemma:300m` and `qwen3:4b-instruct` are already installed; this guide does not download models. The application database named by your private `DATABASE_URL` must have a compatible, already-migrated schema. Check out the local RAG branch you intend to test. See the [README setup](../README.md#local-setup) and [database operations guide](DATABASE_MIGRATIONS.md) if prerequisites are missing.

In your existing, Git-ignored `.env`, retain your own `DATABASE_URL`, PostgreSQL role, and password. Do not overwrite that file. Enable only these development routes and local models:

```dotenv
APP_ENV=development
ENABLE_DOCUMENT_PARSE_ENDPOINT=true
ENABLE_DOCUMENT_INDEX_ENDPOINT=true
ENABLE_DOCUMENT_RETRIEVAL_ENDPOINT=true
ENABLE_RAG_ANSWER_ENDPOINT=true
OLLAMA_BASE_URL=http://127.0.0.1:11434
EMBEDDING_MODEL=embeddinggemma:300m
GENERATION_MODEL=qwen3:4b-instruct
```

On Linux or macOS, from the repository root:

```bash
source .venv/bin/activate
docker compose up -d postgres
ollama list
python -m rag_ai.storage status
uvicorn rag_ai.main:app --reload --host 127.0.0.1 --port 8000
```

Run Uvicorn in its own terminal. PostgreSQL or Ollama may already be running. `ollama list` shows installed models; it does **not** prove the Ollama HTTP service is ready. `python -m rag_ai.storage status` is read-only. If it reports an empty, unversioned, partial, or incompatible schema, stop and consult the [migration guide](DATABASE_MIGRATIONS.md); do not run `upgrade` or `adopt-legacy` automatically. Keep the API bound to `127.0.0.1`. On Windows PowerShell, use the virtual-environment activation described in the README; the remaining application commands are the same. Platform-specific runtime testing is still pending.

## Make a searchable two-page synthetic PDF

With the virtual environment active, use this Linux/macOS shell example from a directory where you want a **synthetic** `engineering_manual.pdf`. On Windows PowerShell, save the Python block to a temporary `.py` file and run it with `python`; the generation code itself is cross-platform. The installed PyMuPDF package generates it entirely from text; no external file or dataset is needed.

```bash
python - <<'PY'
from pathlib import Path
import pymupdf

pages = [
    "The emergency-stop button does not directly disconnect motor power. "
    "Pressing it opens the safety relay. The safety relay is the component "
    "that disconnects electrical power to the drive motor during an emergency "
    "stop. A normal software stop does not disconnect power through the relay.",
    "The battery coolant pump circulates coolant through cold plates. "
    "The radiator transfers heat from the coolant to the surrounding air.",
]
with pymupdf.open() as document:
    for text in pages:
        page = document.new_page()
        spare_height = page.insert_textbox(pymupdf.Rect(72, 72, 520, 250), text, fontsize=10)
        if spare_height < 0 or not page.get_text().strip():
            raise RuntimeError("Synthetic PDF text did not fit or was not extractable")
    Path("engineering_manual.pdf").write_bytes(document.tobytes())
PY
```

This PDF is a test fixture only. It is not an operational safety instruction.

## Test through FastAPI Swagger

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). In Swagger, expand each operation, select **Try it out**, supply the values below, and select **Execute**. Generate one collection UUID and reuse it for indexing, search, and answers:

```bash
python -c "import uuid; print(uuid.uuid4())"
```

1. **`GET /health`:** Check that the API process responds. Health is a liveness check, not database or Ollama readiness.
2. **`POST /documents/parse`:** Upload `engineering_manual.pdf` in the multipart `file` field. Expect a PDF with two numbered pages and extracted text. This call does not persist the file.
3. **`POST /documents/chunk`:** Upload the same PDF. With the documented default chunk settings, expect one ordered chunk per page. This call does not persist chunks.
4. **`POST /collections/{collection_id}/documents/index`:** Use the same PDF and the UUID above. This call **persists the original bytes, chunks, and embeddings** in the application database. The response contains IDs, chunk count, model revision, and measured dimensions, but no document text or vectors. There is no automatic retention expiry or HTTP deletion route.
5. **`POST /collections/{collection_id}/search`:** Use JSON such as `{"query":"What disconnects power to the drive motor during an emergency stop?","top_k":2}`. Inspect the ranked result's filename, page number, chunk ID, and distance. Search returns chunk text, not raw vectors.
6. **`POST /collections/{collection_id}/answer`:** Ask the same question with `top_k` set to 1 or 2. Then ask about the battery coolant pump and an unsupported warranty period. Inspect status, claims, and citations.

The UUID scopes stored records and queries; it is **not a credential or access permission**. A different UUID may have no indexed results. Do not use another person's collection UUID as an access-control test on a shared server; this development API has no caller authorization.

For a Linux/macOS shell, these curl requests correspond to the Swagger steps after you have generated and set one UUID in the current shell. Keep it unchanged for the three collection routes:

```bash
COLLECTION_ID=$(python -c "import uuid; print(uuid.uuid4())")
curl http://127.0.0.1:8000/health
curl -F 'file=@engineering_manual.pdf;type=application/pdf' http://127.0.0.1:8000/documents/parse
curl -F 'file=@engineering_manual.pdf;type=application/pdf' http://127.0.0.1:8000/documents/chunk
curl -F 'file=@engineering_manual.pdf;type=application/pdf' "http://127.0.0.1:8000/collections/${COLLECTION_ID}/documents/index"
curl -H 'Content-Type: application/json' -d '{"query":"What disconnects power to the drive motor during an emergency stop?","top_k":2}' "http://127.0.0.1:8000/collections/${COLLECTION_ID}/search"
curl -H 'Content-Type: application/json' -d '{"query":"What disconnects power to the drive motor during an emergency stop?","top_k":2}' "http://127.0.0.1:8000/collections/${COLLECTION_ID}/answer"
```

The collection commands write only when you call **index**. These shell examples avoid credentials; keep your database connection setting private. PowerShell uses different variable assignment and interpolation syntax, so use Swagger or adapt those shell lines for PowerShell.

## What the user observed in one manual Swagger session

The user reported HTTP 200 for parsing `engineering_manual.pdf`, with **two PDF pages** and text extracted from each. Chunking returned HTTP 200 and **two chunks**: chunk index 0 on page 1 and chunk index 1 on page 2. Document and chunk identifiers are deterministic for identical bytes and chunk settings. Indexing returned HTTP 200, **two indexed chunks**, `embeddinggemma:300m`, a recorded model revision, and **768 measured dimensions**. Generation used `qwen3:4b-instruct`.

| Question | Observed answer/status | Citation |
| --- | --- | --- |
| “What disconnects power to the drive motor during an emergency stop?” | “The safety relay disconnects electrical power to the drive motor during an emergency stop.” | `engineering_manual.pdf`, page 1 |
| “What component circulates coolant through the battery cooling system?” | “The battery coolant pump circulates coolant through the battery cooling system.” | `engineering_manual.pdf`, page 2 |
| “What is the warranty period?” | `insufficient_evidence`, no claims | None |

Selected fields from the observed answer shape look like this; the real API also returns collection, document, and chunk IDs:

```json
{"status":"answered","claims":[{"text":"The safety relay disconnects electrical power to the drive motor during an emergency stop.","citations":[{"source_id":"S1","filename":"engineering_manual.pdf","page_number":1}]}]}
```

For the unsupported question, selected fields were:

```json
{"status":"insufficient_evidence","claims":[]}
```

These are **user-reported observations from one manual session**, not responses re-executed for this guide. Filenames, IDs, source labels, ranking, and wording can vary with the uploaded bytes, settings, model digest, and retrieval results. The example JSON omits other actual response fields for readability.

## How citations work, and what they do not prove

`documents/parser.py` extracts one-based PDF page numbers. `documents/chunker.py` splits each page separately and carries its number into each chunk. `documents/indexer.py` and `storage/postgres.py` persist the source metadata with vectors. Search returns ranked chunks from the requested collection. `rag/context.py` bounds the evidence and assigns request-local labels such as `S1`; the model sees those labels, but it cannot authoritatively choose document IDs or filenames. `rag/service.py` checks that every generated label exists in that request's retrieved context and resolves it to the stored document ID, chunk ID, filename, file type, and page number. Unknown labels fail instead of being silently dropped.

**Citation provenance** means the cited chunk existed in the retrieved evidence. **Factual grounding** means the claim truly follows from that chunk. The application verifies provenance; it cannot prove semantic entailment. In an early automated live test, Ollama cited the correct page but answered “emergency-stop button” rather than “safety relay.” The synthetic PDF was clarified to distinguish the button's action from the relay's function; the original relay assertion remained, and the final test passed. This was a generation/grounding observation, not a demonstrated retrieval or citation-resolution defect. Clearer test wording does not guarantee correctness for real documents.

## Automated end-to-end test and recorded results

[`tests/test_local_rag_e2e.py`](../tests/test_local_rag_e2e.py) creates a separate synthetic two-page PDF in memory, then uses HTTPX `ASGITransport` to call the actual FastAPI parse, chunk, index, search, and answer routes. It uses the installed local Ollama models, a second collection with a synthetic warranty TXT file, and PostgreSQL/pgvector. It checks page-specific citations, abstention, and collection filtering without mocking generation.

Set `RAG_AI_TEST_DATABASE_URL` **privately** to an authorized, disposable, local database named `rag_ai_retrieval_test`, with pgvector installed and permission to create schemas; set `RAG_AI_RUN_OLLAMA_TESTS=1`. Never paste the URL or password into chat, shell history, or a committed file. `RAG_AI_TEST_OLLAMA_URL` optionally selects another explicit **loopback** Ollama endpoint; the default is `http://127.0.0.1:11434`. The new E2E fixture checks the exact database name, URL host/hostaddr, and routing-related PostgreSQL environment settings **before** it requests the shared isolated-schema fixture. It rejects `PGHOST`, `PGHOSTADDR`, `PGDATABASE`, `PGPORT`, `PGSERVICE`, `PGSERVICEFILE`, `PGSYSCONFDIR`, `PGOPTIONS`, `PGLOADBALANCEHOSTS`, and `PGTARGETSESSIONATTRS`, as well as an explicit URL service. It then pins a loopback `hostaddr` and explicit port in the DSN that the shared fixture actually reads. This prevents libpq defaults or service configuration from silently changing the validated destination. [PostgreSQL documents environment defaults](https://www.postgresql.org/docs/current/libpq-envars.html) and [service-file precedence](https://www.postgresql.org/docs/current/libpq-pgservice.html); psycopg's `conninfo_to_dict()` parses the supplied string but does not itself resolve all libpq defaults. The fixture creates a random schema, runs explicit migrations inside it for the test, and drops only that new schema afterward. The strict destination preflight belongs to **this E2E test**. Other existing integration tests use the configured test URL and shared disposable-schema fixture without the same exact-name guard, so verify the destination yourself before running the broader suite.

```bash
python -m pytest -q -s tests/test_local_rag_e2e.py
python -m pytest -q
```

The first command requires the opt-in variables and local services for the live E2E case; its connection-free destination checks run without them. The second command runs the default suite when opt-in variables are unset, or the broader local integration suite when they are set. Do not aim the broader suite at an application database or one containing real documents.

The **completed Codex development session**, not this documentation session, recorded:

| Test | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Final focused live PDF E2E | 1 | 0 | 0 |
| New database preflight tests | 3 | 0 | 0 |
| Complete local PostgreSQL + Ollama suite | 229 | 0 | 0 |
| Complete default suite | 204 | 0 | 25 |

The separate preflight run deselected the live E2E case. Two earlier diagnostic live attempts each failed the relay-wording assertion and were investigated as described above. The runs emitted **13 nonfatal Python 3.14 deprecation warnings** from installed LangSmith. The test database had pgvector 0.8.7 during that session; read-only inspection afterward found no newly remaining test schema. One older orphan schema was left untouched.

The same final focused run observed these timings on one small local workload:

| Operation | Time |
| --- | ---: |
| PDF indexing | 0.078 s |
| Semantic search | 0.032 s |
| Emergency-stop answer | 0.789 s |
| Cooling answer | 0.630 s |
| Unsupported-question answer | 0.297 s |
| Complete test workflow | 1.939 s |

These are observations, **not** latency guarantees, GPU measurements, or scale benchmarks.

### Later PostgreSQL test-safety correction

An independent review found that the earlier URL-only preflight could accept `host=localhost` while libpq used a different effective `hostaddr` from `PGHOSTADDR` or a connection service. The E2E preflight now rejects those routing overrides, requires the dedicated database and loopback address, and passes a normalized, loopback-pinned DSN to the existing schema fixture **before its first connection or schema write**. It does not edit `.env` or change application database configuration. A loopback connection can still pass through a separately configured local proxy or tunnel; the test cannot prove the identity of a server behind that endpoint.

In the correction session, the focused preflight run passed **15 tests** and deselected the live E2E case. The default suite with temporary production/disabled-route overrides passed **216, failed 0, skipped 25**, with 13 LangSmith deprecation warnings. The live E2E module passed **16, failed 0, skipped 0**; the complete opt-in local PostgreSQL/Ollama suite passed **241, failed 0, skipped 0**. Read-only inspection found only the previously known orphan schema after testing. These are separate, later results from the historical table above; no earlier record was replaced. The user's development `.env` route flags were intentional and were not modified.

## Troubleshooting by stage

| Symptom | Check first |
| --- | --- |
| Swagger unavailable | Check that Uvicorn is running on `127.0.0.1:8000`; `/health` checks only process liveness. |
| Route returns 404 | Check `APP_ENV=development` and the route's individual flag. Index also needs the parse flag. Restart Uvicorn after changing settings. |
| Database failure (often 503) | Check Compose service/readiness, private `DATABASE_URL`, and read-only `python -m rag_ai.storage status`. Do not reset a volume. |
| Schema is empty, unversioned, partial, or incompatible | Stop; follow the [database guide](DATABASE_MIGRATIONS.md) for backup-first, explicitly authorized operations. Do not force a migration. |
| Ollama/model failure (often 502) | Check local Ollama service readiness and `ollama list` for both model names. Do not download a replacement model automatically. |
| Model/digest/dimension conflict (409) | Check the configured embedding model and its revision against the collection's indexed metadata. Do not silently mix embedding spaces. |
| Search returns unexpected evidence | Check the collection UUID, source filename/page, distance ordering, query wording, and configured model; there is no guaranteed relevance threshold. |
| Answer fails with invalid citations or generation (502) | Check whether retrieval returned the right page, then inspect controlled provider/citation errors. Do not log full evidence, prompts, or credentials. |
| Unsupported question gets an answer | Treat it as a grounding failure. A valid citation label alone does not prove support; review the retrieved evidence and generated claim. |
| Scanned PDF rejected (422) | The parser has no OCR; use a PDF with extractable text for this test. |
| Live tests skipped | Set both opt-in variables and verify the dedicated test database and local services. A skipped test is not a pass. |

## Limits of this verification

The development routes have no caller authentication or collection authorization. They are disabled by default and must remain on trusted localhost; public or shared deployment is unsupported. Indexing into the ordinary application database retains original document bytes, chunks, and vectors until explicit deletion, with no automatic expiry. The E2E fixture cleans up only its own disposable test schema.

These tests do not guarantee factual grounding, security against prompt injection, disconnected-network operation, Windows/macOS runtime behavior, or large-document performance. OCR, multimodal indexing, and a frontend are not implemented. No cloud request was made during local E2E testing; separate cloud-provider development branches are unmerged and are not part of this stable-main local workflow. The [roadmap](ROADMAP.md) lists later work without treating it as current capability.
