# Local RAG Testing — Manual and Automated Verification

## Overview

RAG-AI currently accepts PDF, TXT, and Markdown for development-only parsing and chunking. The local path is **upload → extract text → split into chunks → Ollama embeddings → PostgreSQL/pgvector → semantic search → Ollama answer → source citations**. FastAPI's Swagger page is the interactive testing interface; a React frontend is planned, not implemented. These routes are for a trusted machine bound to localhost, not public deployment.

The E2E regression is merged on stable `main`. This guide separates the user's **manual Swagger observations** from the prior **Codex automated test runs**. The examples use synthetic engineering text, not a real manual or safety specification.

## Prerequisites and safe startup

Use Python 3.12+, the project's existing virtual environment, Docker Compose with PostgreSQL/pgvector, and a native local Ollama installation. Confirm that `embeddinggemma:300m` and `qwen3:4b-instruct` are already installed; this guide does not download models. The application database named by your private `DATABASE_URL` must have a compatible, already-migrated schema. Phases 8 and 9A.1 are merged on `main`; Phase 9A.2 visual interpretation persistence is on this feature branch and requires schema version 3. See the [README setup](../README.md#local-setup) and [database operations guide](DATABASE_MIGRATIONS.md) if prerequisites are missing.

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

## Inspect embedded PDF images (Phase 8A.1)

`POST /documents/inspect` is a separate development-only route. In your private `.env`, set `APP_ENV=development` and `ENABLE_DOCUMENT_VISUAL_INSPECTION_ENDPOINT=true`, then start `uvicorn rag_ai.main:app --reload --host 127.0.0.1 --port 8000`. Open `http://127.0.0.1:8000/docs` and upload a PDF to **documents → inspect**. The route is disabled by default and rejected in production; it has no authentication, so keep the server on a trusted localhost machine.

This short script creates two synthetic files in the current directory using the installed PyMuPDF package:

```python
from pathlib import Path
import pymupdf

with pymupdf.open() as document:
    document.new_page().insert_text((72, 72), "Synthetic text-only page")
    Path("text_only.pdf").write_bytes(document.tobytes())

with pymupdf.open() as document:
    page = document.new_page()
    page.insert_text((72, 72), "Synthetic page with image")
    pixel = pymupdf.Pixmap(pymupdf.csRGB, 1, 1, bytes([0, 0, 0]), False)
    page.insert_image(pymupdf.Rect(80, 100, 160, 180), pixmap=pixel)
    Path("with_image.pdf").write_bytes(document.tobytes())
```

Save it as `make_inspection_pdfs.py` and run `python make_inspection_pdfs.py`. Upload each file in Swagger. Compare `image_count` (0 versus 1) and the second file's `image_bounds`; both report page size and `has_text=true`. The response includes a safe filename, SHA-256 document ID, and one-based page numbers, but no image bytes or document text. The endpoint accepts image-only PDFs too. It rejects malformed or password-protected PDFs, oversize uploads, excessive pages, and more than **100 image occurrences per document**. It detects embedded raster images, not vector drawings or image meaning; it performs no OCR and stores nothing.

Page width, height, and `image_bounds` use **displayed page coordinates in PDF points**, after any page rotation, with the visible page's top-left as `(0, 0)`. PyMuPDF's image boxes start in unrotated coordinates; the inspector transforms them with the page rotation matrix and clips them to the displayed page. A partly off-page occurrence has one clipped visible box. A completely off-page occurrence still contributes to `image_count` but has no entry in `image_bounds`, so the two lengths may differ. Nonfinite or inverted source boxes are rejected. These boxes describe raster image placement only, not the image's meaning, its visible pixels, or vector artwork.

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

The historical Phase 8 service-free run passed **341 tests**, skipped **25** opt-in integration cases, and failed none. The focused visual modules passed **125 tests**. Both runs reported 13 nonfatal LangSmith deprecation warnings. These are [previously recorded results](SESSION_LOG.md#2026-11-01--phase-8-independent-review-remediation), not results from following this guide; use the commands above for your own environment.

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

These tests do not guarantee factual grounding, security against prompt injection, disconnected-network operation, Windows/macOS runtime behavior, or large-document performance. OCR, visual interpretation persistence, multimodal retrieval, and a frontend are not implemented. No cloud request was made during local E2E testing; separate cloud-provider development branches are unmerged and are not part of this stable-main local workflow. The [roadmap](ROADMAP.md) lists later work without treating it as current capability.

## Phase 8 visual testing on main

### Phase 9A.1 source metadata on this branch

With a separately authorized, backed-up version-2 schema and an already indexed text-bearing PDF, set `APP_ENV=development` and `ENABLE_VISUAL_ELEMENT_STORAGE_ENDPOINT=true`. Call `POST /collections/{collection_id}/documents/{document_id}/visual-elements/index`, then `GET /collections/{collection_id}/documents/{document_id}/visual-elements?limit=100&offset=0`. These routes read the original stored PDF and return bounded source metadata, including heuristic table cells; they neither call Ollama nor change text-RAG answers. Use synthetic data and a dedicated disposable schema for automated migration tests. Do not upgrade the real application database as a test. Collection UUIDs do not authorize callers. Phase 9A.2 interpretation and Phase 9B retrieval are future work.

These steps apply to merged Phase 8 on `main`, including vector/table inspection, `/documents/preview`, and visual analysis. Keep `uvicorn rag_ai.main:app --reload --host 127.0.0.1 --port 8000` bound to loopback. In your existing private `.env`, set `APP_ENV=development` and `ENABLE_DOCUMENT_VISUAL_INSPECTION_ENDPOINT=true`. `/documents/inspect` and `/documents/preview` then appear at `http://127.0.0.1:8000/docs`. These inspection/preview routes store nothing and do not call Ollama. Inspection returns a safe filename, original-byte SHA-256 ID, one-based PDF pages, displayed page size/text presence, raster occurrences/boxes, vector path counts/boxes/kinds, and heuristic table cells. Preview returns a PNG for one selected page or region; form fields `page_number` and optional `x0,y0,x1,y1` use displayed-page PDF points. In installed PyMuPDF 1.28.2, `page.get_pixmap(clip=...)` takes those displayed coordinates directly, including on rotated pages; the preview renderer does not derotate the requested box. Image/path metadata from inspection has a separate unrotated-to-displayed transformation. A box may be clipped to the page; wholly off-page raster/vector items count but have no visible box. A vector path is not a diagram, and image-based tables have no deterministic cell extraction.

To make a small synthetic document, save and run this script with the existing PyMuPDF installation:

```python
import pymupdf

pixel = pymupdf.Pixmap(pymupdf.csRGB, 20, 20, bytes([0, 0, 0] * 400), False)
with pymupdf.open() as doc:
    doc.new_page().insert_text((50, 50), "Synthetic engineering text")
    doc.new_page().insert_image(pymupdf.Rect(50, 50, 150, 150), pixmap=pixel)
    page = doc.new_page()
    page.draw_rect(pymupdf.Rect(50, 50, 150, 150))
    page.draw_line((50, 50), (150, 150))
    page = doc.new_page()
    for y in (50, 80, 110):
        page.draw_line((50, y), (250, y))
    for x in (50, 150, 250):
        page.draw_line((x, 50), (x, 110))
    for text, point in (("Relay", (60, 70)), ("Open", (160, 70)),
                        ("Pump", (60, 100)), ("On", (160, 100))):
        page.insert_text(point, text)
    doc.new_page().insert_image(pymupdf.Rect(50, 50, 150, 150), pixmap=pixel)
    with open("synthetic_visual.pdf", "wb") as output:
        output.write(doc.tobytes())
with open("synthetic_visual.png", "wb") as output:
    output.write(pixel.tobytes("png"))
with open("synthetic_visual.jpg", "wb") as output:
    output.write(pixel.tobytes("jpeg"))
```

Upload `synthetic_visual.pdf` to **POST `/documents/inspect`**: pages 2 and 5 contain raster occurrences, page 3 has vector paths, page 4 has a bordered table, and page 5 has no extractable text. Upload the same PDF to **POST `/documents/preview`** with `page_number=3`, then with region `x0=40,y0=40,x1=160,y1=160`; compare the returned PNG images. These form inputs select one page and at most one region. Image coordinates returned by `/images/analyze` use native pixels, not PDF points.

For optional local model analysis, run `ollama list` and select a compatible installed vision model. The recorded local check used `qwen3-vl:4b-instruct`; verify its availability on your machine or consult the [official Ollama vision guidance](https://docs.ollama.com/capabilities/vision) before installing a model. This repository does not download one. Set `VISION_MODEL` to the selected installed name and `ENABLE_DOCUMENT_VISUAL_ANALYSIS_ENDPOINT=true` in the private development configuration, then restart the backend. In Swagger, upload the PDF to **POST `/documents/analyze-visual`** with `page_number=3`, `task=describe_diagram`, and optionally the same region. Upload the PNG or JPEG to **POST `/images/analyze`** with `task=describe_figure`. The app renders/validates one image and sends it only to configured loopback Ollama; it returns model-generated description/observations/limitations plus application-owned SHA-256 source, filename, page, and region metadata. A model-generated description can be wrong, and no visual evidence enters PostgreSQL or text-RAG citations in this phase.

The service-free checks are `python -m pytest -q tests/test_visual_inspection.py tests/test_visual_structure.py tests/test_vision_provider.py tests/test_multimodal_api.py`. They use synthetic data and a mocked Ollama HTTP boundary; they do **not** verify live image understanding. If no model is installed, leave analysis disabled. A 404 means a route is disabled, a 4xx means an upload/page/region/limit was rejected, and 502 indicates local Ollama or structured-output failure. No database or cloud service is involved. Public deployment remains prohibited because these routes have no caller authorization and native PDF/image processing still needs stronger isolation.

On 2026-11-01, a separate local-only check used the already installed `qwen3-vl:4b-instruct` and the five-page synthetic `RAG_AI_Phase8_Visual_Test.pdf` through `/documents/analyze-visual`. All 15 requests returned HTTP 200 and valid structured evidence, with no generation-limit failures. They comprised six page/task requests (page 1 diagram and labels, page 2 diagram, page 3 figure, page 4 table, page 5 scanned page), five targeted value checks, one page-1 crop, and three content checks. The six primary requests took 3.81–7.20 seconds each. Against the source PDF, the page-2 sensor → controller → motor and power → controller connections, page-4 table rows, and page-5 inspection values were described correctly in sampled outputs. The tiny **Ollama provider** label in page 1 was omitted in a full-page label request and misread as **Qlana provider** in a crop. On page 5 the model also said there were five labeled fields although the source has four, while reading the four values correctly. Successful HTTP and schema validation therefore do not establish visual-reading accuracy. These observations are from one local model/fixture run, not a general quality benchmark. The API still returns a generic 502 for provider failures; server logs record only a fixed failure category, so check local model availability, output budget, and the relevant mocked tests without logging images or raw model output.
