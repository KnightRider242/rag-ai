# Security test plan

This plan covers the current FastAPI health endpoint and development-only parsing, chunking, local indexing, and semantic retrieval. It is a test plan and risk record, not a security audit or claim of production readiness.

## Threat model and protected assets

| Threat | Asset at risk | Current boundary or control |
| --- | --- | --- |
| Malformed or deceptive uploads | API availability and parser process | Extension/MIME checks, PDF structure validation, UTF-8 decoding, clear rejection |
| Oversized files, compressed PDFs, or excessive chunks | Memory, CPU, temporary disk | Configurable file, page, extracted-character, and chunk-count limits |
| Path traversal in filenames | Host filesystem | Filename reduced to a display basename; no upload path is opened |
| Document contents in logs or tracing | Confidential document text | No document-content logging or external tracing in this feature |
| Unauthorized use of development document endpoints | Uploaded content and service capacity | Parse and chunk routes absent by default and in production; explicit development opt-in only |
| Unauthorized collection indexing | Original bytes, chunks, vectors, database capacity | Index route requires an additional development flag and a UUID collection key; no authentication or collection authorization exists, so localhost-only use is mandatory |
| Wrong or changed embedding model | Vector compatibility and future retrieval quality | Compare returned dimensions, model name, and Ollama digest; record them per collection and reject conflicts |
| Local model endpoint redirection or oversized response | Document confidentiality, memory | Provider construction validates loopback HTTP; redirects and proxies are disabled and JSON responses are size-bounded |
| Accumulated embedding vectors | API memory and database availability | At most 16 vectors per batch; measured dimensions and total element budget checked before remaining batches; temporary database staging and atomic final write |
| Cross-collection or incompatible retrieval | Stored chunk text and ranking correctness | Parameterized collection-scoped joins; model name/digest/dimension checks against indexed vectors; mixed-dimension candidates filtered before cosine distance |
| Unauthorized search or prompt injection | Stored document text and future answers | Search route absent by default and in production; localhost-only development use. No authentication or document authorization exists; retrieved text remains untrusted data and is not passed to an LLM |

Protected assets are uploaded document bytes and extracted text, future tenant documents and access policies, provider credentials, database credentials, and service availability.

## Trust boundaries

1. Client to multipart parser: filename, MIME type, and bytes are untrusted. MIME and extension checks are declarations, so PDF bytes are also validated by PyMuPDF.
2. Multipart parser to document extractor: the file-size limit is enforced when the route reads the upload. The multipart framework may spool content to temporary disk before this check.
3. Extractor to API response: parse/chunk routes return content to the caller. The index route returns identifiers and model metadata and persists original bytes, text, and vectors in the local database; no content logging, cloud call, or tracing is implemented.
4. Local indexing crosses the Ollama HTTP and PostgreSQL boundaries. The provider is restricted to loopback, while the database is bound to localhost by Compose. Collection IDs scope records but do not authorize callers.
5. Local retrieval sends the query to loopback Ollama, then reads ranked chunk text through a collection-filtered PostgreSQL query. The returned text is sensitive and untrusted. Cloud-provider boundaries still require explicit data-handling decisions before implementation.

## Security testing strategy

- Use small synthetic files and no real business documents or credentials.
- For each new attack surface, add negative tests that verify rejection codes and that protected side effects do not occur.
- Run the complete pytest suite, review code paths and Git changes, scan changes for secrets, and record any unresolved risks in the session report.
- Keep functional tests for valid PDF, TXT, Markdown, `/health`, and response metadata to catch regressions.
- Do not recommend merging with an unresolved critical or high-severity security issue.

## Current feature requirements and checks

The ingestion tests cover malformed and spoofed PDFs, unsupported extensions and MIME types, invalid UTF-8, empty documents, image-only PDFs, oversized uploads, page and extracted-text limits, safe filenames, literal treatment of document instructions, and absence of document text in captured logs. Chunking tests cover invalid configuration, maximum chunk counts, deterministic IDs, page isolation, inert document instructions, no content logging, and inherited ingestion validation. Parse and chunk routes process files without executing or saving them; the separately gated index route persists them locally.

The parse and chunk routes are registered only when `APP_ENV=development` and `ENABLE_DOCUMENT_PARSE_ENDPOINT=true`. They are absent by default and in `test` or `production`. Production configuration that attempts to enable them fails at startup. Tests verify route absence in OpenAPI and HTTP 404 responses, explicit development access, and rejection of the invalid production configuration.

The index route additionally requires `ENABLE_DOCUMENT_INDEX_ENDPOINT=true` and a configured database URL. It is absent by default and outside development; production configuration that tries to enable it fails. Tests cover the gate, provider-level loopback validation, rejected redirects, invalid vector responses, changed model digests, dimension conflicts, element-budget rejection after the first dimension probe, bounded batches, rollback after partial staging or final-write failure, collection-scoped database keys, and real local parse → chunk → embed → persist behavior when PostgreSQL and Ollama are available.

The search route separately requires `APP_ENV=development` and `ENABLE_DOCUMENT_RETRIEVAL_ENDPOINT=true`; production opt-in is rejected. Unit/API tests cover invalid queries and `top_k`, malformed/zero/non-finite query vectors, one local embedding request, controlled failures, provider closure, no cloud fallback, and no query text in captured logs or HTTP errors. Opt-in disposable-schema tests cover collection/model/digest/dimension isolation, mixed vector dimensions, zero stored vectors, deterministic ranking, metadata, schema rejection, and database error sanitization; all seven retrieval integration tests passed in the dedicated local test database on 2026-10-29. The expanded synthetic Ollama evaluation checks first-rank and top-three results among nine candidates; its four queries passed the preset quality expectations in that local run. These tests do not establish production authorization or general retrieval quality.

The development routes have no authentication or tenant permissions and must be used only on a trusted machine through localhost. A deployment that accepts documents must add authentication and authorization before exposing a document route. It must also enforce a request-size cap at the ingress proxy; the route-level limit alone does not constrain multipart parsing or temporary spooling. These remain **high-severity blockers for public or production document ingestion**. The environment gate is not an authentication mechanism and does not itself prevent a development server from binding to a public interface.

## Feature-specific security requirements

- **Ingestion and chunking (current):** Reject deceptive or malformed input, bound file size and extraction/chunk creation work, preserve PDF page boundaries, treat content as data, avoid filesystem paths and content logging, and return clear errors.
- **Local indexing (current):** Require separate development opt-in, validate the loopback Ollama URL at the provider boundary, reject redirects, bound chunk and embedding sizes, reject malformed/model-mismatched vectors, store model name/digest/dimensions, stage bounded batches outside the final transaction, use parameterized collection-scoped SQL, and avoid leaking content or connection details in API errors. Real database and Ollama tests use synthetic content and opt-in environment settings.
- **Local retrieval (current):** Restrict route registration to explicit development opt-in, use the loopback provider, validate the query vector, filter eligible rows in a collection-scoped materialized CTE before cosine ranking, avoid query/content/vector logging, and return controlled errors. The collection filter prevents accidental cross-collection ranking but is not user authorization.
- **Authorized retrieval (future):** Authenticate callers, authorize every document access, scope database and vector queries by tenant and document permissions, and test cross-tenant denial.
- **LLM, embedding, and tracing integrations (future):** Require authorization before transferring confidential content, enforce provider data-handling controls, prevent retrieved instructions from gaining system authority, and exclude sensitive content from traces by default.
- **Local/cloud and VLM providers (future):** Test that endpoint configuration and provider selection respect collection policy on Windows, macOS, and Linux; reject unauthorized cloud calls and any automatic cloud fallback, including after a local Ollama failure. Cover visual content and metadata as sensitive data.

## Known limitations and planned improvements

- PyMuPDF processes untrusted PDF bytes in-process. Keep the library updated; consider process isolation and CPU/time limits before public exposure.
- The development gate controls route registration, not network binding. Run the development server with `--host 127.0.0.1` and do not enable the route on a shared or public host. A peer-address or Host-header check cannot guarantee network isolation when a local reverse proxy forwards public traffic, so no application-level check is treated as a substitute for binding and network controls.
- Scanned PDFs are detected when no text is extractable and images are present. OCR is not implemented.
- Page and extracted-character limits reduce resource exhaustion risk, but PDF parsing can still consume CPU or memory before those limits are reached.
- Chunk sizes are limited to 64–10,000 characters, overlap to at most half the chunk size, and total chunks to a validated configured maximum. A preflight check rejects a source before splitting when its non-whitespace characters cannot fit in the remaining chunk budget. Tests verify that this rejection occurs before the splitter runs. LangChain still materializes a list for one source page or text document before the final count check. The existing extracted-character cap bounds input, but CPU and memory can still rise for whitespace-heavy or separator-sensitive text; use process memory and time limits before wider exposure.
- Python 3.14 test runs emit deprecation warnings from the transitive `langsmith` package (`_beta_decorator.py`, use of `asyncio.iscoroutinefunction`). No application code or tracing configuration causes the warning. Monitor upstream compatibility; no dependency change was needed for this phase.
- The parser returns extracted text to the caller. It must not be exposed to other users through future storage, caching, traces, or logs.
- SHA-256 document IDs are deterministic and can reveal that two uploads have identical bytes. Do not treat an ID as an access token or expose cross-tenant existence through ID lookups.
- The Phase 4 index route stores original bytes, extracted chunks, and vectors in PostgreSQL but has no collection authorization, HTTP deletion endpoint, automatic retention expiry, or encryption-at-rest control. Storage-layer deletion is collection-scoped and cascades to derived rows, but backups and snapshots may retain copies. Keep indexing development-only on a trusted localhost machine. A collection UUID does not grant access. Do not expose the route publicly until authentication, authorization, ingress limits, least-privilege database roles, and resource isolation are implemented and tested.
- Explicit versioned SQL migrations track checksums and reject unknown or changed histories. An unversioned Phase 4 schema must be backed up and explicitly adopted after shape verification; partial or incompatible schemas are rejected. Integration tests use disposable schemas so they do not alter existing application records. The Compose database role remains a broad development role. The element budget is checked after one bounded provider batch because dimensions are measured from the response; that first batch still consumes local inference resources. Temporary staging avoids retaining all vectors in Python and keeps the permanent write transaction short, but holds one database connection during embedding and uses temporary database space. Large uploads and model execution can still consume CPU, RAM, and disk before application limits fully apply.
- Phase 5 retrieval returns stored text through an unauthenticated development route. UUIDs are not secrets or authorization tokens. Public or shared use is blocked until caller identity, collection/document permissions, ingress and query-rate controls, database least privilege, and cross-tenant authorization tests exist. Filtering by collection protects against accidental SQL join leakage, not a caller choosing another collection UUID. Exact search may scan many stored vectors; benchmark and bound database work before wider use. A missing collection still requires a local query embedding before the store can return an empty list.
- Before LLM, embedding, or tracing integrations, require explicit authorization for confidential data transfer, provider retention controls, and tests that prevent prompt injection from gaining system authority.
