# Security test plan

This plan covers the current FastAPI health endpoint and development-only document parser. It is a test plan and risk record, not a security audit or claim of production readiness.

## Threat model and protected assets

| Threat | Asset at risk | Current boundary or control |
| --- | --- | --- |
| Malformed or deceptive uploads | API availability and parser process | Extension/MIME checks, PDF structure validation, UTF-8 decoding, clear rejection |
| Oversized files or compressed PDFs with many pages/text | Memory, CPU, temporary disk | Configurable file, page, and extracted-character limits |
| Path traversal in filenames | Host filesystem | Filename reduced to a display basename; no upload path is opened |
| Document contents in logs or tracing | Confidential document text | No document-content logging or external tracing in this feature |
| Unauthorized use of parsing endpoint | Uploaded content and service capacity | Route absent by default and in production; explicit development opt-in only |
| Future prompt injection or cross-tenant retrieval | Answers and tenant documents | No retrieval, LLM calls, or document persistence exists yet |

Protected assets are uploaded document bytes and extracted text, future tenant documents and access policies, provider credentials, database credentials, and service availability.

## Trust boundaries

1. Client to multipart parser: filename, MIME type, and bytes are untrusted. MIME and extension checks are declarations, so PDF bytes are also validated by PyMuPDF.
2. Multipart parser to document extractor: the file-size limit is enforced when the route reads the upload. The multipart framework may spool content to temporary disk before this check.
3. Extractor to API response: extracted text is returned to the requesting client only; no persistence, logging, external provider call, or tracing is implemented.
4. Future database, retrieval, and provider boundaries require tenant authorization and data-handling decisions before implementation.

## Security testing strategy

- Use small synthetic files and no real business documents or credentials.
- For each new attack surface, add negative tests that verify rejection codes and that protected side effects do not occur.
- Run the complete pytest suite, review code paths and Git changes, scan changes for secrets, and record any unresolved risks in the session report.
- Keep functional tests for valid PDF, TXT, Markdown, `/health`, and response metadata to catch regressions.
- Do not recommend merging with an unresolved critical or high-severity security issue.

## Current feature requirements and checks

The ingestion tests cover malformed and spoofed PDFs, unsupported extensions and MIME types, invalid UTF-8, empty documents, image-only PDFs, oversized uploads, page and extracted-text limits, safe filenames, literal treatment of document instructions, and absence of document text in captured logs. Files are processed without executing their contents or saving them permanently.

The parse route is registered only when `APP_ENV=development` and `ENABLE_DOCUMENT_PARSE_ENDPOINT=true`. It is absent by default and in `test` or `production`. Production configuration that attempts to enable it fails at startup. Tests verify route absence in OpenAPI and HTTP 404 responses, explicit development access, and rejection of the invalid production configuration.

The development route has no authentication or tenant permissions and must be used only on a trusted machine through localhost. A deployment that accepts documents must add authentication and authorization before exposing a document route. It must also enforce a request-size cap at the ingress proxy; the route-level limit alone does not constrain multipart parsing or temporary spooling. These remain **high-severity blockers for public or production document ingestion**. The environment gate is not an authentication mechanism and does not itself prevent a development server from binding to a public interface.

## Feature-specific security requirements

- **Ingestion (current):** Reject deceptive or malformed input, bound file size and extraction work, treat content as data, avoid filesystem paths and content logging, and return clear errors.
- **Persistence and retrieval (future):** Authenticate callers, authorize every document access, scope database and vector queries by tenant and document permissions, and test cross-tenant denial.
- **LLM, embedding, and tracing integrations (future):** Require authorization before transferring confidential content, enforce provider data-handling controls, prevent retrieved instructions from gaining system authority, and exclude sensitive content from traces by default.

## Known limitations and planned improvements

- PyMuPDF processes untrusted PDF bytes in-process. Keep the library updated; consider process isolation and CPU/time limits before public exposure.
- The development gate controls route registration, not network binding. Run the development server on `127.0.0.1` and do not enable the route on a shared or public host.
- Scanned PDFs are detected when no text is extractable and images are present. OCR is not implemented.
- Page and extracted-character limits reduce resource exhaustion risk, but PDF parsing can still consume CPU or memory before those limits are reached.
- The parser returns extracted text to the caller. It must not be exposed to other users through future storage, caching, traces, or logs.
- SHA-256 document IDs are deterministic and can reveal that two uploads have identical bytes. Do not treat an ID as an access token or expose cross-tenant existence through ID lookups.
- Before persistence or retrieval, implement document ownership, tenant-scoped queries, vector filters, and authorization tests that prove cross-tenant access is denied.
- Before LLM, embedding, or tracing integrations, require explicit authorization for confidential data transfer, provider retention controls, and tests that prevent prompt injection from gaining system authority.
