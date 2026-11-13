# Architecture decisions

These decisions describe the approved direction as of 2026-10-28. They guide future work; they do not imply that planned integrations exist. Revisit a decision on its feature branch if implementation evidence warrants a change, and record the reason.

## Current baseline (`main` and Phase 9C feature branch)

Stable `main` includes Phases 1–6, 8, 9A.1, 9A.2, 9B.1, and 9B.2: text RAG, visual processing, source metadata, immutable local interpretations, and combined retrieval. This Phase 9C branch adds locally generated multimodal answers. Development routes have no authentication or collection authorization and are restricted to trusted localhost use. There is no cloud provider, agent, MCP tool, or frontend.

## AD-01 — Local-first and explicit processing location

**Decision:** Develop and verify the first embedding and answer pipelines with local services. Support local Ollama and later explicitly authorized cloud AI providers through a policy based on the collection's data classification and caller authorization. Confidential documents must never be sent to cloud LLM, embedding, or vision providers through automatic fallback.

**Reason:** Processing location affects confidentiality and data-handling obligations. A local-first path keeps early integration tests independent of external credentials and services.

**Implementation status:** Phases 4 and 6 implement loopback-only Ollama embedding and generation providers; Phase 8 adds a loopback-only vision provider on `main`. None automatically falls back to a cloud service. General data-classification policy and cloud-provider authorization are not implemented.

## AD-02 — Provider-independent capabilities

**Decision:** Define separate, small interfaces for embedding, LLM generation, and vision-language model (VLM) understanding when each capability is introduced. Keep local Ollama and later cloud-provider clients and configuration behind those boundaries; do not add speculative abstractions or SDKs before use. The Ollama endpoint must be configurable by the application. Provider selection and failures must be explicit and observable without logging document content.

**Reason:** Provider changes should not alter document ownership, provenance, or authorization behavior.

**Implementation status:** Small embedding and generation provider interfaces and local Ollama implementations exist on stable `main`. Phase 8 added a separate local VLM interface on `main`. A small local `qwen3-vl:4b-instruct` evaluation completed 15 HTTP 200 requests without generation-limit failures; it also exposed visual-reading errors and does not establish general accuracy. Cloud provider implementations remain planned.

## AD-03 — Collection-scoped persistence and vector compatibility

**Decision:** When storage is introduced, associate every original document, extracted unit, chunk, and vector with a collection and enforce collection access on every read and write. Record the embedding model identifier, relevant model version, vector dimensionality, and processing configuration. Reject or isolate incompatible vectors and define re-embedding/migration procedures before changing models.

**Reason:** Collection boundaries prevent accidental cross-tenant access; model metadata prevents silently comparing vectors from incompatible spaces.

**Implementation status:** Phase 4 stores collection-scoped documents, chunks, vectors, model names, Ollama digests, measured dimensions, and chunking configuration. Database keys and foreign keys keep records within a collection. Explicit versioned SQL migrations track the schema; an existing unversioned Phase 4 schema requires verified adoption. Storage-layer document deletion removes a collection's original bytes and derived rows transactionally, with no automatic retention expiry or HTTP deletion route. Phase 5 applies collection/model/digest/dimension filters to exact search. Caller and document authorization remain planned; a collection UUID and SHA-256 IDs are not access tokens. Re-indexing one document replaces its prior chunks and vectors in that collection.

## AD-04 — Preserve sources and trace citations

**Decision:** Future persistence must retain authorized access to the original source and a stable mapping from derived text or visual evidence back to document, page, and, where applicable, region or table. Citations must resolve to that source mapping. Treat multimodal PDFs, images, figures, diagrams, and tables as first-class future evidence rather than inferring their content from text alone.

**Reason:** Answers need verifiable evidence, and later multimodal features need source provenance.

**Implementation status:** Phase 4 retains original source bytes, filenames, document IDs, chunk IDs, and PDF page numbers in the local database. Phase 6 resolves request-local labels to these retrieved source locations. Phase 8 assigns upload hashes and PDF page/region or standalone-image coordinates to returned visual observations without storing those interpretations. Phases 9A.1 and 9A.2 persist source regions and unverified local interpretations. Phase 9B.1 makes those interpretations searchable; Phase 9B.2 combines retrieval. Phase 9C resolves text and visual answer citations from application-owned metadata and checks cited records after generation. Citation resolution does not establish factual entailment.

## AD-05 — Separate evaluation data and security gates

**Decision:** Maintain distinct development and evaluation datasets with documented rights and handling rules. Use synthetic data in automated tests. Plan automated tests on Windows, macOS, and Linux as platform-specific integrations arrive. Every feature branch needs relevant functional and negative/security tests, a secret review, and an explicit record of unresolved risks. Critical or high-severity unresolved issues block a merge recommendation.

**Reason:** Separation reduces evaluation leakage; repeated security review catches new trust boundaries as capabilities expand.

**Implementation status:** Synthetic automated tests and branch/reporting rules exist. A separate, nine-document and four-question synthetic retrieval evaluation dataset tests first-rank and top-three behavior; it is not a production quality benchmark. Governance for future real datasets remains planned.

## AD-06 — Incremental branches and deployment boundary

**Decision:** Implement one scoped capability per dedicated branch, with documentation and tests. Keep development document routes disabled by default and confined to trusted localhost use. Before public document processing, add authentication, authorization, ingress request-size limits, and stronger PDF/resource isolation. Deployment hardening is explicitly planned for Phase 13, but any earlier public exposure must satisfy these controls first.

**Reason:** The current development gate does not authenticate callers or constrain the network interface, and multipart parsing can spool data before route-level checks.

**Implementation status:** Branch workflow and development route gate exist. Public deployment controls are not implemented.

## AD-07 — Cross-platform local runtime

**Decision:** Target the same Python backend behavior on Windows, macOS, and Linux, with platform-specific setup documentation and future automated cross-platform verification. Install Ollama natively on the host for planned local inference, while Docker Compose runs PostgreSQL with pgvector. Support CPU inference and GPU acceleration only for configurations supported by Ollama and the host hardware; the application must not assume a GPU or fixed Ollama URL.

**Reason:** Local development should work across supported operating systems without coupling database deployment to inference hardware. Platform and accelerator differences must be tested rather than assumed.

**Implementation status:** The Python backend uses a configurable loopback Ollama endpoint and the existing Docker Compose pgvector database. The local integration was verified on Nobara Linux 44 with `embeddinggemma:300m` and a 768-dimensional provider response. Windows/macOS runtime behavior, accelerator validation, and cross-platform CI are not yet verified.

## AD-08 — Exact, compatibility-checked retrieval before approximate search

**Decision:** Reuse `EmbeddingProvider` for one locally embedded query and `PostgresIndexStore` for exact pgvector cosine ranking. Check model name, Ollama digest, and measured dimensions against actual indexed embeddings. Use one read-only repeatable-read snapshot for the compatibility check and ranking. Materialize collection-scoped, same-dimension, nonzero vectors and IDs before the cosine operation; fetch text only for the top ranked IDs and break equal-distance ties by stable identifiers. Expose a separately gated development route and a thin LangChain Core adapter over the same service. Evaluate actual ranked chunk IDs with synthetic cases before considering thresholds or approximate indexes.

**Reason:** The existing unbounded-dimension vector column can contain different models and sizes. Filtering before distance evaluation avoids invalid comparisons, while a stable snapshot prevents a concurrent index update from changing compatibility between checks. Reusing the existing provider and store keeps local-only network and collection rules in one place. Exact search is the simplest correctness baseline; no performance evidence yet justifies HNSW or IVFFlat.

**Implementation status:** Phase 5 implements the service, store search, API, adapter, and evaluation metrics without a schema migration. Local PostgreSQL retrieval integration passed (7 tests), and the expanded Ollama evaluation ranked all four labeled sources first on 2026-10-29. There is no caller/document authorization, public route, similarity cutoff, or production retrieval benchmark.

## AD-09 — Local structured generation over bounded evidence

**Decision:** Keep generation separate from embedding and retrieval. `RAGService` reuses `RetrievalService`, supplies a bounded, explicitly delimited evidence context to a loopback-only Ollama chat provider, and resolves model-cited request labels against application-owned retrieval results. Use a strict structured answer contract with explicit `answered` and `insufficient_evidence` states. Reject invalid output and unknown citations; abstain without generation when retrieval is empty.

**Reason:** This retains Phase 5 collection/model protections while making prompt size, provider access, citation provenance, and abstention independently testable. A separate generation interface leaves room for later approved provider choices without a speculative routing framework.

**Implementation status:** Phase 6 implements single-turn local Qwen3 generation behind a separate development flag. Citation validation checks references, not whether claims are entailed by evidence. Prompt injection, insufficient evidence, and factual grounding remain model-behavior risks; public access is blocked by missing caller and collection authorization.

## AD-10 — Separate bounded visual evidence from text RAG (Phase 8)

**Decision:** Use PyMuPDF for deterministic raster occurrence, vector-path, heuristic table, and page/region rendering metadata. Keep PDF boxes in displayed-page points and standalone-image boxes in native pixels. A small loopback-only Ollama vision provider returns validated descriptions. The application, not the model, assigns SHA-256 source identity, filename, page, and region. Keep these visual observations out of PostgreSQL indexing and text-RAG citations until a separately designed Phase 9.

**Reason:** Detection, cell extraction, model interpretation, and source provenance have different trust levels. One bounded region and a structured response limit the data and work per request. The split permits source checks without treating model descriptions as verified facts.

**Status and limits:** Merged into `main` with synthetic and mocked tests. The vision provider rejects missing or non-boolean completion flags, non-`stop` completion reasons, invalid structured output, redirects, and oversized responses; it closes the HTTP client after each request. This checks the response contract, not whether a description is true. A later small local check of `RAG_AI_Phase8_Visual_Test.pdf` using `qwen3-vl:4b-instruct` returned HTTP 200 for all 15 requests, with no generation-limit failures. In one crop it misread “Ollama provider” as “Qlana provider”; it also reported five fields where the scanned page contained four. These observations are not general accuracy metrics. PyMuPDF table finding is heuristic, path counts are not diagram counts, scanned PDFs have no deterministic OCR, and native PDF/image processing still needs process-level isolation before public use.

## AD-11 — Persist bounded visual source occurrences (Phase 9A.1)

**Decision:** Keep original PDF bytes in `indexed_documents`. Derive deterministic IDs from source SHA-256, extractor name/version, page/type/occurrence, and four-decimal displayed-page coordinates. Store raster placements, heuristic tables, and visible vector-path occurrences under a composite collection/document foreign key. Replace one extractor's complete set under a document-row lock after verifying source bytes before and during the write. Do not invoke a vision model in this path.

**Reason:** A PDF can reuse an image or have overlapping elements; coordinates alone do not identify occurrences. The composite key, verified source hash, and transaction keep evidence attached to the correct document while preserving text chunks and embeddings. On-demand rendering can reuse original bytes without storing duplicate image files.

**Limits:** Phase 9A.1 stores no model interpretations, semantic diagrams, standalone image uploads, visual vectors, or visual citations. Changing extractor version removes the prior active set for that extractor. Native PDF parsing needs process isolation before wider deployment, and collection scoping is not caller authorization.

## AD-12 — Persist unverified local interpretations separately (Phase 9A.2)

**Decision:** Store each model interpretation in an append-only row linked to the complete visual-element key. Record a versioned predefined task prompt, local model name and `/api/tags` digest, render scale, PNG hash, and bounded structured output. Check the model digest before and after inference. Recheck PDF and element state under short database row locks before insertion. A caller-supplied UUID key makes retries idempotent.

**Reason:** Source geometry and model descriptions have different trust levels. Rendering and inference can be slow, so the final lock must cover only source validation and insertion. Immutable rows preserve interpretation history; source deletion cascades it.

**Status and limits:** Merged on `main`. Ollama's chat API does not atomically bind a response to a digest, leaving a narrow revision-swap race between tag checks. Interpretations are explicitly unverified and are not used by text-RAG answers. Development routes have no caller authorization, and native PDF processing needs stronger isolation before public use.

## AD-13 — Index interpretation text separately from text RAG (Phase 9B.1)

**Decision:** Use the existing local embedding provider and embedding-space contract to vectorize only a bounded, versioned representation of persisted interpretation descriptions and observations. Store each vector under its exact collection/document/element/interpretation key and model revision. Search visual vectors in a separate collection-scoped snapshot, checking name, revision, dimensions, and representation version before cosine ranking. Return the original PDF region with the unverified interpretation.

**Reason:** Interpretation text is searchable without rerunning vision inference or mixing its trust level with source metadata and verified facts. Exact foreign keys and short final write transactions keep derived vectors tied to surviving source rows. A separate search route preserves the existing text search contract.

**Status and limits:** Merged on `main`. This searches model-generated text, not image pixels. Development routes remain unauthenticated and unsuitable for public use.

## AD-14 — Fuse text and visual retrieval in one snapshot (Phase 9B.2)

**Decision:** Embed the question once, reject incompatible populated embedding spaces, and run both exact cosine searches in one PostgreSQL repeatable-read, read-only transaction. Fuse bounded candidate ranks with `1 / (60 + rank)`, prefer text on ties, and retain one interpretation per visual element. Return typed source provenance and original cosine distances.

**Reason:** One snapshot keeps both candidate sets consistent during concurrent source changes. Rank fusion allows unlike cosine distributions to share a deterministic result list without treating either distance as a probability.

**Status and limits:** Merged on `main`. RRF is a heuristic: its score is not factual confidence, and a fixed candidate pool can miss relevant evidence. Interpretations remain unverified; the route does not generate answers or perform image-to-image search.

## AD-15 — Resolve multimodal answer citations from fresh source records (Phase 9C)

**Decision:** Keep text-only answers unchanged. A separate development route uses one combined retrieval call, a bounded quoted context, and the existing strict local structured-generation contract. The model emits only request-local `T` and `V` labels. The application maps labels to exact text chunks or stored visual interpretations, then checks cited records in a fresh, collection-scoped PostgreSQL snapshot before returning them.

**Reason:** Model-authored IDs or coordinates cannot establish provenance. A source may be deleted or changed while generation runs. Visual descriptions are unverified model output and must remain labeled as such in the response.

**Limits:** A valid citation establishes source identity at the verification snapshot, not semantic factual correctness or a durable post-response lock. A source can change after that check. Development route gates are not authentication.
