# Architecture decisions

These decisions describe the approved direction as of 2026-10-28. They guide future work; they do not imply that planned integrations exist. Revisit a decision on its feature branch if implementation evidence warrants a change, and record the reason.

## Current baseline (Phases 1–6 feature branch)

The FastAPI backend exposes `/health` and, with explicit development configuration, `/documents/parse`, `/documents/chunk`, `/collections/{collection_id}/documents/index`, `/collections/{collection_id}/search`, and `/collections/{collection_id}/answer`. PDF, TXT, and Markdown extraction, LangChain text splitting, local Ollama embeddings and generation, PostgreSQL/pgvector persistence, exact semantic search, and single-turn cited answers are implemented on this branch. Development routes have no authentication or collection authorization and are restricted to trusted localhost use. There is no cloud provider, vision model, agent, MCP tool, or frontend.

## AD-01 — Local-first and explicit processing location

**Decision:** Develop and verify the first embedding and answer pipelines with local services. Support local Ollama and later explicitly authorized cloud AI providers through a policy based on the collection's data classification and caller authorization. Confidential documents must never be sent to cloud LLM, embedding, or vision providers through automatic fallback.

**Reason:** Processing location affects confidentiality and data-handling obligations. A local-first path keeps early integration tests independent of external credentials and services.

**Implementation status:** Phases 4 and 6 implement loopback-only Ollama embedding and generation providers with no cloud fallback. General data-classification policy and cloud-provider authorization are not implemented.

## AD-02 — Provider-independent capabilities

**Decision:** Define separate, small interfaces for embedding, LLM generation, and vision-language model (VLM) understanding when each capability is introduced. Keep local Ollama and later cloud-provider clients and configuration behind those boundaries; do not add speculative abstractions or SDKs before use. The Ollama endpoint must be configurable by the application. Provider selection and failures must be explicit and observable without logging document content.

**Reason:** Provider changes should not alter document ownership, provenance, or authorization behavior.

**Implementation status:** Small embedding and generation provider interfaces and local Ollama implementations exist. The VLM interface remains planned for Phase 8; multiple provider implementations remain planned for Phase 7.

## AD-03 — Collection-scoped persistence and vector compatibility

**Decision:** When storage is introduced, associate every original document, extracted unit, chunk, and vector with a collection and enforce collection access on every read and write. Record the embedding model identifier, relevant model version, vector dimensionality, and processing configuration. Reject or isolate incompatible vectors and define re-embedding/migration procedures before changing models.

**Reason:** Collection boundaries prevent accidental cross-tenant access; model metadata prevents silently comparing vectors from incompatible spaces.

**Implementation status:** Phase 4 stores collection-scoped documents, chunks, vectors, model names, Ollama digests, measured dimensions, and chunking configuration. Database keys and foreign keys keep records within a collection. Explicit versioned SQL migrations track the schema; an existing unversioned Phase 4 schema requires verified adoption. Storage-layer document deletion removes a collection's original bytes and derived rows transactionally, with no automatic retention expiry or HTTP deletion route. Phase 5 applies collection/model/digest/dimension filters to exact search. Caller and document authorization remain planned; a collection UUID and SHA-256 IDs are not access tokens. Re-indexing one document replaces its prior chunks and vectors in that collection.

## AD-04 — Preserve sources and trace citations

**Decision:** Future persistence must retain authorized access to the original source and a stable mapping from derived text or visual evidence back to document, page, and, where applicable, region or table. Citations must resolve to that source mapping. Treat multimodal PDFs, images, figures, diagrams, and tables as first-class future evidence rather than inferring their content from text alone.

**Reason:** Answers need verifiable evidence, and later multimodal features need source provenance.

**Implementation status:** Phase 4 retains original source bytes, filenames, document IDs, chunk IDs, and PDF page numbers in the local database. Phase 6 resolves request-local labels to these retrieved source locations. Visual provenance remains planned for Phases 8–9; citation resolution does not establish factual entailment.

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
