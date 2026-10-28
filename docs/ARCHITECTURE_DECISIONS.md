# Architecture decisions

These decisions describe the approved direction as of 2026-10-28. They guide future work; they do not imply that planned integrations exist. Revisit a decision on its feature branch if implementation evidence warrants a change, and record the reason.

## Current baseline (Phases 1–3)

The FastAPI backend exposes `/health` and, with explicit development configuration, in-memory `/documents/parse` and `/documents/chunk` routes. PDF, TXT, and Markdown extraction and LangChain text splitting are implemented. PostgreSQL with pgvector is configured for local development, but the API has no database integration. There are no embeddings, retrieval, LLM calls, vision models, agents, MCP tools, or frontend yet.

## AD-01 — Local-first and explicit processing location

**Decision:** Develop and verify the first embedding and answer pipelines with local services. Support local Ollama and later explicitly authorized cloud AI providers through a policy based on the collection's data classification and caller authorization. Confidential documents must never be sent to cloud LLM, embedding, or vision providers through automatic fallback.

**Reason:** Processing location affects confidentiality and data-handling obligations. A local-first path keeps early integration tests independent of external credentials and services.

**Implementation status:** Policy direction approved; provider calls and policy enforcement are not implemented.

## AD-02 — Provider-independent capabilities

**Decision:** Define separate, small interfaces for embedding, LLM generation, and vision-language model (VLM) understanding when each capability is introduced. Keep local Ollama and later cloud-provider clients and configuration behind those boundaries; do not add speculative abstractions or SDKs before use. The Ollama endpoint must be configurable by the application. Provider selection and failures must be explicit and observable without logging document content.

**Reason:** Provider changes should not alter document ownership, provenance, or authorization behavior.

**Implementation status:** Planned for Phases 4, 7, and 8 as each capability arrives.

## AD-03 — Collection-scoped persistence and vector compatibility

**Decision:** When storage is introduced, associate every original document, extracted unit, chunk, and vector with a collection and enforce collection access on every read and write. Record the embedding model identifier, relevant model version, vector dimensionality, and processing configuration. Reject or isolate incompatible vectors and define re-embedding/migration procedures before changing models.

**Reason:** Collection boundaries prevent accidental cross-tenant access; model metadata prevents silently comparing vectors from incompatible spaces.

**Implementation status:** Planned for Phases 4–5. Current SHA-256 document and chunk IDs are deterministic identifiers, not access tokens.

## AD-04 — Preserve sources and trace citations

**Decision:** Future persistence must retain authorized access to the original source and a stable mapping from derived text or visual evidence back to document, page, and, where applicable, region or table. Citations must resolve to that source mapping. Treat multimodal PDFs, images, figures, diagrams, and tables as first-class future evidence rather than inferring their content from text alone.

**Reason:** Answers need verifiable evidence, and later multimodal features need source provenance.

**Implementation status:** Current parsing preserves PDF page numbers and chunk IDs but stores no original files or citations. Persistent provenance is planned for Phases 4, 6, 8, and 9.

## AD-05 — Separate evaluation data and security gates

**Decision:** Maintain distinct development and evaluation datasets with documented rights and handling rules. Use synthetic data in automated tests. Plan automated tests on Windows, macOS, and Linux as platform-specific integrations arrive. Every feature branch needs relevant functional and negative/security tests, a secret review, and an explicit record of unresolved risks. Critical or high-severity unresolved issues block a merge recommendation.

**Reason:** Separation reduces evaluation leakage; repeated security review catches new trust boundaries as capabilities expand.

**Implementation status:** Synthetic automated tests and branch/reporting rules exist. Evaluation datasets and their governance are planned for Phase 5.

## AD-06 — Incremental branches and deployment boundary

**Decision:** Implement one scoped capability per dedicated branch, with documentation and tests. Keep development document routes disabled by default and confined to trusted localhost use. Before public document processing, add authentication, authorization, ingress request-size limits, and stronger PDF/resource isolation. Deployment hardening is explicitly planned for Phase 13, but any earlier public exposure must satisfy these controls first.

**Reason:** The current development gate does not authenticate callers or constrain the network interface, and multipart parsing can spool data before route-level checks.

**Implementation status:** Branch workflow and development route gate exist. Public deployment controls are not implemented.

## AD-07 — Cross-platform local runtime

**Decision:** Target the same Python backend behavior on Windows, macOS, and Linux, with platform-specific setup documentation and future automated cross-platform verification. Install Ollama natively on the host for planned local inference, while Docker Compose runs PostgreSQL with pgvector. Support CPU inference and GPU acceleration only for configurations supported by Ollama and the host hardware; the application must not assume a GPU or fixed Ollama URL.

**Reason:** Local development should work across supported operating systems without coupling database deployment to inference hardware. Platform and accelerator differences must be tested rather than assumed.

**Implementation status:** Python and Docker Compose setup exist. Native Ollama integration, its configurable application endpoint, accelerator validation, and cross-platform CI are planned; they are not implemented.
