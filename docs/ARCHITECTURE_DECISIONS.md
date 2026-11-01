# Architecture decisions

These decisions describe the approved direction as of 2026-10-28. They guide future work; they do not imply that planned integrations exist. Revisit a decision on its feature branch if implementation evidence warrants a change, and record the reason.

## Current baseline (merged Phases 1–6 and dependent Phase 7B–7D feature branches)

The FastAPI backend exposes `/health` and, with explicit development configuration, `/documents/parse`, `/documents/chunk`, `/collections/{collection_id}/documents/index`, `/collections/{collection_id}/search`, and `/collections/{collection_id}/answer`. PDF, TXT, and Markdown extraction, LangChain text splitting, local Ollama embeddings and generation, PostgreSQL/pgvector persistence, exact semantic search, and single-turn cited answers are merged. The unmerged OpenAI and Gemini parent branches and this dependent Anthropic branch add explicitly gated cloud generation; embeddings stay local. Development routes have no authentication or collection authorization and are restricted to trusted localhost use. There is no vision model, agent, MCP tool, or frontend.

## AD-01 — Local-first and explicit processing location

**Decision:** Develop and verify the first embedding and answer pipelines with local services. Support local Ollama and later explicitly authorized cloud AI providers through a policy based on the collection's data classification and caller authorization. Confidential documents must never be sent to cloud LLM, embedding, or vision providers through automatic fallback.

**Reason:** Processing location affects confidentiality and data-handling obligations. A local-first path keeps early integration tests independent of external credentials and services.

**Implementation status:** Phases 4 and 6 implement loopback-only Ollama embedding and generation with no cloud fallback. Dependent Phase 7B–7D branches add OpenAI, Gemini, and Anthropic behind one development-only operator-controlled collection allowlist; general data-classification policy and production caller authorization are not implemented.

## AD-02 — Provider-independent capabilities

**Decision:** Define separate, small interfaces for embedding, LLM generation, and vision-language model (VLM) understanding when each capability is introduced. Keep local Ollama and later cloud-provider clients and configuration behind those boundaries; do not add speculative abstractions or SDKs before use. The Ollama endpoint must be configurable by the application. Provider selection and failures must be explicit and observable without logging document content.

**Reason:** Provider changes should not alter document ownership, provenance, or authorization behavior.

**Implementation status:** Small embedding and generation provider interfaces and local Ollama implementations exist. The dependent Phase 7B–7D branches add OpenAI, Gemini, and Anthropic generation implementations; cloud embeddings and the VLM interface remain planned.

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

## AD-10 — Explicit collection-approved cloud generation

**Decision:** Keep local Ollama as the default embedding and generation provider. Permit `gpt-6-luna` generation only when a trusted development operator selects OpenAI, enables cloud transfer, allowlists the collection UUID, and configures a secret credential. Check the gate before retrieval or provider construction. Use the fixed HTTPS Responses endpoint with `store: false`, no tools, a strict root-object wire schema, and bounded output. Convert validated wire output into the existing `GeneratedAnswer`; keep `RAGService` and citation resolution provider-independent. Do not automatically retry across providers.

**Reason:** A configured credential does not authorize sending documents. An explicit collection gate limits accidental transfer while the fixed endpoint and validated envelope prevent redirect, tool, and malformed-output ambiguity. The existing retrieval and citation pipeline remains one auditable implementation.

**Implementation status:** The unmerged parent branch implements OpenAI text generation with mocked provider tests. One later authorized synthetic live request received HTTP 429 without a completed answer. The question and bounded retrieved text, filenames, and page numbers may be transmitted for an allowlisted request. `store: false` does not by itself remove provider abuse-monitoring retention. A collection UUID allowlist is a single-developer data-transfer control, not caller identity or production authorization. Live OpenAI generation quality and retention configuration remain unverified.

## AD-11 — Gemini uses the same cloud transfer gate and answer contract

**Decision:** Add only `gemini-3.5-flash-lite` on Google's fixed HTTPS `generateContent` endpoint. Use `systemInstruction`, one user content item, one candidate, and Google's JSON `responseFormat`, without tools or fallback. Map the existing output limit to `maxOutputTokens`; leave Ollama context/temperature and Gemini thinking controls unset. Accept only a successful model-role text candidate with `STOP` and a model version equal to the configured alias or that alias plus a numeric version suffix. Validate the resulting JSON with `GeneratedAnswer`, then let `RAGService` resolve citation labels. Reuse the development-only cloud flag, server-owned collection UUID allowlist, and secret-key gate before retrieval.

**Reason:** Provider-specific request and response fields belong at the provider boundary. Reusing the gate and domain contract keeps cloud transfer policy, abstention, and citations consistent without duplicating retrieval or orchestration. The narrow version rule avoids assuming a stable alias must be returned verbatim while rejecting unrelated model families.

**Implementation status:** The unmerged Gemini parent branch implements text generation with mocked HTTP tests only. Actual API acceptance, model-version strings, quality, cost, and regional/data-handling behavior remain unverified. Caller authentication and document authorization are still absent, so public/shared deployment remains blocked.

## AD-12 — Anthropic Messages output stays behind the shared answer boundary

**Decision:** Add only `claude-haiku-5-5` through the fixed HTTPS Messages API, using top-level `system`, one user message, `max_tokens`, and `output_config.format` JSON schema. Omit sampling settings, effort, tools, and native citations. Allow thinking blocks as non-answer metadata but require exactly one text block in an assistant `end_turn` message from the configured model. Validate that text with `GeneratedAnswer` and retain application-owned citation resolution in `RAGService`. Reuse the existing development-only cloud flag, collection allowlist, and secret-key gate before retrieval.

**Reason:** Anthropic's response envelope and adaptive thinking differ from OpenAI and Gemini, while the RAG answer and provenance contract does not. Native Anthropic citations are incompatible with JSON structured output and would duplicate the application's request-local source-label mapping. A bounded output budget can end in `max_tokens`, which must fail closed rather than return partial JSON.

**Implementation status:** This dependent branch adds mocked Anthropic Messages support only. The latest documented stable API version is sent, but actual account/model acceptance, response shape, cost, data residency, and quality remain unverified. Public deployment still needs caller and document authorization.
