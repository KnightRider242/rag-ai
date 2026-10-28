# RAG-AI development roadmap

This roadmap records the approved development direction. Phases 1–3 are implemented; later phases are plans, not current capabilities. Each phase requires its own scoped branch, tests, security review, and documentation before merge. See [architecture decisions](ARCHITECTURE_DECISIONS.md) and the [security test plan](SECURITY_TEST_PLAN.md).

| Phase | Status | Intended outcome |
| --- | --- | --- |
| 1. Backend foundation | Complete | FastAPI health API, typed configuration, tests, and local PostgreSQL with pgvector configuration. The API does not yet use the database. |
| 2. Document ingestion | Complete | Development-only PDF, TXT, and Markdown extraction with metadata and input limits; no persistence. |
| 3. Document chunking | Complete | Development-only LangChain text splitting with source-page metadata, deterministic IDs, and resource bounds; no persistence. |
| 4. Local embeddings and pgvector | Planned | Generate embeddings with native local Ollama at a configurable endpoint and store collection-scoped documents, chunks, vectors, and model metadata in PostgreSQL/pgvector via Docker Compose. Define migrations, access boundaries, and re-embedding behavior before implementation. |
| 5. Semantic retrieval and evaluation | Planned | Retrieve authorized chunks and measure retrieval quality using a separate evaluation dataset. |
| 6. Local text RAG with citations | Planned | Generate text answers locally from retrieved evidence and cite traceable source locations; handle insufficient evidence explicitly. |
| 7. Multi-provider LLM and embedding support | Planned | Add independent embedding and LLM interfaces for local Ollama and explicitly authorized cloud AI providers, with explicit local/cloud processing policies. Confidential data must never fall back to a cloud provider automatically. |
| 8. Multimodal document understanding | Planned | Add an independent vision-language model (VLM) interface; extract and represent multimodal PDFs, images, figures, diagrams, and tables while preserving links to original source files and page or region locations. |
| 9. Multimodal RAG and visual citations | Planned | Retrieve text and visual evidence and return citations that identify the original visual source. |
| 10. LangGraph agentic workflows | Planned | Add bounded, observable workflows only where they improve a defined user task; enforce tool and data permissions. |
| 11. MCP tool integration | Planned | Expose or consume approved tools through MCP with explicit scopes, validation, and audit controls. |
| 12. React frontend | Planned | Build a user interface for authorized document and answer workflows, including citation inspection. |
| 13. Deployment and security hardening | Planned | Complete deployment controls, access enforcement, resource isolation, observability safeguards, and operational validation. Public document processing remains blocked until its security requirements are met. |

## Cross-phase acceptance principles

- Develop locally first and keep provider selection explicit. Design embedding, LLM, and vision boundaries so a later provider can be added without changing document ownership rules.
- Target a cross-platform Python backend on Windows, macOS, and Linux. Plan native Ollama installations on each platform, with CPU inference and GPU acceleration only where the host hardware and Ollama support it. Keep PostgreSQL/pgvector in Docker Compose and document platform-specific setup. A future RAG-AI setting must select the Ollama endpoint; do not assume a fixed host or silently switch to a cloud endpoint.
- Keep stored documents scoped to a collection and enforce its authorization boundary in storage and retrieval. Record embedding model identity, version where available, and vector dimensionality so incompatible vectors are not mixed silently.
- Preserve original source files and traceable locations for citations when persistence is introduced. Treat extracted text, images, diagrams, and tables as untrusted data.
- Use separate development and evaluation datasets. Automated tests use synthetic data; evaluation data must have documented rights and handling rules before use.
- Add functional and security tests for each new attack surface. Do not treat a completed phase or passing tests as a production security approval.
- Add automated Windows, macOS, and Linux test coverage as integrations arrive. Keep development fixtures separate from evaluation datasets and record platform-specific limitations rather than assuming parity.

The next implementation milestone is Phase 4, only when separately requested. No Ollama, embedding, vector storage, retrieval, LLM, multimodal, agent, MCP, or frontend integration is implemented by this roadmap.
