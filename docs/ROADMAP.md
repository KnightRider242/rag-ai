# RAG-AI development roadmap

This roadmap records the approved development direction. Phases 1–5 are merged; Phase 6 is implemented locally on `feat/local-rag-answers` and awaits human review. Later phases are plans, not current capabilities. Each phase requires its own scoped branch, tests, security review, and documentation before merge. See [architecture decisions](ARCHITECTURE_DECISIONS.md) and the [security test plan](SECURITY_TEST_PLAN.md).

| Phase | Status | Intended outcome |
| --- | --- | --- |
| 1. Backend foundation | Complete | FastAPI health API, typed configuration, tests, and local PostgreSQL with pgvector configuration. The API does not yet use the database. |
| 2. Document ingestion | Complete | Development-only PDF, TXT, and Markdown extraction with metadata and input limits; no persistence. |
| 3. Document chunking | Complete | Development-only LangChain text splitting with source-page metadata, deterministic IDs, and resource bounds; no persistence. |
| 4. Local embeddings and pgvector | Complete for local development | Use native local Ollama at a configurable loopback endpoint and store collection-scoped original documents, chunks, vectors, model names, digests, and measured dimensions in PostgreSQL/pgvector via Docker Compose. Explicit versioned migrations and storage-layer document deletion exist; indexing is development-only and has no collection authorization. |
| 5. Semantic retrieval and evaluation | Complete for local development | Development-only collection-scoped exact cosine search, model compatibility checks, a LangChain adapter, and a separate synthetic evaluation dataset. Caller/document authorization remains unimplemented. |
| 6. Local text RAG with citations | Implemented locally; human review pending | Generate text answers locally from retrieved evidence and cite traceable source locations; handle insufficient evidence explicitly. |
| 7. Multi-provider LLM and embedding support | Planned | Add independent embedding and LLM interfaces for local Ollama and explicitly authorized cloud AI providers, with explicit local/cloud processing policies. Confidential data must never fall back to a cloud provider automatically. |
| 8. Multimodal document understanding | Planned | Add an independent vision-language model (VLM) interface; extract and represent multimodal PDFs, images, figures, diagrams, and tables while preserving links to original source files and page or region locations. |
| 9. Multimodal RAG and visual citations | Planned | Retrieve text and visual evidence and return citations that identify the original visual source. |
| 10. LangGraph agentic workflows | Planned | Add bounded, observable workflows only where they improve a defined user task; enforce tool and data permissions. |
| 11. MCP tool integration | Planned | Expose or consume approved tools through MCP with explicit scopes, validation, and audit controls. |
| 12. React frontend | Planned | Build a user interface for authorized document and answer workflows, including citation inspection. |
| 13. Deployment and security hardening | Planned | Complete deployment controls, access enforcement, resource isolation, observability safeguards, and operational validation. Public document processing remains blocked until its security requirements are met. |

## Cross-phase acceptance principles

- Develop locally first and keep provider selection explicit. Design embedding, LLM, and vision boundaries so a later provider can be added without changing document ownership rules.
- Target a cross-platform Python backend on Windows, macOS, and Linux. Ollama runs natively for local inference, with CPU or supported GPU use according to host hardware. Keep PostgreSQL/pgvector in Docker Compose and document platform-specific setup. RAG-AI selects a configurable loopback Ollama endpoint and does not silently switch to a cloud endpoint.
- Keep stored documents scoped to a collection; add collection authorization before any shared or public access and enforce it in storage and retrieval. Record embedding model name, Ollama digest, and measured vector dimensions so incompatible vectors are not mixed silently.
- Preserve original source bytes and traceable locations for future citations. Treat extracted text, images, diagrams, and tables as untrusted data.
- Use separate development and evaluation datasets. Automated tests use synthetic data; evaluation data must have documented rights and handling rules before use.
- Add functional and security tests for each new attack surface. Do not treat a completed phase or passing tests as a production security approval.
- Add automated Windows, macOS, and Linux test coverage as integrations arrive. Keep development fixtures separate from evaluation datasets and record platform-specific limitations rather than assuming parity.

Phase 6 adds single-turn local answers from retrieved text with application-resolved citations and explicit abstention. Multimodal processing, agents, MCP integration, and a frontend remain planned. Public document access remains blocked by missing authentication, authorization, ingress limits, and stronger resource isolation.


## Application flow

```mermaid
flowchart TB
    subgraph P4["Document indexing — Phase 4"]
        direction LR
        Upload["Upload PDF, TXT, or MD"] --> IndexRoute["FastAPI index endpoint<br/>api/documents.py"]
        IndexRoute --> Parse["Validate and parse<br/>documents/parser.py"]
        Parse --> Chunk["LangChain text chunking<br/>documents/chunker.py"]
        Chunk --> Indexer["Index coordination<br/>documents/indexer.py"]
        Indexer --> DocEmbed["Local Ollama chunk embeddings<br/>embeddings/ollama.py"]
        DocEmbed --> Save["PostgresIndexStore.save()"]
    end

    DB["Shared PostgreSQL + pgvector<br/>storage/postgres.py"]

    subgraph P5["Semantic retrieval — Phase 5"]
        direction LR
        Question["Search question"] --> SearchRoute["FastAPI search endpoint<br/>api/retrieval.py"]
        SearchRoute --> Service["RetrievalService<br/>retrieval/service.py"]
        Service --> QueryEmbed["Query vector via same Ollama provider<br/>embeddings/ollama.py"]
        QueryEmbed --> Search["PostgresIndexStore.search()"]
        Search --> Compatible["Collection + model name,<br/>digest, dimension checks"]
        Compatible --> Rank["Cosine distance + top K"]
        Rank --> Chunks["Chunk text + source metadata"]
        Chunks --> JSON["JSON search response"]
    end

    Save --> DB
    DB --> Search
    Adapter["LangChain BaseRetriever adapter<br/>retrieval/langchain.py"] --> Service

    subgraph P6["Cited answers — Phase 6"]
        direction LR
        AnswerQuestion["Answer question"] --> AnswerRoute["Development answer route<br/>api/answers.py"]
        AnswerRoute --> RAG["RAGService<br/>rag/service.py"]
        Evidence["Retrieved chunks"] --> Context["Bounded context<br/>rag/context.py"]
        Context --> Gen["Local Ollama chat<br/>generation/ollama.py"]
        Gen --> Validate["Structured output + citation validation"]
        Validate --> Answer["Cited answer or abstention"]
    end

    RAG --> Service
    Chunks --> Evidence
    RAG --> Context
```

All document routes are for trusted localhost development. Collection filtering and model compatibility protect search correctness; they do not authenticate callers. The LangChain adapter delegates to the same retrieval service. RAGService also reuses that service, then sends bounded evidence to local Ollama and resolves cited labels against retrieved chunks; citation validity alone does not prove factual support.
