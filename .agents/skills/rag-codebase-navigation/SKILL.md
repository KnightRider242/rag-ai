---
name: rag-codebase-navigation
description: Navigate RAG-AI architecture and data flow. Use for unfamiliar subsystems, architecture investigations, or changes spanning components.
---

# Codebase navigation

1. Start with `docs/CODEBASE_MAP.md` and the relevant architecture decision; verify each useful claim against current code.
2. Locate entry points, models, interfaces, and tests with targeted path and symbol searches. Read only the functions needed to trace the task.
3. Trace upstream callers and downstream effects through configuration, FastAPI, document processing, provider, and PostgreSQL boundaries as applicable.
4. Explain the relationship in plain language, citing relevant files. Separate implemented behavior from roadmap plans and note any documentation drift.
