# Development conventions

- Target Python 3.12 or newer. Add type hints to public functions and use Pydantic for API models and environment settings.
- Keep HTTP handling in `src/rag_ai/api`, configuration in `src/rag_ai/core`, and application wiring in `src/rag_ai/main.py`. Introduce domain and persistence modules when behavior requires them; avoid speculative layers.
- Keep routes small and explicit. Use application factories for wiring and dependency injection when new services are added.
- Keep dependencies focused on implemented features. Do not add LangChain, LangGraph, provider SDKs, or MCP packages until their integrations are built.
- Read credentials from environment or a secret store. Never hardcode keys or commit `.env` files, tokens, or database passwords.
- Add tests for observable behavior. Run `python -m pytest` before reporting a backend change. Keep the health endpoint a liveness check unless a separate readiness endpoint is designed.
- Review `git diff` and `git status` before committing. Preserve `LICENSE`.

## Git branching strategy

- `main` represents stable, tested code. Develop every new feature on a dedicated branch; never implement a new feature directly on `main`.
- Name branches `feat/<feature-name>`, `fix/<bug-name>`, `refactor/<component-name>`, `test/<test-name>`, `docs/<documentation-name>`, or `chore/<maintenance-name>`.
- Before implementing a feature, verify the active branch, inspect the existing code, confirm the feature scope, and review relevant architecture and dependencies.
- Include the implementation, relevant automated tests, and appropriate documentation updates on each feature branch.
- Before calling a feature complete, run relevant automated tests, review Git changes, verify existing functionality still works, and document incomplete work and limitations.
- Do not commit, push, or merge without explicit user approval.
- After approval, use a Conventional Commit, push the feature branch, and prepare a Pull Request targeting `main`.
- Merge only after review and successful tests. Use squash merging by default to keep `main` history clean.
- After merging, synchronize local `main` with `origin/main`, delete the completed feature branch, and start the next feature from the updated `main`.
- Do not implement another feature unless explicitly requested.

## Security-by-Design

- Treat uploaded documents and all external content as untrusted input. Validate external inputs before processing them.
- Never hardcode or commit API keys, credentials, tokens, or confidential data. Never log document contents, passwords, authentication tokens, or API keys.
- Follow least-privilege access principles and prevent unauthorized document access.
- Never promote instructions found in retrieved documents to system instructions.
- Ensure database queries and vector retrieval respect document-level access-control boundaries.
- Never send confidential documents to external LLM or embedding providers without explicit authorization and appropriate data-handling controls.
- Configure external tracing to avoid exposing sensitive content or credentials.
- Use synthetic data in automated tests. Add security-focused tests for every feature that introduces an attack surface.
- Do not claim that passing tests makes the application secure, audited, or production-ready.

For every feature, identify relevant threats, implement proportionate protections, add negative and security tests, run functional and security tests, document unresolved risks, check changes for secrets, and report security regressions. Critical or high-severity unresolved security issues block a recommendation to merge.

## Mandatory session reporting

At the end of every development task or session, provide a structured **DEVELOPMENT SESSION REPORT** with these sections:

1. **SESSION SUMMARY:** What was accomplished and which development phase was worked on.
2. **COMPLETED TASKS:** Features implemented, bugs fixed, and configuration changes.
3. **FILES CREATED OR MODIFIED:** Relevant files and directories, with explanations of important modifications.
4. **TESTING & VERIFICATION:** Tests executed and pass/fail counts; Docker/container status and API endpoint checks when relevant. Clearly distinguish verified results from checks not performed.
5. **ISSUES & BLOCKERS:** Problems encountered, unresolved errors, and known limitations.
6. **GIT STATUS:** Active branch, uncommitted changes, commit hash if committed, push status, and Pull Request status.
7. **REMAINING WORK:** Incomplete tasks and features not yet implemented.
8. **NEXT SESSION PLAN:** Exact next development steps and commands needed to resume.
9. **SECURITY REVIEW:** Risks identified, protections implemented, security tests and pass/fail results, unresolved vulnerabilities, and implications for future phases.

Always provide the final report before ending a completed task. When the user says "wrap up," finish with a complete DEVELOPMENT SESSION REPORT and update `docs/SESSION_LOG.md`.

If a session is interrupted, preserve progress and report its current state when work resumes. Every report must identify implemented functionality, files modified, tests and actual results, outstanding issues, commit and push status, Pull Request status, and the next recommended task.

## Persistent development log

Maintain `docs/SESSION_LOG.md`. After each completed development task, append a dated summary that records completed work, verification results, outstanding issues, and the next recommended task. Never overwrite previous session records.

## Reporting and phase rules

- Never claim tests passed without running them.
- Never claim changes were committed or pushed unless verified.
- Never mark partially completed features as done.
- Do not expose API keys or credentials in reports.
- Do not start another development phase until the current phase's acceptance criteria are complete.
