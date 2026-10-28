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
