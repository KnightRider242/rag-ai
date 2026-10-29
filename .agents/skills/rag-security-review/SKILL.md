---
name: rag-security-review
description: Review RAG-AI attack surfaces. Use for security-sensitive changes or a relevant pre-merge security review.
---

# RAG-AI security review

1. Identify changed trust boundaries and protected data; read only related code, negative tests, and `docs/SECURITY_TEST_PLAN.md`.
2. Check applicable input validation, SQL or command injection, XSS, authentication and authorization, collection isolation, prompt injection, sensitive data exposure, secrets and logs, dependency vulnerabilities, provider network access, unauthorized cloud transfer, and resource exhaustion.
3. Verify relevant negative tests and that localhost-only development gating is never described as production authentication.
4. Report protections, tests actually run, and unresolved risks with severity. Apply the high/critical merge-blocker rule in `AGENTS.md`; passing tests alone is no security assurance.
