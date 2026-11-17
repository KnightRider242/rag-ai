# Real-PDF evaluation baseline

## Current status

Two approved PWC-Net PDFs and a private 16-case manifest are available outside Git. Their hashes match the manifest. A disposable-database discovery run indexed both PDFs and identified five inspected visual elements. A pre-fix 16-question baseline ran; its context failures and retrieval metrics are recorded below. Human factual-accuracy scoring remains pending. The example PDFs are generated in memory by `tests/test_real_pdf_baseline.py`; binary PDFs, private source content, and credentials are not committed.

PDF ingestion replaces each PyMuPDF-extracted NUL (`U+0000`) with a visible replacement character (`U+FFFD`) before page metadata, chunking, embeddings, or PostgreSQL writes. Each page reports `replaced_nul_count`. The original bytes and SHA-256 document identity are preserved; other valid Unicode is unchanged, and invalid surrogate code points are rejected. UTF-8 TXT and Markdown uploads containing NUL are rejected with HTTP 422. Chunk IDs remain deterministic for exact bytes and settings, but chunks whose formerly invalid text contained NUL have new IDs after normalization.

Visual inspection keeps the 200-cell cap. Oversized candidates from PyMuPDF's text-table fallback are skipped before cell extraction; each page reports `skipped_oversized_text_tables`, so a nonzero count means table detection was partial. Drawn-grid tables still fail at the cap, and image/vector occurrence extraction and its existing limits continue. In the exact-byte discovery, the main PDF reported 10 skipped fallback candidates and the supplement 2; all persisted table candidates were within the cap. The `/visual-elements/index` response reports stored element count, while `/documents/inspect` provides the per-page partial-detection counts.

The example manifest in [`tests/evaluation/real_pdf_manifest.example.json`](../tests/evaluation/real_pdf_manifest.example.json) has four text, four visual, four mixed, and four deliberately unanswerable questions across two authored synthetic manuals. Expected answers and page pointers were written from the fixture design, independently of model output. Its `labels_verified_from_source` fields remain `false` until a human checks the rendered PDFs. Visual `element_id` fields are omitted until the extracted source regions have been inspected. Those cases are **unscored** for metrics requiring unavailable labels.

## Run an approved real baseline

1. Choose two or three non-sensitive PDFs and obtain explicit permission for local model processing. Keep their files and the completed manifest outside this repository and review TAR. Record each PDF's SHA-256 and set `approved_for_local_evaluation: true` only for approved files. Use extractable text; the parser has no OCR.
2. Inspect the original PDFs yourself. Write approximately four questions in each category, the expected answer or abstention, relevant document/page, and chunk index where determinable. Set `labels_verified_from_source: true` only after independent source inspection. If a chunk or element cannot be identified, leave that label pending; never infer ground truth from Qwen3 or Qwen3-VL output.
3. Use the existing development routes on a separately verified disposable test schema to index the PDFs, call `/visual-elements/index`, and inspect the paginated `/visual-elements` records against the original PDF. Copy only relevant `element_id` values into `selected_visual_elements`, with a predefined task. The evaluation test analyzes only those selected elements. The source hash and deterministic element IDs must still match when the final run reindexes the same bytes.
4. Privately set `RAG_AI_TEST_DATABASE_URL` to a disposable loopback `rag_ai_retrieval_test` database with pgvector, set `RAG_AI_REAL_PDF_MANIFEST` to the external manifest path, and set `RAG_AI_RUN_REAL_PDF_EVAL=1`. Run `python -m pytest -q -s tests/test_real_pdf_baseline.py -k live_real_pdf_baseline --tb=line`. The fixture reuses the local E2E destination preflight before any database connection, creates a random schema, applies existing migrations only there, and cleans up that schema. It rejects source files or the manifest inside this repository. It uses local EmbeddingGemma, Qwen3-VL, and Qwen3; there is no cloud fallback.

The runner calls the current parse, chunk, text index, visual extraction, selected interpretation, visual embedding index, text search, visual search, combined search, and multimodal answer routes. It does not change ranking, prompts, citation contracts, or production code. Before running, confirm model availability and a bounded runtime budget; each selected visual element invokes the local vision model once.

## Results and measurement rules

The reusable runner is [`tests/evaluation/real_pdf_baseline.py`](../tests/evaluation/real_pdf_baseline.py). Its sanitised output model has a checked [JSON Schema](../tests/evaluation/real_pdf_results.schema.json). Results contain case IDs, source IDs, page/coordinate metadata for candidate inspection, scores, counts, timings, and fixed stage codes. They omit PDF bytes, extracted text, question text, generated claim text, credentials, and raw model responses.

| Measure | Scored denominator | Meaning |
| --- | --- | --- |
| Text Hit@K | Verified cases with resolved chunk labels | At least one relevant chunk retrieved. |
| Text Recall@K | Verified relevant chunks | Number retrieved divided by labeled relevant chunks; macro and micro forms are reported. |
| Text MRR@K | Verified cases with resolved chunk labels | Reciprocal rank of the first relevant chunk. |
| Visual Hit@K | Verified cases with exact, resolved element IDs | At least one labeled element retrieved. |
| Combined coverage | Verified cases whose required chunk and element IDs all resolve | Every labeled evidence item appears in the fused top K. |
| Citation identity | Returned citations | IDs, page, coordinates, and stored interpretation provenance match indexed records. This does not establish factual support. |
| Abstention correctness | Verified unanswerable cases | Returned `insufficient_evidence` with no claims, as checked by the API contract. |
| Human answer support and visual accuracy | Independently reviewed cases | Null until a reviewer checks the original PDF and model output; `apply_human_reviews()` records those judgments afterward. |
| Latency | Attempted calls | Separate text, visual, and combined search timings; answer endpoint timing includes retrieval and generation. A test-only timer records Qwen3 generation duration when the live opt-in runs. Selected Qwen3-VL interpretation calls have their own timing and denominator. |

The runner accepts an optional in-memory `answer_observer(case_id, response)` callback for an authorized human review workflow. Keep any captured claim wording in a private location outside Git and the review TAR; the default sanitized report discards it. `apply_human_reviews()` adds post-run judgments by case ID and recomputes only their aggregate metrics.

Each case includes a failure stage: parsing, indexing, extraction, interpretation, retrieval, context, generation, or citation verification. API errors can combine causes: `generation_invalid` is reported as generation because the public code cannot always distinguish malformed output from invalid labels. The source-citation check reports mismatched identities as citation verification failures. All aggregates include numerators and denominators; null means unscored, never zero accuracy.

## Phase 9E context-budget follow-up

The approved 16-question PWC-Net run before the context fix had 16/16 `evidence_too_large` failures at the default 4,000-character budget. At 8,000 characters and 2,000 per evidence, 11/16 still failed on context and five reached generation. Text Hit@5 was 87.5%, visual Hit@5 was 100%, and combined coverage@5 was 50%. The five completed answers yielded five structurally valid citation identities; factual accuracy has not been human-scored. These are pre-fix observations, not results of the new packing behavior.

The multimodal context now excludes oversized or non-fitting complete records while retaining ranking order and a request-local map to full citation provenance. Each packed section includes a bounded, quoted filename and page number for source attribution; those bytes count against the total budget. It fails closed if no safe evidence remains. Budget exclusions log counts without document content or IDs. Use `--tb=line` for private runs to avoid traceback content from PDF fixtures. The configured character budget remains 4,000 by default.

One approved post-fix run on the same 16-question external manifest and a verified disposable `rag_ai_retrieval_test` database completed at the default 4K budget before the filename/page attribution adjustment: **0/16 context failures**, **12/12 structurally valid citation identities**, text Hit@5 **87.5%**, visual Hit@5 **100%**, and combined coverage@5 **50%**. All 16 cases reached generation, answer status matched the labels in **11/16**, and the four unanswerable cases abstained correctly.

The final guarded run on 2026-11-17 used the adjusted prompt, the same exact-hash PDFs and 16-case manifest, and a new loopback disposable pgvector container with tmpfs-only database storage. At the default 4K total and 1,200-character per-evidence budgets, it had **0/16 context failures**, **13/16 answer statuses correct**, **4/4 abstentions correct**, and **14/14 structurally valid citation identities**. Text Hit@5 was **87.5%**, visual Hit@5 **100%**, and combined coverage@5 **50%**: unchanged from both earlier runs. No failure stage occurred. Mean generation time was **5.46 seconds across 16 calls**; mean visual-interpretation time was **12.65 seconds across five selected elements**. The container was removed after the run. Human answer support and visual-interpretation accuracy remain unscored; these figures do not guarantee results for other PDFs or model revisions.

## Observations and limits

The synthetic focused tests validate manifest rules, PDF text/figure extraction, route orchestration with mocked responses, denominator handling, selected-element bounds, source-coordinate mismatch detection, and stage-error classification. The guarded real-PDF run measured retrieval coverage and local latency, but did not independently score visual interpretation accuracy or answer support. The existing development routes lack caller authorization; use only a trusted local setup and a disposable database. Structural citation validity cannot prove semantic factual correctness.
