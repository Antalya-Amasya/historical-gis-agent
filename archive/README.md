# Preserved historical audit artifacts

These files are non-production audit records retained during the 2026-10-03 worktree consolidation.

- `codex_outputs`: 15 historical JSON acceptance outputs plus one ignored Pleiades index summary.
- `deepseek_harness`: 11 audit scripts, `_rag_ids.json`, and an existing pytest failure transcript.
- `consolidation_20261003`: preservation manifest and offline acceptance results.

Archived scripts retain their original text, including historical absolute paths. They were not executed, refactored, or added to runtime imports. Production tests do not depend on this archive.

The preservation manifest records source/destination paths, byte counts, SHA-256 hashes, excluded caches, and exact duplicate source datasets. Old source directories were retired only after complete history containment, successful offline validation, and byte-for-byte archive verification.

The frozen replay compares complete Evidence IDs, text, scores, and metadata with existing accepted baselines; it does not relabel data or retune retrieval. The combined smoke uses the production retriever and authority pipeline with an injected read-only fixture store, scripted provider, and existing canonical-place fixture. It is not a live-provider or live-Chroma acceptance run.

Focused regression results: Sol-A 726 passed; Sol-B 267 passed; response/presentation 31 passed. First Sol-B run encountered 27 environment setup errors because the parent temporary directory was missing; after creating it, all 267 passed. No production repair was made.

Production remains `roman_republic_primary_sources_v2`; v3 remains shadow; entity metadata defaults to false. No provider calls, embeddings, Chroma rebuilds, or push occurred.
