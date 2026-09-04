# AGENTS.md

## What this repo is

- Local-first, source-grounded Sunni Islamic study/content assistant. Python core, FastAPI server, minimal web client; SQLite/FTS5 knowledge store; OpenAI-compatible model backends.
- `agentodo.md` is the canonical spec. Read it (especially §26 phases and §28 first task) before writing code.

## Commands

```bash
uv sync            # install deps (uv-managed venv; no system pip)
uv run pytest -q   # tests (live model tests auto-skip when routed model is not loaded)
uv run ruff check agent ingestion retrieval evaluation apps tests scripts   # lint (line length 120)
uv run uvicorn apps.api.server:app --host 127.0.0.1 --port 8017     # API + web client
uv run python -m scripts.run_regression --models ling_tiny   # grounded regression (§19)
scripts/dev.sh bench --models ling_tiny   # capability benchmark harness (needs LM Studio up)
```

Ingest Qur'an corpus first: `uv run python -c "from ingestion.quran_ingest import QuranIngestor; QuranIngestor().ingest()"` (server refuses to start without it).

## Environment

- LM Studio runs at `http://127.0.0.1:1234/v1` (OpenAI-compatible). Loaded model IDs are in `/api/v0/models`; unload via `POST /api/v1/models/unload` with `{"instance_id": ...}`. Only ~12GB VRAM: never run two large models at once.
- LM Studio accepts but IGNORES Ling's `enable_thinking` chat_template_kwarg — Ling always reasons; budget max_tokens ≥4096 for it (see evaluation/model-benchmark.md).
- Arabic FTS: `unicode61 remove_diacritics 2` does not strip Uthmani Quranic marks, so the index stores a normalized `search` column from `ingestion/arabic_norm.py`; queries go through `search_form()` too. Never FTS against `quran.arabic`.

## Non-negotiable constraints (from agentodo.md §0, §5, §29)

- Knowledge corpus is restricted to the approved **Sunni source registry**: never ingest, retrieve, cite, or summarize non-approved material. Never ingest first and filter later — registry/license/quality gates run before ingestion (§5.2); enforcement lives in `agent/policy/source_policy.py` (`assert_ingestible`, `retrieval_filter`) and every new data path must call them.
- Never invent Qur'an references, hadith, scholars, book titles, gradings, or quotations. Qur'an text served to users comes only from the hash-verified `quran-uthmani-json` dataset; model output is never Qur'an. If unverifiable, return the exact §12 notice (`agent/validators/pipeline.py::UNVERIFIABLE_NOTICE`).
- The LLM is not the source of Islam: model pretraining is never evidence; answers come from retrieval + validators over the approved corpus.
- Never expose chain-of-thought. Store only provenance, decisions, confidence, validation results.
- Local-first: no mandatory cloud API. All model backends must be replaceable behind an OpenAI-compatible `/v1` interface; model routing is runtime config (`configs/config.yaml`), never hard-coded (§3).
- RAG is the knowledge mechanism. Fine-tuning is for behavior only, and only after the evaluation suite exists (Phase 5 → Phase 6, §16–17).
- YouTube lectures are secondary material, never automatically authoritative (§22). Never auto-publish religious content without a human review gate (§21).

## Working process

- Follow the phase order in §26. Phases 0–5 are done (see README status table + evaluation/grounded-regression.md baseline); corpora ingested: Qur'an (hash-verified Uthmani), Kutub al-Sittah hadiths, Kemenag tafsir + Indonesian translation, and three classic English tafsirs (quran_campaign chunks). Fine-tuning (Phase 6) and content engine (Phase 7) remain. Fine-tuning must demonstrably beat the Phase 5 baseline before release (§16–17).
- The regression suite is the gate for model/routing changes: run `scripts.run_regression` after any routing/prompt/validator change; hallucination rate must not regress.
- The loaded `gemma-4-e4b-uncensored-hauhaucs-aggressive` is a third-party variant — disqualified for religious answers by the Phase 5 regression (invented grading, 0% refusal). Religious task classes must route to ling_tiny or gemma_qat.
- Classic English tafsirs (`tafsir-*-en`) are owner-approved for PRIVATE LOCAL STUDY ONLY — print-edition copyright unresolved; never publish or redistribute their text (§21). Their chunks came from quran_campaign's kb.db (hash-pinned in registry); 164 mis-tagged chunks are in `tafsir_en_quarantine`, excluded from retrieval until human-fixed.
- Adding a data source: add to `knowledge/registry/approved_sources.yaml` (with license + verification), then an ingestor module calling `assert_ingestable` before any write; dataset file hash goes in registry notes (pattern: `quran-uthmani-json`).

## Models (§2, §4)

- Ling-3.0-tiny = primary agent/reasoning model; Gemma 4 QAT = fast/mobile/multimodal path. Both must be swappable without agent changes.
- Target hardware is Profile A: RTX 3060 12GB + 64GB RAM. Start at 4K–8K context (measure KV-cache cost before increasing); keep 8–12GB RAM free for host; activated parameter count ≠ VRAM requirement.

## Acceptance

- §27 "Definition of Done" is the project-level acceptance checklist (provenance on every claim, deterministic Qur'an refs, hadith attribution metadata, uncertainty admitted, no fabricated citations).

