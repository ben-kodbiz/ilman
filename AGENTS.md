# AGENTS.md

## What this repo is

- Ilman (Huurs): local-first, source-grounded Sunni Islamic study assistant + context-aware Islamic companion. Python core, FastAPI server, PWA client, Gradio dev console; SQLite/FTS5 knowledge store; OpenAI-compatible model backends.
- Specs: `agentodo.md` is canonical (read §26 phases, §28 first task). The companion/validation evolution specs are `fix_me.md` (companion v1), `fixme_v2.md` (harness), `fixme_v3.md` (evidence judge), `fixme_v3.1.md` (validation hardening). All are implemented; docs describe the current state.

## Commands

```bash
uv sync            # install deps (uv-managed venv; no system pip)
uv run pytest -q   # 445 tests (live model tests auto-skip when routed model is not loaded)
uv run ruff check agent ingestion retrieval evaluation apps tests scripts   # lint (line length 120)
uv run uvicorn apps.api.server:app --host 127.0.0.1 --port 8017     # API + PWA client
uv run python -m apps.devui.console                                # Gradio dev console :7860

# gates — run the relevant one after ANY routing/prompt/validator/companion change
uv run python -m scripts.run_regression --models ling_tiny       # grounded regression; hallucination must not regress
uv run python -m scripts.run_companion_eval --models ling_tiny  # companion matrix; crisis routing must stay 100%
uv run python -m scripts.run_companion_score --models ling_tiny # fixme_v2 §28 weighted score
uv run python -m evaluation.bench.v31_metrics                   # validator quality; fails if false-support >5%
```

Ingest Qur'an corpus first: `uv run python -c "from ingestion.quran_ingest import QuranIngestor; QuranIngestor().ingest()"` (server refuses to start without it). Vector index build: `uv run python -c "from retrieval.vector_store import VectorStore; VectorStore().build()"` (one-time, ~12 min).

## Environment

- LM Studio at `http://127.0.0.1:1234/v1` (OpenAI-compatible). Loaded model IDs in `/api/v0/models`; unload via `POST /api/v1/models/unload` with `{"instance_id": ...}`. Only ~12GB VRAM: never run two large models at once.
- LM Studio accepts but IGNORES Ling's `enable_thinking` chat_template_kwarg — Ling always reasons; budget max_tokens ≥4096 for it (see evaluation/model-benchmark.md). Empty QA model output gets ONE doubled-budget retry, then the honest notice — never the companion "I hear you" line (QA vs companion fallback semantics are mode-checked).
- Arabic FTS: `unicode61 remove_diacritics 2` does not strip Uthmani Quranic marks — the index stores a normalized `search` column from `ingestion/arabic_norm.py`; queries go through `search_form()` too. Never FTS against `quran.arabic`.
- Models emit malformed citation forms (`[quran:surah:al-imran:3:109]`) — the harness rebinds genuine quotes to the real citation id (right-to-left replacement); never trust a bare bracket form unless it resolves to a pack citation.

## Non-negotiable constraints (agentodo §0/§5/§29 + fixme_v3.1)

- Knowledge corpus is restricted to the approved **Sunni source registry**: never ingest, retrieve, cite, or summarize non-approved material. Gates run BEFORE ingestion (§5.2); enforcement lives in `agent/policy/source_policy.py` (`assert_ingestible`, `retrieval_filter`) and every new data path must call them.
- Never invent Qur'an references, hadith, scholars, book titles, gradings, or quotations. Qur'an text served to users comes only from the hash-verified `quran-uthmani-json` dataset. If unverifiable, return the exact §12 notice (`agent/validators/pipeline.py::UNVERIFIABLE_NOTICE`).
- **The LLM is not the source of Islam**: model pretraining is never evidence. Retrieval proposes; the Evidence Judge decides what Ilman may claim.
- **VALID_CITATION ≠ SUPPORTED_CLAIM** (fixme_v3.1): citation existence, relevance, and claim entailment are three separate checks (`agent/validators/evidence_judge.py`). Similarity ≠ entailment — cosine alone can never mark a claim SUPPORTED. Claim types gate language strength (guarantees need near-verbatim support; DIAGNOSIS/PREDICTION never allowed as religious certainty).
- **Inference laundering is blocked** ("A. Therefore B." — B is judged independently; removing A removes its dependents). Quarantined evidence NEVER re-enters generation (§4: no `passages[:1]` fallbacks); ubiquitous stems (allah/quran/prophet) don't count as relevance overlap.
- Never expose chain-of-thought — not in responses, not in traces, not in chat logs. Store only decisions, verdicts, provenance.
- Local-first: no mandatory cloud API. Model backends are replaceable behind OpenAI-compatible `/v1`; routing is runtime config (`configs/config.yaml`), never hard-coded (§3).
- RAG is the knowledge mechanism. Fine-tuning is behavior-only and gated on the Phase 5 baseline (§16–17). YouTube lectures are secondary material (§22). Never auto-publish religious content without a human review gate (§21).
- Mental-health guardrail (fixme_v3.1 §13): depression ≠ weak iman/punishment/shaytan — such equivalence claims classify as high-risk and are blocked without sources; the system never diagnoses.
- Safety overrides everything: high-risk input (EN/MS) short-circuits to the model-free canned response; companion policy can never soften it; no religious guilt in crisis mode.

## Working process

- Phase order (§26): Phases 0–5 + companion v1/v2/v2.1/v3.1 are done (all merged to `main`; see README status table + evaluation baselines: grounded 87%/7% hallucination, companion score 100.0, v3.1 false-support 0.00%). Remaining: fine-tuning (Phase 6, gated), content engine (Phase 7, ON HOLD per owner — reference `/mnt/AI/dev/tadabbur-yt` for mechanics), PWA/Android polish (Phase 8/9).
- The regression gates are mandatory after changes: `run_regression` (hallucination must not regress), `run_companion_eval` (crisis routing must stay 100%), `run_companion_score`, and `v31_metrics` (false-support must stay <5%).
- Classic English tafsirs (`tafsir-*-en`) are owner-approved for PRIVATE LOCAL STUDY ONLY — print copyright unresolved; never publish or redistribute their text (§21). Chunks came from quran_campaign's kb.db (hash-pinned); 164 mis-tagged chunks are in `tafsir_en_quarantine`, excluded from retrieval until human-fixed.
- Companion chat logs (`knowledge/processed/companion_logs/`, gitignored): all user+companion turns captured as structured JSONL for owner troubleshooting — local-only, never published, no chain-of-thought (§31), crisis turns flagged `sensitive` (exports default to excluding them; redaction via `DELETE /api/v1/logs/{session}` or `--redact`). Eval/test session ids (`case-`, `scen-`, `eval-`, `s1`-`s6`, `adv-`, `v3-`, `dua-test`, `pillar-`, `fatihah-`, `owner-` is NOT excluded) never log. Analyze: `uv run python -m scripts.companion_logs stats|sessions|read|export|export-all|watch`.
- Adding a data source: add to `knowledge/registry/approved_sources.yaml` (with license + verification), then an ingestor module calling `assert_ingestible` before any write; dataset file hash goes in registry notes (pattern: `quran-uthmani-json`).
- Branch convention: feature work on `featureNN` branches, merge to `main` with `--no-ff` after gates pass. Never push chat logs, derived DBs, or the classic-tafsir source DB.

## Models (§2, §4)

- ling_tiny = primary agent/reasoning model; gemma_qat (gemma-4-12b-it-qat, official) = secondary; qwen_small/medium = benchmark candidates. All swappable via config without code changes.
- `gemma-4-e4b-uncensored-hauhaucs-aggressive` is a third-party variant — disqualified for religious answers by regression (invented grading, 0% refusal). Religious task classes route to ling_tiny or gemma_qat.
- Target hardware Profile A: RTX 3060 12GB + 64GB RAM. Start at 4K–8K context (measure KV-cache cost before increasing); keep 8–12GB RAM free; activated parameter count ≠ VRAM.

## Acceptance

- agentodo §27 "Definition of Done" + fixme_v3.1 §41 checklist: provenance on every claim, deterministic Qur'an refs, hadith attribution metadata, uncertainty admitted, no fabricated citations, no false religious confidence. When uncertain: **less claim, more transparency** (fixme_v3.1 §44).
