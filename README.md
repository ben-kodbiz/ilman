# Ilman (Huurs)

Local-first, source-grounded Sunni Islamic study assistant and Islamic
companion. The design spec is [`agentodo.md`](agentodo.md); the companion
specifications are [`fix_me.md`](fix_me.md), [`fixme_v2.md`](fixme_v2.md),
[`fixme_v3.md`](fixme_v3.md) and [`fixme_v3.1.md`](fixme_v3.1.md); project
conventions live in [`AGENTS.md`](AGENTS.md).

**The LLM is not the source of Islam** (spec §29): every answer is retrieval +
validation over the approved corpus, never model memory. And (fixme_v3.1 §44):
**false religious confidence is a more serious failure than an incomplete
answer** — the system optimizes for never pretending weak evidence is strong.

## Status

| Layer (spec) | State |
|---|---|
| 0 — architecture, source policy, model abstraction | done |
| 1 — model benchmark harness | done (Ling report + 4-model companion benchmark) |
| 2 — knowledge core: Qur'an (hash-verified Uthmani), EN translation, Kutub al-Sittah hadiths (34,153, gradings preserved), Tafsir Kemenag (6236/ayah, + ID translation), classic EN tafsirs (Sa'di/Ibn Kathir/Qurtubi, 18,940 chunks, 164 quarantined), IslamQA.info EN fatwas (TIER 4 contemporary, islamqa-info-en) | done |
| 3 — hybrid retrieval: reference + Arabic/EN-ID translation/hadith/tafsir FTS + vector leg (65,565 nomic embeddings), RRF fusion, tier balancing, concept anchors, mandatory source filter | done |
| 4 — Agent: intent router, memory (§15), tool-calling loop, query planner, evidence quarantine, repair rounds | done |
| 5 — grounded regression (87% / 7% hallucination), companion eval (100.0 score), v3.1 validation metrics (0.00% false-support) | done |
| Companion v1 (fix_me.md): empathy-first, crisis short-circuit, intent/emotion/state, memory controls | done |
| Companion Harness v2 (fixme_v2): state → policy → context → model → validation | done |
| Evidence Judge v2.1 (fixme_v3): claim→evidence entailment, query planning, quarantine | done |
| Validation hardening v3.1 (fixme_v3.1): claim taxonomy, inference boundaries, per-claim verdicts, bounded repair + final gate | done |
| Dev console (Gradio) + PWA web client + chat logging | done |
| 6 — fine-tuning | not started (gated: must beat Phase 5 baseline, §16–17) |
| 7 — content engine (YouTube/Tadabbur ingestion → drafts → human review) | on hold |
| 8/9 — PWA polish, Android offline packs | seed done |

All work is merged to `main`. Corpus license notes: hadiths (Unlicense) and
Kemenag (MIT) are open; **classic English tafsirs are owner-approved for
private local study only** (print copyright unresolved, §21 hold — never
published or redistributed).

## The answer pipeline (what actually runs)

```text
USER
  ↓ SAFETY GATE — model-free, high-risk short-circuits to a canned
  ↓           compassionate response (never religious guilt)
UNDERSTAND — deterministic intent + emotion + entity classification (no model)
  ↓
STATE ENGINE — mode (qa/study/companion/reflection/dua/crisis), risk level,
  ↓           emotion continuity across threads
POLICY ENGINE — machine-readable ResponsePolicy: RAG on/off, follow-up
  ↓           limits, tone, word budget, safety override
MEMORY ROUTER — categorized facts (only stable, explicitly-shared ones;
  ↓           transient emotions never persist) + relevance retrieval
QUERY PLANNER — structured info-need; modern terms expand to classical
  ↓           concepts; concept anchors load canonical sources directly
HYBRID RETRIEVAL — 5 lexical legs + vector leg, RRF fusion
  ↓
EVIDENCE QUARANTINE — irrelevant passages removed BEFORE the model sees them
  ↓           (all-irrelevant → INSUFFICIENT_EVIDENCE, never reintroduced)
CONTEXT BUILDER — budgeted ContextPack (recent turns + relevant memory +
  ↓           evidence only; never whole-database prompts)
LOCAL MODEL (Ling / Gemma / Qwen — runtime-config routed, swappable)
  ↓
CLAIM EXTRACTION — quote-aware, typed (DIRECT_FACT/ATTRIBUTION/INFERENCE/
  ↓           CAUSAL/GUARANTEE/RULING/DIAGNOSIS/PREDICTION/…)
EVIDENCE JUDGE — per-claim entailment vs the pack: SUPPORTS/PARTIAL/
  ↓           BACKGROUND/IRRELEVANT; three-level citation checks
  ↓           (exists ≠ relevant ≠ supports); claim-strength policy
  ↓           (guarantees need near-verbatim; diagnosis never allowed)
LANGUAGE GATE — strong connectives ("The Quran proves…", "cures…") only on
  ↓           SUPPORTS claims
BOUNDED REPAIR — max 2 rounds, ALWAYS revalidated; unsupported claim
  ↓           sentences + their inference dependents removed
FINAL ANSWER GATE — surviving high-risk claims force the safe fallback;
  ↓           diagnosis sentences stripped; follow-up count ≤ 1
COMPANION VALIDATION — no dependency language, no human pretense,
  ↓           no preachy openers, crisis-mode guilt ban
USER
```

Core invariants (fixme_v3.1): `VALID_CITATION ≠ SUPPORTED_CLAIM`;
`similarity ≠ entailment` (cosine alone can never reach SUPPORTS);
quarantined evidence never re-enters generation; one unsupported high-risk
claim cannot hide behind supported minor claims.

## Companion behavior

- **"I feel lonely."** → empathy first, no verse dump (mode: companion,
  `requires_rag: false`, one gentle follow-up). Islamic guidance is *offered*,
  retrieved only when asked or the policy engine decides it helps.
- **"What does Islam say about loneliness?"** → empathize, then cited RAG.
- **"Is there any dua for depression?"** → candidate supplication set
  retrieved; the actual Prophetic dua quoted with citation; careful scope
  ("no guaranteed cure", professional-support nudge) — the original
  inference-laundering failure (`Allah alone cures depression [quran:112:4]`)
  is a permanent regression test.
- **"I want to kill myself"** (EN/MS) → model-free canned safety response,
  real-world contact guidance, crisis mode, logged as `sensitive`.
- Emotional statements route through concept expansions
  ("loneliness" → "Allah is near responds to dua" → 13:28/Bukhari 6369);
  mental-health terms never auto-equate to Islamic concepts (§13: depression
  ≠ weak iman ≠ punishment — classified high-risk when asserted).

## Commands

```bash
uv sync                                                            # deps
uv run pytest -q                                                   # 445 tests (live tests auto-skip without a model)
uv run ruff check agent ingestion retrieval evaluation apps tests scripts

# --- ingest corpora (hash-gated; server refuses to start without Qur'an) ---
uv run python -c "from ingestion.quran_ingest import QuranIngestor; QuranIngestor().ingest()"
uv run python -c "from ingestion.hadith_ingest import HadithIngestor; HadithIngestor().ingest_all()"
uv run python -c "from ingestion.tafsir_ingest import TafsirIngestor; TafsirIngestor().ingest()"
uv run python -c "from ingestion.tafsir_en_ingest import TafsirEnIngestor; TafsirEnIngestor().ingest_all()"
uv run python -c "from ingestion.web_fatwa_ingest import WebFatwaIngestor; WebFatwaIngestor().ingest()"  # after scripts/harvest_islamqa.py
uv run python -c "from retrieval.vector_store import VectorStore; VectorStore().build()"   # ~12 min, one-time

# --- run ---
uv run uvicorn apps.api.server:app --host 127.0.0.1 --port 8017     # API + PWA
uv run python -m apps.devui.console                                 # Gradio dev console :7860

# --- gates (run after any routing/prompt/validator/companion change) ---
uv run python -m scripts.run_regression --models ling_tiny           # grounded regression
uv run python -m scripts.run_companion_eval --models ling_tiny      # §21 companion matrix
uv run python -m scripts.run_companion_score --models ling_tiny      # fixme_v2 §28 weighted score
uv run python -m evaluation.bench.v31_metrics                       # validator quality (fails >5% false-support)

# --- chat log analysis (all companion turns captured) ---
uv run python -m scripts.companion_logs stats|sessions|watch
uv run python -m scripts.companion_logs read <session>
uv run python -m scripts.companion_logs export <session> -o out.md   # human-readable transcript
```

The chat endpoints need an OpenAI-compatible backend (default: LM Studio at
`127.0.0.1:1234`, see `configs/config.yaml`). Search works without a model.

## Evaluation baselines

| Gate | Baseline |
|---|---|
| Grounded regression (§19-20) | **87% pass, 7% hallucination** (ling-3.0-tiny; remaining failures are known Ling CoT-variance + the 40:40 rationalization trap, all checker-verified) |
| Companion matrix (fix_me.md §21) | **25/25 on all four models** (ling / gemma-4-12b-qat / qwen3.5-4b / qwen3.5-9b); crisis routing 7/7 everywhere |
| Companion score (fixme_v2 §28) | **100.0** (31/32 cases, 5/5 multi-turn scenarios) |
| v3.1 validator quality (fixme_v3.1 §34-36) | **100% on 93 cases; RELIGIOUS_FALSE_SUPPORT_RATE 0.00%; unsupported-claim escape 0.00%** |

Reports: `evaluation/grounded-regression.md`, `evaluation/companion-benchmark.md`,
`evaluation/model-benchmark.md`, `evaluation/v3_1/README.md`.

## Dev console

`uv run python -m apps.devui.console` → http://127.0.0.1:7860

- **Companion (v2)** tab — full harness chat with live developer trace
  (intent/emotion/risk/policy/validation/evidence-status/planned-query),
  model-role switching, session id, memory view/clear
- **Grounded QA** tab — AgentOrchestrator with tool-call and citation metadata
- **Search (all legs)** tab — raw hybrid retrieval for corpus debugging

## Chat logging

Every user→companion exchange is captured (local-only, gitignored
`knowledge/processed/companion_logs/`) as structured JSONL: full pipeline
metadata per turn — mode, intent, emotion, risk, citations, per-claim
verdicts, evidence sufficiency, policy decision, planned query, latency.
**No chain-of-thought ever** (§31). Crisis turns are flagged `sensitive`
(excluded from exports by default; redaction via `DELETE /api/v1/logs/{id}`
or `--redact`). Eval/test sessions never log. This is the troubleshooting
dataset for future pipeline enhancement.

## API surface

- `POST /api/v1/companion` — v2 harness (public shape, no internal trace);
  v1 engine kept at `/api/v1/companion/v1`
- `POST /api/v1/answer` — grounded QA agent (tool loop)
- `GET /api/v1/search` — hybrid retrieval; `/api/v1/hadith/*`, `/api/v1/tafsir/*`
- `GET/DELETE /api/v1/logs/*`, `/api/v1/memories*` — log + memory controls (§25)
- OpenAPI docs at `/docs`

## Layout

- `agent/core/` — model router, agent orchestrator, companion harness, query
  planner, observability (DebugTrace)
- `agent/state/`, `agent/policy/`, `agent/context/`, `agent/safety/` — v2 harness
  layers (state machine, ResponsePolicy, ContextPack, risk gate)
- `agent/companion/` — intent classifier, memory router, safety patterns,
  **chat logging**
- `agent/validators/` — evidence packs, citation validation, grade-attribution
  check, **claims + claim_policy (typed taxonomy), evidence_judge (entailment),
  companion_validator (tone/policy), entailment (§23 interface)**
- `agent/policy/source_policy.py` — Sunni registry gates (ingest + retrieval)
- `ingestion/` — hash-verified Qur'an/translation/hadith/tafsir ingestors,
  Arabic search normalization
- `retrieval/` — hybrid retrieval + vector store (nomic embeddings)
- `evaluation/` — suites, runners, reports, `companion/` (cases/scenarios),
  `v3_1/` (93-case validator dataset + metrics)
- `apps/api/`, `apps/web/`, `apps/devui/` — FastAPI server, PWA client,
  Gradio console
- `scripts/` — regression/companion/score runners, companion_logs CLI
- `tests/` — **445 tests** (policy gates, retrieval, validators incl. the
  v3.1 battery, companion golden tests, chat logging, dev console)

## Invariants (never bypass)

- Nothing is ingested or retrieved outside the approved Sunni registry
  (`SourcePolicy.assert_ingestible` / `retrieval_filter`) — gates run BEFORE
  ingestion, never filter-later.
- Qur'an text served to users comes only from the hash-verified
  `quran-uthmani-json` dataset; model output is never Qur'an.
- Every source-dependent answer carries an evidence pack; citations are
  validated (existence), judged for relevance and **claim entailment**
  (v3.1 three-level check); unsupported claims are removed with their
  inference dependents, never averaged away.
- Similarity ≠ entailment: cosine/embedding scores alone can never mark a
  claim SUPPORTED; claim type gates the strength of language allowed.
- Diagnosis ("you have depression") and predictions ("Allah will…") never
  ship as religious certainty; mental-health ≠ spiritual-state equivalence
  claims are high-risk and blocked without authoritative sources.
- Crisis input never reaches the model; safety policy overrides companion
  policy; no religious guilt in crisis mode.
- No evidence → `I could not verify this from the approved source corpus.`
  — never a guess. When uncertain: less claim, more transparency.
