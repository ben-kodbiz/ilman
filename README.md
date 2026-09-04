# Huurs

Local-first, source-grounded Sunni Islamic study and content assistant.
The design spec is [`agentodo.md`](agentodo.md); project conventions live in
[`AGENTS.md`](AGENTS.md). **The LLM is not the source of Islam** (spec §29):
every answer is retrieval + validation over the approved corpus, never model
memory.

## Status

| Phase (spec §26) | State |
|---|---|
| 0 — architecture, source policy, model abstraction | done |
| 1 — model benchmark harness | built; Ling report in `evaluation/model-benchmark.md`; 4-model run deferred |
| 2 — Qur'an knowledge core (SQLite + FTS5, 6236 ayahs, stable citation IDs, hash-verified ingestion) | done |
| 2 — English translation corpus (aligned per-ayah, CC-BY-SA-4.0, translation FTS leg) | done |
| 2 — Tafsir Kemenag (TIER 2, 6236 per-ayah entries, MIT) + Indonesian translation | done |
| 2 — Classic English tafsirs (Sa'di, Ibn Kathir, Qurtubi; 18,940 chunks, quran_campaign) | done |
| 2 — Kutub al-Sittah hadith corpus (34,153 hadiths, AR+EN, grading metadata preserved) | done |
| 3 — hybrid retrieval (reference + Arabic/translation/hadith/tafsir FTS + vector leg, RRF, source filter) | done |
| 4 — Agent: intent router, memory (§15), tool-calling loop, repair rounds | done |
| 5 — Grounded evaluation suite + 4-model regression baseline | done |
| Companion enhancement (fix_me.md) — safety routing, intent, state, memory controls, eval | done |
| 8 — API server + web client (PWA shell) | seed done |

Hadith corpus: all six Kutub al-Sittah collections ingested from the
registry-approved `fawazahmed0/hadith-api` dataset (Unlicense). Grading
metadata is preserved verbatim from the dataset where present (Abu Dawud,
Tirmidhi, Nasa'i, Ibn Majah); Bukhari/Muslim carry collection-level sahih
grading and their records state this — per-hadith gradings are never invented.

Tafsir corpus: Tafsir Kemenag (official Indonesian Ministry of Religious
Affairs exegysis, TIER 2, MIT-licensed compilation) — 6236 per-ayah entries,
paired automatically with referenced ayahs in retrieval, always rendered as
interpretation, never as Qur'an text. The same dataset added an Indonesian
(Kemenag) Qur'an translation. Ibn Kathir English remains registry-pending
(no licensed machine-readable edition yet).

Tafsir corpus now spans four works: Tafsir Kemenag (Indonesian, per-ayah) plus
three classical English tafsirs ingested from the quran_campaign extraction
(as-Sa'di 10 vols, Ibn Kathir abridged, al-Qurtubi vols covering surahs 1-94)
— 18,940 ayah-anchored chunks with scholar/volume/page provenance; 164 chunks
with misparsed ayah tags are quarantined with reasons, never silently dropped.
License note: the English print-edition copyrights are unresolved; these are
owner-approved for private local study ONLY — never published or redistributed
(§21). The quran_campaign DB's own quran/translations tables were truncated
and were NOT imported; Qur'an text remains the hash-verified Uthmani dataset.

Phase 5 regression baseline (2026-09-04, see `evaluation/grounded-regression.md`):
with the vector retrieval leg live, Ling passes 80% with hallucination DOWN to
7% (was 13%) — the semantic leg + exact-claim-verification prompt fixed the
Ya-Sin and fake-verse rationalization traps. Emotional statements ("I am
lonely") route through intent-detected concept expansions to genuinely
comforting evidence. Known remaining: Bukhari-1 grading-misattribution probe
(parallel-narration grades), Ling CoT variance on large packs.

The loaded third-party E4B variant is disqualified for
religious answers (invented grading, 0% refusal) — use official checkpoints.

## Companion mode

`POST /api/v1/companion` — the context-aware companion (fix_me.md):
empathy-first responses, conversation modes (qa/study/companion/reflection/
dua/crisis) shown in the UI, short-lived conversation state (never silently
persisted), memory controls (view/clear/disable), and a **model-free crisis
short-circuit** — high-risk input (EN/MS) never reaches the LLM; it returns a
canned compassionate safety response with real-world contact guidance.
Religious claims in companion replies pass the same deterministic citation
validation as Q&A mode; fabricated citations get their sentences removed or
the honest §12 notice.

```bash
uv run python -m scripts.run_companion_eval --models ling_tiny   # §21 matrix
```
Latest: **25/25 pass on all four benchmark models** (ling / gemma-4-12b-qat /
qwen3.5-4b / qwen3.5-9b), crisis routing 7/7 everywhere — see
`evaluation/companion-benchmark.md`. Correctness is architecture-driven
(routing + validation), so even the 4B model is companion-grade.

## Quick start

```bash
uv sync                                   # deps
uv run pytest -q                          # tests (needs nothing external)
uv run python -c "from ingestion.quran_ingest import QuranIngestor; QuranIngestor().ingest()"
uv run python -c "from ingestion.hadith_ingest import HadithIngestor; HadithIngestor().ingest_all()"
uv run python -c "from ingestion.tafsir_ingest import TafsirIngestor; TafsirIngestor().ingest()"
uv run uvicorn apps.api.server:app --host 127.0.0.1 --port 8017
# open http://127.0.0.1:8017/  (API docs at /docs)
```

The answer endpoint needs an OpenAI-compatible backend (default: LM Studio at
`127.0.0.1:1234`, see `configs/config.yaml`). Search works without a model.

## Layout

- `agent/` — intent router, memory store (§15), model router, AgentOrchestrator (tool loop)
- `agent/policy/` — approved/excluded registries, ingestion gate, retrieval filter
- `agent/tools/` — schema-strict tool layer (Qur'an + hadith + memory) + reference handling
- `agent/validators/` — evidence packs, citation validation, grounded response pipeline
- `knowledge/registry/` — approved_sources.yaml / excluded_sources.yaml (§5)
- `ingestion/` — hash-verified Qur'an/translation/hadith ingestion, Arabic search normalization
- `retrieval/` — hybrid retrieval (5 lexical legs + vector leg w/ nomic embeddings, RRF, tier balancing, source filter)
- `evaluation/` — benchmark suites, grounded regression (§19-20), reports
- `apps/api/`, `apps/web/` — FastAPI server (agent + pipeline modes), web client
- `tests/` — 172 tests; gates, retrieval, validators, memory, intent, agent (mock + live)

## Invariants (never bypass)

- Nothing is ingested or retrieved outside the approved registry
  (`SourcePolicy.assert_ingestible` / `retrieval_filter`).
- Qur'an text served to users comes only from the hash-verified dataset
  (`quran-uthmani-json`); model output is never Qur'an.
- Every source-dependent answer carries an evidence pack; citations are
  validated against it, unsupported citations fail the response.
- No evidence -> `I could not verify this from the approved source corpus.` — never a guess.
