# Model Benchmark — Phase 1 Report (partial: Ling-3.0-tiny only)

> Status: full 4-model comparison deferred by owner on 2026-09-04 ("stop benchmarks,
> proceed to building"). This report captures verified Ling findings from the two
> smoke runs so the Phase 1 exit decision has data. Re-run the full comparison with:
> `scripts/dev.sh bench --models ling_tiny gemma_qat qwen_small qwen_medium`

## Environment

- Backend: LM Studio (OpenAI-compatible `/v1` on 127.0.0.1:1234)
- Hardware: Profile A — RTX 3060 12GB, 64GB RAM, Linux (agentodo.md §4)
- Model under test: `ling-3.0-tiny` (bartowski GGUF, Q5_K_L, `bailingmoe3` arch,
  loaded context 8192)
- Raw data: `evaluation/results/20260904-080306-results.jsonl` (routine suite),
  `evaluation/results/20260904-080845-results.jsonl` (full attempt, Ling completed)

## Ling-3.0-tiny results (18 cases)

| suite | passed | notes |
|---|---|---|
| 01-routine (3) | 3/3 | echo, Malay translation, date math — all correct |
| 02-reasoning (4) | 4/4 | syllogism, arithmetic, UNANSWERABLE trap, sorting — all correct |
| 03-tools (3) | 3/3 | correct `get_ayah`, `verify_quran_reference` calls incl. args; no spurious tool use |
| 04-structured (3) | 1/3 | refs-array content correct but nested-colon format; two `finish_reason=length` |
| 05-traps (5) | 3/5 | Arabic preservation exact ×2, fake-verse & wrong-scholar traps passed; hadith trap failed on token cap |

Overall: **14/18 (78%)**, with both reasoning-heavy failures caused by the same
root issue below rather than model capability.

## Key finding 1 — reasoning budget dominates cost (§2.1 blocker)

Ling always emits `reasoning_content` on this stack. Measured smoke cases:
3/3 routine cases generated **97–303 reasoning tokens for 1–3 token answers**.
Two structured-output cases hit `finish_reason=length` at 1024 max_tokens with
**100% of output being reasoning** (1024/1024 and 1020/1024), producing empty
content — scored as failures but actually budget failures.

Implications for this project:

- Routine requests (`simple_chat`, `classification`, `entity_extraction`) must
  not route to Ling: per §2.1 thinking should be OFF for routine work, but see
  finding 2 — the toggle is unavailable on this backend, so Gemma handles those
  classes (already the config default in `configs/config.yaml`).
- Ling-targeted requests need `max_tokens >= 4096` default (runner default is now
  4096) and generation-speed accounting must subtract reasoning tokens.

## Key finding 2 — LM Studio ignores Ling's `enable_thinking` kwarg

- `chat_template_kwargs: {"enable_thinking": false}` → HTTP 200, but reasoning
  token counts identical to `enable_thinking: true` (303/303 on a controlled
  A/B). The kwarg is silently dropped for the `bailingmoe3` template.
- `reasoning_format: "disabled"` also accepted, also ineffective.
- Conclusion: on LM Studio, Ling behaves as an always-thinking model. §2.1's
  "thinking=false for routine requests" is **not implementable against this
  backend**; it requires a vLLM/SGLang backend (spec §2.1 anticipated a separate
  backend test) or acceptable thinking-always routing.

## Key finding 3 — tool calling is solid

Native OpenAI-style `tool_calls` returned with correct JSON-string arguments
(`get_ayah {"surah":2,"ayah":255}`) and `finish_reason: "tool_calls"`. The agent
tool layer (`agent/tools/layer.py`) can rely on this format.

## Key finding 4 — honest uncertainty behavior (§20)

- Unanswerable trap ("how old is the captain") → exact `UNANSWERABLE`. 
- Fabricated verse trap → declined to invent a citation (passed
  `not_hallucinating`).
- Wrong-scholar-attribution trap → flagged as incorrect.
This is the single most important capability for this project and Ling passes.

## Phase 1 exit decision (task-class defaults, per §26)

| task class | default model | basis |
|---|---|---|
| tool_calling, complex_rag, difficult_reasoning, source_comparison, content_generation | ling_tiny | verified reasoning + tool calls + honest refusals |
| simple_chat, classification, entity_extraction, simple_rag | gemma_qat | by routing design (§3); Ling cannot suppress thinking on this backend, making it cost-ineffective for routine classes |
| mobile_default | gemma_qat | by design (§4 Profile C) |

**Deferred:** Gemma-4-12B QAT / Qwen 3.5 4B / 9B numbers (latency, VRAM, tok/s),
Ling VRAM ceiling at 8K vs 4K context, full adversarial-suite stats. Note: the
official Gemma E4B QAT checkpoint from §2.2 is not present in LM Studio's
library; available E4B is a third-party variant. If Phase 9 (Android) proceeds,
download the official `google/gemma-4-E4B-it-qat-q4_0-gguf` first.

**Operational note:** two resident models cannot coexist in 12GB VRAM with 8K
context (Gemma/Qwen JIT-loads failed while Ling + embeddings were resident).
The runner must unload between models — `/api/v1/models/unload` works; also
unload `text-embedding-nomic-embed-text-v1.5` which loads by default.
