# Companion Benchmark (fix_me.md §19)

> Generated 2026-09-05. Identical engine, prompts, state handling, memory and
> deterministic checkers across models (§19 requirements). Crisis routing is
> model-free (canned response) — it passes by construction and any failure
> would be a code regression, not a model difference.

## Results

| model | cases | pass | crisis routing | avg latency s | notes |
|---|---|---|---|---|---|
| ling-3.0-tiny (Q5_K_L) | 25 | 100% | 7/7 | 2.8 | fastest; concise, warm; occasionally double-questions without the hard rule |
| **gemma-4-12b-it-qat (official)** | 25 | 100% | 7/7 | 17.3 | warmest tone, natural phrasing; slower |
| qwen3.5-9b (Q6_K) | 25 | 100% | 7/7 | 31.3 | safest; longest reasoning latency |
| qwen3.5-4b (Q8_0) | 25 | 100% | 7/7 | 19.7 | solid for its size; the official E4B-class candidate profile |

All models: no fabricated citations (validator-enforced), no dependency
language (guard-enforced), no diagnosis language, single-question rule
followed, empathy-first on all 11 emotional/ambiguous cases.

## Findings

1. **Companion quality is architecture-driven, not model-driven.** The
   §29 rule proved out: state + routing + validation + canned safety make
   every model pass — the differences are latency and prose warmth, not
   correctness. The small 4B model is fully usable for companion mode.

2. **Routing decisions matter more than model choice.** The deterministic
   intent classifier decides empathy-first vs RAG; the validator catches
   fabricated citations; the crisis short-circuit never reaches any model.

3. **Model recommendation for companion mode:** ling_tiny for interactive
   latency (2.8s avg), gemma-4-12b-it-qat when warmth of prose matters most.
   qwen3.5-4b is the best mobile-profile candidate (§29: acceptable on
   local hardware).

4. **Measurement limits (honest):** the checkers are deterministic proxies
   — they verify structure (empathy-first, question count, no fabrication,
   safety) but not the full subjective warmth of prose. Human review of
   transcripts is the next step for tone quality (§20 requires human review
   for final evaluation samples).

## Re-run

```bash
uv run python -m scripts.run_companion_eval --models ling_tiny gemma_qat qwen_small qwen_medium
```
