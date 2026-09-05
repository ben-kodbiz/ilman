# v3.1 Validation Dataset (fixme_v3.1 §34-36)

Deterministic validation-quality benchmark — measures the VALIDATOR stack
(claim extraction → evidence judge → language gate), not the model. 93 cases
across seven categories; model behavior is covered by the grounded +
companion suites.

## Run

```bash
uv run python -m evaluation.bench.v31_metrics
```

Exits non-zero if `RELIGIOUS_FALSE_SUPPORT_RATE` or
`UNSUPPORTED_CLAIM_ESCAPE_RATE` exceed 5%.

## Categories

| file | category | covers |
|---|---|---|
| citation.yaml | citation traps | exists≠relevant≠supports (§5), not-in-pack, no-pack |
| entailment.yaml | entailment | inference chains, similarity≠entailment (§22), quote vs paraphrase |
| attribution.yaml | attribution | grader/scholar misuse, misquotation, enumeration |
| mental_health.yaml | mental health | §13 guardrail: depression↔iman/punishment/shaytan equivalence ban |
| dua.yaml | dua requests | §14-16: cure-guarantee upgrades, planner candidate sets |
| companion.yaml | companion claims | §27: dependency, exclusivity, spiritual guilt forms |
| adversarial.yaml | adversarial | §18 model-answer forms of the adversarial queries |
| rulings.yaml | rulings | fiqh-grade claims need authoritative sources |

## Metrics (§35-36)

- **RELIGIOUS_FALSE_SUPPORT_RATE** (§36, the critical one): unsupported
  religious claims wrongly classified as supported. Target <5% (then 2%/1%);
  current baseline **0.00%**.
- **UNSUPPORTED_CLAIM_ESCAPE_RATE** (§35): unsupported claims reaching
  ANSWERABLE/SUPPORTS verdicts. Current baseline **0.00%**.
- per-category pass rates; overall currently 100%.

## Case expectation forms

```yaml
expect: {no_supports: true}          # nothing may judge SUPPORTS
expect: {no_inference_supports: true} # inference/guarantee/causal claims only
expect: {supports_citation: cid}     # that citation must be SUPPORTS
expect: {partial_ok: cid}            # PARTIAL-or-better (heuristic-judge limit case)
expect: {sufficiency_not: answerable} # aggregate sufficiency bound
expect: {extracts_type: claim_type}  # typed claim-extraction check
expect: {language_violation: bool}    # §24 language gate behavior
```

## Known heuristic-judge limit (documented, by design)

True semantic paraphrases with disjoint stems ("no equal" ↔ "none comparable")
cannot reach SUPPORTS by the lexical+stem judge — cosine similarity alone is
deliberately barred from SUPPORTS (§22). `cite_006` encodes this: the honest
PARTIAL verdict, pending the optional local-NLI `EntailmentBackend`
(`agent/validators/entailment.py`, §23) being adopted after benchmarking.
