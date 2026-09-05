"""fixme_v3.1 §34-36 — validation-quality evaluation.

100-case dataset across seven categories, run through the REAL validator
stack (judge + claim policy + language gate) with **fixed model answers**:
this measures VALIDATOR quality (escape/false-support rates), not model
quality — models are measured by the companion/grounded suites.

Metrics (§35-36):
    RELIGIOUS_FALSE_SUPPORT_RATE   (§36 — the critical one)
    unsupported_claim_escape_rate
    claim extraction recall/precision (typed)
    citation existence/relevance accuracy
    repair/post-repair handled by the harness live suite
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from agent.validators.claim_policy import extract_typed_claims
from agent.validators.evidence_judge import (
    EvidenceJudge,
    Verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _embed(text: str):
    """Lazy semantic signal: corpus embedder when LM Studio is reachable,
    else None (judge degrades to lexical-only)."""
    global _EMBED_CACHE
    if _EMBED_CACHE is not None:
        return _EMBED_CACHE(text)
    try:
        from agent.core.config import load_config
        from agent.core.embeddings import EmbeddingClient

        client = EmbeddingClient(load_config())
        client.embed_one("ping")
        _EMBED_CACHE = client.embed_one
    except Exception:
        _EMBED_CACHE = None
    return _EMBED_CACHE(text) if _EMBED_CACHE else None


_EMBED_CACHE = None
DATASET_DIR = REPO_ROOT / "evaluation" / "v3_1"
RESULTS_DIR = REPO_ROOT / "evaluation" / "results"

# ---------------------------------------------------------------- helpers


def _passage(row: dict):
    from retrieval.hybrid import RetrievedPassage

    cid = row["citation_id"]
    if cid.startswith("hadith:"):
        return RetrievedPassage(
            citation_id=cid, surah=0, ayah=0, arabic="", translation=row["text"],
            source_id=cid.split(":")[1], tier=1, leg="hadith", score=-1.0,
            collection=cid.split(":")[1], hadithnumber=int(cid.split(":")[2]),
        )
    if cid.startswith("tafsir-en:"):
        return RetrievedPassage(
            citation_id=cid, surah=2, ayah=153, arabic="", translation=row["text"],
            source_id="tafsir-sadi-en", tier=2, leg="tafsir", score=-1.0,
            scholar="al-Sa'di",
        )
    parts = cid.split(":")
    s, a = int(parts[1]), int(parts[2])
    return RetrievedPassage(
        citation_id=cid, surah=s, ayah=a, arabic="", translation=row["text"],
        source_id="quran-uthmani-json", tier=0, leg="reference", score=1.0,
    )


@dataclass
class CaseOutcome:
    case_id: str
    category: str
    passed: bool
    detail: str = ""


def run_case(case: dict) -> CaseOutcome:
    """Each case: evidence passages + a candidate answer + the expectation.

    Expectation forms:
      expect: {no_supports: true}        — nothing may judge SUPPORTS
      expect: {supports_citation: cid}   — that citation must be SUPPORTS
      expect: {sufficiency_not: answerable}
      expect: {extracts_type: claim_type} — claim-type extraction check
      expect: {language_violation: true} — must trip the §24 gate
      expect: {language_violation: false}
    """
    from agent.validators.pipeline import EvidencePack

    judge = EvidenceJudge(embed=_embed)
    if "passages" in case:
        pack = EvidencePack(
            query=case.get("query", "q"),
            passages=[_passage(r) for r in case["passages"]],
        )
    else:
        pack = None
    answer = case.get("answer", "")

    if "extracts_type" in case["expect"]:
        claims = extract_typed_claims(answer)
        want = case["expect"]["extracts_type"]
        ok = any(c.claim_type.value == want for c in claims)
        return CaseOutcome(case["id"], case["category"], ok,
                            f"want type {want}; got {[c.claim_type.value for c in claims]}")

    judgement = judge.judge_answer(answer, pack, topic=case.get("topic"))
    exp = case["expect"]

    if exp.get("no_supports"):
        ok = all(x.verdict is not Verdict.SUPPORTS for x in judgement.claim_support)
        return CaseOutcome(case["id"], case["category"], ok,
                            f"verdicts: {[x.verdict.value for x in judgement.claim_support]}")

    if exp.get("no_inference_supports"):
        # §9: inference/guarantee claims must not be SUPPORTS; premises may be
        bad = [
            x for x in judgement.claim_support
            if x.verdict is Verdict.SUPPORTS
            and x.claim_type in ("inference", "guarantee", "causal_claim")
        ]
        return CaseOutcome(
            case["id"], case["category"], not bad,
            f"supported inference claims: {[x.claim[:60] for x in bad]}",
        )

    if "supports_citation" in exp:
        cid = exp["supports_citation"]
        ok = any(
            x.verdict is Verdict.SUPPORTS and x.citation == cid
            for x in judgement.claim_support
        )
        return CaseOutcome(case["id"], case["category"], ok, f"wanted SUPPORTS on {cid}")

    if "sufficiency_not" in exp:
        bad = exp["sufficiency_not"]
        ok = judgement.sufficiency.value != bad
        return CaseOutcome(case["id"], case["category"], ok,
                           f"sufficiency {judgement.sufficiency.value} != {bad}")

    if "language_violation" in exp:
        from agent.validators.evidence_judge import language_strength_ok

        violations = language_strength_ok(answer, judgement)
        ok = bool(violations) is exp["language_violation"]
        return CaseOutcome(case["id"], case["category"], ok,
                           f"violations={violations[:1]} want={exp['language_violation']}")

    if "sufficiency_is" in exp:
        ok = judgement.sufficiency.value == exp["sufficiency_is"]
        return CaseOutcome(case["id"], case["category"], ok,
                           f"sufficiency {judgement.sufficiency.value}")

    return CaseOutcome(case["id"], case["category"], True, "no expectation")


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for path in sorted(DATASET_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for case in data.get("cases", []):
            case.setdefault("category", data.get("category", path.stem))
            cases.append(case)
    return cases


def compute_metrics(outcomes: list[CaseOutcome], cases: list[dict]) -> dict:
    by_cat: dict[str, list[CaseOutcome]] = {}
    for o in outcomes:
        by_cat.setdefault(o.category, []).append(o)
    total = len(outcomes)
    passed = sum(o.passed for o in outcomes)
    metrics = {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "per_category": {
            cat: {"passed": sum(o.passed for o in outs), "total": len(outs)}
            for cat, outs in sorted(by_cat.items())
        },
    }

    # §35-36: escape rate — unsupported religious claims the validator
    # wrongly classified as supported (i.e. no_supports cases that PASSED as
    # supported). A failure of a no_supports case IS a false support.
    false_support = sum(
        1 for c, o in zip(cases, outcomes)
        if c["expect"].get("no_supports") and not o.passed
    )
    no_support_total = sum(1 for c in cases if c["expect"].get("no_supports"))
    metrics["religious_false_support_rate"] = (
        round(false_support / no_support_total, 4) if no_support_total else 0.0
    )
    # escape rate: unsupported claims reaching a SUPPORTS/ANSWERABLE verdict
    escapes = sum(
        1 for c, o in zip(cases, outcomes)
        if (c["expect"].get("no_supports") or c["expect"].get("sufficiency_not") == "answerable")
        and not o.passed
    )
    escape_denom = sum(
        1 for c in cases
        if c["expect"].get("no_supports")
        or c["expect"].get("sufficiency_not") == "answerable"
    )
    metrics["unsupported_claim_escape_rate"] = (
        round(escapes / escape_denom, 4) if escape_denom else 0.0
    )
    return metrics


def main() -> None:
    cases = load_cases()
    print(f"v3.1 validation dataset: {len(cases)} cases")
    outcomes = []
    for case in cases:
        o = run_case(case)
        outcomes.append(o)
        mark = "PASS" if o.passed else "FAIL"
        if not o.passed:
            print(f"  [{mark}] {o.category}/{o.case_id}: {o.detail[:120]}")
    metrics = compute_metrics(outcomes, cases)
    print()
    print(f"pass: {metrics['passed']}/{metrics['total']} ({metrics['pass_rate']:.1%})")
    for cat, m in metrics["per_category"].items():
        print(f"  {cat:16} {m['passed']}/{m['total']}")
    print()
    print(f"RELIGIOUS_FALSE_SUPPORT_RATE: {metrics['religious_false_support_rate']:.2%} (target <5%)")
    print(f"UNSUPPORTED_CLAIM_ESCAPE_RATE: {metrics['unsupported_claim_escape_rate']:.2%} (target <5%)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}-v31-metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nmetrics: {out}")
    if metrics["religious_false_support_rate"] > 0.05:
        raise SystemExit(2)
    if metrics["unsupported_claim_escape_rate"] > 0.05:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
