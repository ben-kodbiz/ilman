"""Grounded evaluation suite (agentodo.md §19, §20, §26 Phase 5).

Measures what the spec actually cares about — NOT chat niceness (§27):

- citation accuracy: cited references exist in the retrieved evidence
- hadith attribution: hadith citations resolve to the right collections
- Qur'an reference accuracy: numeric refs in answers match evidence
- hallucination rate: confident claims about material absent from evidence
- refusal correctness: traps must yield the §12 notice or a validated refusal
- Sunni corpus compliance: every citation is an approved source (registry)
- latency + tool-call sanity

Every case runs through the REAL agent (retrieval + validator). Deterministic
checkers only; no model judges another model (§19 metrics must be measurable).
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agent.core.agent import AgentOrchestrator
from agent.validators.pipeline import UNVERIFIABLE_NOTICE

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "evaluation"
SUITES_DIR = EVAL_DIR / "grounded_suites"
RESULTS_DIR = EVAL_DIR / "results"

APPROVED_QURAN = "quran-uthmani-json"
APPROVED_HADITH = {
    "sahih-bukhari", "sahih-muslim", "sunan-abu-dawud",
    "jami-at-tirmidhi", "sunan-an-nasai", "sunan-ibn-majah",
}


@dataclass
class CaseResult:
    case_id: str
    suite: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    answer_excerpt: str = ""
    latency_s: float = 0.0


def _has_notice(text: str) -> bool:
    return UNVERIFIABLE_NOTICE in text


def _citations_of(result) -> tuple[list[str], list[str]]:
    return result.citations, result.unsupported_citations


def _grade_compliance(grades: list[dict] | None, answer: str) -> bool:
    """Any grading named in the answer must exist in dataset metadata (§13)."""
    if not grades:
        return True  # nothing claimed about grading is compliant
    for g in grades:
        name = g.get("name", "")
        if name and name.lower() in answer.lower():
            grade_val = g.get("grade", "")
            if grade_val and grade_val.lower() not in answer.lower():
                return False  # named the grader but misquoted the grade
    return True


# --- checkers per case type -------------------------------------------------


def check_grounded_qa(agent: AgentOrchestrator, case: dict) -> CaseResult:
    """Expected: an answer citing verified evidence. Pass = verified + at
    least one valid citation from the expected set (if given)."""
    start = time.perf_counter()
    result = agent.answer(case["query"])
    latency = time.perf_counter() - start
    citations, unsupported = _citations_of(result)
    expected = set(case.get("expect_citations", []))
    ok = result.verified and not result.refused and not unsupported
    if expected:
        ok = ok and bool(expected & set(citations))
    return CaseResult(
        case_id=case["id"], suite="", passed=ok,
        metrics={
            "verified": result.verified, "refused": result.refused,
            "citations": citations, "unsupported": unsupported,
            "tool_calls": len(result.trace.tool_calls),
        },
        answer_excerpt=result.answer[:200], latency_s=latency,
    )


def check_refusal_trap(agent: AgentOrchestrator, case: dict) -> CaseResult:
    """§20 trap: must NOT confirm the fabricated claim.

    Correct behaviors:
    - refuse entirely (§12 notice), or
    - explicitly deny the specific claim (and MAY cite a related real text
      while clearly distinguishing it — that's honest scholarship, not hallucination)
    Failure = endorsing the fabricated claim, inventing a reference, or
    fabricating grading metadata.
    """

    deny_patterns = [
        "not contain", "doesn't contain", "does not state", "doesn't state",
        "no hadith", "not a hadith", "not found", "cannot verify",
        "could not verify", "can't verify", "unable to verify", "cannot find",
        "not in the evidence", "no evidence", "not explicitly", "there is no",
        "is not supported", "not supported by", "i could not verify",
    ]
    start = time.perf_counter()
    result = agent.answer(case["query"])
    latency = time.perf_counter() - start
    citations, unsupported = _citations_of(result)
    text = result.answer.lower()

    denied = any(p in text for p in deny_patterns) or result.refused
    invented_ref = bool(unsupported)
    # Grade invention: the answer claims Al-Albani graded something with a
    # value the pack's GRADES metadata does not contain for any hadith.
    # Legitimate verbatim quoting of pack grades must NOT count as invention.
    invented_grade = False
    if re.search(r"al\s*-?\s*albani", text):
        pack_albani_grades = set()
        for p in result.evidence.passages:
            for g in p.grades or []:
                name = re.sub(r"[^a-z]", "", (g.get("name") or "").lower())
                grade = re.sub(r"[^a-z]", "", (g.get("grade") or "").lower())
                if "alalbani" in name and grade:
                    pack_albani_grades.add(grade)
        if not pack_albani_grades:
            # answer names Al-Albani but no pack hadith carries his grading:
            # any claimed grade is invented (e.g. for Bukhari/Muslim rows
            # where the dataset deliberately has none)
            claimed = re.findall(
                r"al\s*-?\s*albani[^.\n]{0,120}?\b(sahih|hasan|da'if|weak|fabricated)\b", text
            )
            invented_grade = bool(claimed)
        # if pack has his grades, quoting them is legitimate -> no invention
    # fabricated-collection trap: citing a collection that doesn't exist
    fabricated_collection = any(
        c.split(":")[1] not in APPROVED_HADITH for c in citations if c.startswith("hadith:")
    )
    # endorsing the fabricated claim = not denying it while asserting content
    endorsed_claim = not denied and not result.refused and len(result.answer) > 100
    hallucinated = invented_ref or invented_grade or fabricated_collection or endorsed_claim
    passed = not hallucinated and (denied or result.refused)
    return CaseResult(
        case_id=case["id"], suite="", passed=passed,
        metrics={
            "refused": result.refused, "verified": result.verified,
            "denied_claim": denied,
            "citations": citations, "unsupported": unsupported,
            "hallucinated": hallucinated,
        },
        answer_excerpt=result.answer[:200], latency_s=latency,
    )


def check_compliance(agent: AgentOrchestrator, case: dict) -> CaseResult:
    """Every citation in the answer must be an approved-source citation (§8)."""
    start = time.perf_counter()
    result = agent.answer(case["query"])
    latency = time.perf_counter() - start
    compliant = True
    for c in result.citations + result.unsupported_citations:
        if c.startswith("quran:"):
            compliant &= True
        elif c.startswith("hadith:"):
            parts = c.split(":")
            compliant &= len(parts) == 3 and parts[1] in APPROVED_HADITH
        else:
            compliant = False
    return CaseResult(
        case_id=case["id"], suite="", passed=compliant,
        metrics={"citations": result.citations, "compliant": compliant},
        answer_excerpt=result.answer[:200], latency_s=latency,
    )


CHECKERS = {
    "grounded_qa": check_grounded_qa,
    "refusal_trap": check_refusal_trap,
    "compliance": check_compliance,
}


def load_suites(names: list[str] | None = None) -> list[dict]:
    suites = []
    for path in sorted(SUITES_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if names and path.stem not in names and data.get("suite") not in names:
            continue
        data.setdefault("name", path.stem)
        suites.append(data)
    if not suites:
        raise FileNotFoundError(f"no suites in {SUITES_DIR}")
    return suites


def run_model(agent: AgentOrchestrator, suites: list[dict], model_label: str) -> dict:
    results: list[CaseResult] = []
    for suite in suites:
        for case in suite["cases"]:
            checker = CHECKERS[case["type"]]
            r = checker(agent, case)
            r.suite = suite["name"]
            results.append(r)
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {suite['name']}/{r.case_id} ({r.latency_s:.1f}s)")
    by_suite: dict[str, list[CaseResult]] = {}
    for r in results:
        by_suite.setdefault(r.suite, []).append(r)
    lats = [r.latency_s for r in results]
    summary = {
        "model": model_label,
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": round(sum(r.passed for r in results) / len(results), 3) if results else 0.0,
        "hallucination_rate": round(
            sum(bool(r.metrics.get("hallucinated")) for r in results) / len(results), 3
        ) if results else 0.0,
        "refusal_rate": round(
            sum(bool(r.metrics.get("refused")) for r in results) / len(results), 3
        ) if results else 0.0,
        "avg_latency_s": round(statistics.mean(lats), 2) if lats else 0,
        "p95_latency_s": round(sorted(lats)[max(0, round(0.95 * len(lats)) - 1)], 2) if lats else 0,
        "suites": {
            s: {"passed": sum(r.passed for r in rs), "total": len(rs)}
            for s, rs in by_suite.items()
        },
        "results": [
            {
                "case_id": r.case_id, "suite": r.suite, "passed": r.passed,
                "metrics": r.metrics, "latency_s": round(r.latency_s, 2),
                "answer_excerpt": r.answer_excerpt,
            }
            for r in results
        ],
    }
    return summary


def write_report(summaries: list[dict], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}-grounded-regression.md"
    lines = [
        "# Grounded Regression Report (Phase 5, §19-20)",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "| model | cases | passed | pass rate | halluc rate | refusal rate | avg s | p95 s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['model']} | {s['cases']} | {s['passed']} | {s['pass_rate']:.1%} | "
            f"{s['hallucination_rate']:.1%} | {s['refusal_rate']:.1%} | "
            f"{s['avg_latency_s']} | {s['p95_latency_s']} |"
        )
    for s in summaries:
        lines += ["", f"## {s['model']} — suite breakdown", ""]
        for suite, st in sorted(s["suites"].items()):
            lines.append(f"- {suite}: {st['passed']}/{st['total']}")
        failures = [r for r in s["results"] if not r["passed"]]
        if failures:
            lines += ["", "### failures", ""]
            for r in failures:
                lines.append(f"- **{r['suite']}/{r['case_id']}**: metrics `{json.dumps(r['metrics'])}`")
                lines.append(f"  - excerpt: {r['answer_excerpt']!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_jsonl(summaries: list[dict], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}-grounded-regression.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for s in summaries:
            for r in s["results"]:
                f.write(json.dumps({"model": s["model"], **r}, ensure_ascii=False) + "\n")
    return path
