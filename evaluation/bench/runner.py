"""Benchmark runner (agentodo.md §19-20, §26 Phase 1).

Runs YAML suites against configured models, times each request, records
usage/latency/GPU memory, and emits JSONL results + a Markdown report at
evaluation/model-benchmark.md.

Only model capability is measured here. This is NOT a knowledge benchmark:
expected answers are either deterministic tasks (math, format, tool calls) or
behavioral requirements from the spec itself (refuse to fabricate, admit
uncertainty). Religious text for Arabic-preservation cases is provided in the
prompt and echoed, never recalled from model memory.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from agent.core.config import load_config
from agent.core.model import ChatMessage, ModelBackend
from evaluation.bench.checkers import run_checks
from evaluation.bench.tools_ref import resolve_tools

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES_DIR = REPO_ROOT / "evaluation" / "suites"
RESULTS_ROOT = REPO_ROOT / "evaluation" / "results"

DEFAULT_SYSTEM = (
    "You are a careful Islamic study assistant. Answer only from material provided in the "
    "prompt or tool results. Never invent Qur'an references, hadith, scholars, or quotations. "
    "If you cannot verify something, say so plainly."
)


def _suite_name_of(path: Path) -> str:
    """Declared `suite:` name inside a suite file, for CLI matching."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("suite:"):
                    return line.split("suite:", 1)[1].strip()
    except OSError:
        pass
    return ""


@dataclass
class CaseResult:
    model: str
    suite: str
    case_id: str
    passed: bool
    checks: list[dict]
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    tokens_per_s: float
    finish_reason: str
    content_excerpt: str = ""
    tool_calls: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def gpu_memory_used_mi() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


class BenchmarkRunner:
    def __init__(self, config_path: Path | None = None, suites_dir: Path = SUITES_DIR):
        self.config_path = config_path
        self.suites_dir = suites_dir
        self.app_config = load_config(config_path)

    def load_suites(self, names: list[str] | None = None) -> list[dict]:
        """Match suites by filename stem or their declared `suite:` name."""
        suites = []
        files = sorted(self.suites_dir.glob("*.yaml"))
        if names:
            wanted = set(names)
            files = [
                f for f in files
                if f.stem in wanted or _suite_name_of(f) in wanted
            ]
        for path in files:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            suites.append({"name": path.stem, **data})
        if not suites:
            raise FileNotFoundError(f"no suites matching {names or 'all'} in {self.suites_dir}")
        return suites

    def run(
        self,
        model_roles: list[str],
        suite_names: list[str] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict:
        backend_name = self.app_config.defaults.get("backend", "")
        backend_cfg = self.app_config.backends[backend_name]
        backend = ModelBackend(backend_cfg)
        suites = self.load_suites(suite_names)

        models = {}
        for role in model_roles:
            models[role] = backend_cfg.models.get(role) or role

        results: list[CaseResult] = []
        started = datetime.now(UTC).isoformat()
        env_gpu = gpu_memory_used_mi()

        for role, model_id in models.items():
            label = f"{role}:{model_id}"
            print(f"\n=== {label} ===")
            warmup = self._warmup(backend, model_id)
            if warmup is not None:
                print(f"  [unavailable] {warmup}")
                results.append(CaseResult(
                    model=label, suite="warmup", case_id="warmup",
                    passed=False, checks=[{"type": "availability", "passed": False, "detail": warmup}],
                    latency_s=0, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
                    tokens_per_s=0, finish_reason="unavailable",
                ))
                continue
            for suite in suites:
                suite_cases = suite.get("cases", [])
                for case in suite_cases:
                    res = self._run_case(backend, model_id, label, suite["name"], case,
                                        max_tokens, temperature)
                    results.append(res)
                    mark = "PASS" if res.passed else "FAIL"
                    print(f"  [{mark}] {suite['name']}/{res.case_id} ({res.latency_s:.1f}s)")
        ended = datetime.now(UTC).isoformat()
        return self._summary(started, ended, backend_name, models, results, env_gpu, max_tokens, temperature)

    def _warmup(self, backend: ModelBackend, model_id: str) -> str | None:
        try:
            resp = backend.chat(
                model_id,
                [ChatMessage(role="user", content="Say OK")],
                max_tokens=100, temperature=0.0,
            )
            if not resp.content and not resp.reasoning and not resp.tool_calls:
                return f"empty response (finish={resp.finish_reason})"
            return None
        except Exception as e:
            return str(e)[:200]

    def _run_case(
        self, backend: ModelBackend, model_id: str, label: str,
        suite_name: str, case: dict, max_tokens: int, temperature: float,
    ) -> CaseResult:
        messages = [ChatMessage(role="system", content=case.get("system", DEFAULT_SYSTEM))]
        tools = resolve_tools(case.get("tools_ref"))
        expects = case.get("expect", [])
        for turn in case.get("messages", [{"role": "user", "content": case.get("prompt", "")}]):
            if turn.get("content"):
                messages.append(ChatMessage(role=turn["role"], content=turn["content"]))
        content = ""
        tool_calls_out: list[dict] = []
        checks: list[dict] = []
        finish = ""
        latency = prompt_toks = comp_toks = reason_toks = 0
        tps = 0.0
        try:
            resp = backend.chat(
                model_id, messages,
                tools=tools or None,
                max_tokens=case.get("max_tokens", max_tokens),
                temperature=case.get("temperature", temperature),
            )
            latency = resp.latency_s
            prompt_toks = resp.usage.get("prompt_tokens", 0)
            comp_toks = resp.usage.get("completion_tokens", 0)
            reason_toks = resp.usage.get("reasoning_tokens", 0)
            finish = resp.finish_reason
            gen = max(comp_toks - reason_toks, 1)
            tps = gen / latency if latency > 0 else 0.0
            content = resp.content
            tool_calls_out = [{"name": tc.name, "arguments": tc.arguments} for tc in resp.tool_calls]
            checks = run_checks(resp, expects)
        except Exception as e:
            checks = [{"type": "request", "passed": False, "detail": str(e)[:300]}]
        passed = bool(checks) and all(c["passed"] for c in checks)
        excerpt = re.sub(r"\s+", " ", content)[:300]
        return CaseResult(
            model=label, suite=suite_name, case_id=case["id"],
            passed=passed, checks=checks, latency_s=latency,
            prompt_tokens=prompt_toks, completion_tokens=comp_toks, reasoning_tokens=reason_toks,
            tokens_per_s=tps, finish_reason=finish,
            content_excerpt=excerpt, tool_calls=tool_calls_out,
        )

    def _summary(self, started, ended, backend_name, models, results, env_gpu, max_tokens, temperature) -> dict:
        by_model: dict[str, list[CaseResult]] = {}
        for r in results:
            by_model.setdefault(r.model, []).append(r)
        model_summaries = {}
        for label, rs in by_model.items():
            real = [r for r in rs if r.suite != "warmup"]
            lat = [r.latency_s for r in real if r.latency_s > 0]
            tps = [r.tokens_per_s for r in real if r.tokens_per_s > 0]
            model_summaries[label] = {
                "cases": len(real),
                "passed": sum(r.passed for r in real),
                "pass_rate": round(sum(r.passed for r in real) / len(real), 3) if real else 0,
                "avg_latency_s": round(statistics.mean(lat), 2) if lat else 0,
                "p95_latency_s": round(self._p95(lat), 2) if lat else 0,
                "avg_tokens_per_s": round(statistics.mean(tps), 1) if tps else 0,
                "reasoning_tokens_total": sum(r.reasoning_tokens for r in real),
            }
        return {
            "started_at": started,
            "ended_at": ended,
            "backend": backend_name,
            "models": models,
            "params": {"max_tokens": max_tokens, "temperature": temperature},
            "gpu_memory_used_mi_before": env_gpu,
            "gpu_memory_used_mi_after": gpu_memory_used_mi(),
            "models_summary": model_summaries,
            "results": [r.to_dict() for r in results],
        }

    @staticmethod
    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = max(0, round(0.95 * len(s)) - 1)
        return s[idx]


def write_outputs(summary: dict, out_dir: Path = RESULTS_ROOT) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = out_dir / f"{stamp}-results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in summary["results"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md_path = out_dir / f"{stamp}-report.md"
    md_path.write_text(render_report(summary), encoding="utf-8")
    return jsonl_path, md_path


def render_report(summary: dict) -> str:
    lines = [
        "# Model Benchmark — raw run output",
        "",
        f"- Backend: `{summary['backend']}`",
        f"- Started: {summary['started_at']}  Ended: {summary['ended_at']}",
        f"- Params: `{summary['params']}`",
        f"- GPU memory used before/after: "
        f"{summary['gpu_memory_used_mi_before']} / {summary['gpu_memory_used_mi_after']} MiB",
        "",
        "## Per-model summary",
        "",
        "| model | cases | passed | pass rate | avg latency s | p95 s | gen tok/s | reasoning tok |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, s in summary["models_summary"].items():
        lines.append(
            f"| {label} | {s['cases']} | {s['passed']} | {s['pass_rate']:.1%} | "
            f"{s['avg_latency_s']} | {s['p95_latency_s']} | {s['avg_tokens_per_s']} | "
            f"{s['reasoning_tokens_total']} |"
        )
    lines += ["", "## Suite breakdown", "", "| model | suite | passed/cases | pass rate |", "|---|---|---|---|"]
    seen: dict[tuple[str, str], list[int]] = {}
    for r in summary["results"]:
        if r["suite"] == "warmup":
            continue
        key = (r["model"], r["suite"])
        seen.setdefault(key, [0, 0])
        seen[key][1] += 1
        if r["passed"]:
            seen[key][0] += 1
    for (model, suite), (passed, total) in sorted(seen.items()):
        lines.append(f"| {model} | {suite} | {passed}/{total} | {passed/total:.1%} |")
    lines += ["", "## Failures", ""]
    fails = [r for r in summary["results"] if not r["passed"]]
    if not fails:
        lines.append("None.")
    for r in fails:
        lines.append(f"### {r['model']} — {r['suite']}/{r['case_id']}")
        lines.append(f"- checks: `{json.dumps(r['checks'], ensure_ascii=False)}`")
        lines.append(f"- finish: `{r['finish_reason']}`, latency {r['latency_s']:.1f}s")
        if r["content_excerpt"]:
            lines.append(f"- excerpt: {r['content_excerpt']!r}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run model benchmark suites")
    parser.add_argument("--models", nargs="+", default=["ling_tiny", "gemma_qat", "qwen_medium"],
                        help="model roles from configs/config.yaml")
    parser.add_argument("--suites", nargs="+", default=None, help="suite names (default: all)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    runner = BenchmarkRunner()
    summary = runner.run(args.models, args.suites, args.max_tokens, args.temperature)
    jsonl_path, md_path = write_outputs(summary)
    print(f"\nResults: {jsonl_path}")
    print(f"Report:  {md_path}")


if __name__ == "__main__":
    main()
