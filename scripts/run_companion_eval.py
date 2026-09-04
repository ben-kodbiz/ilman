"""Companion evaluation CLI (fix_me.md §19-21).

Runs the companion suite against a model. Identical engine + checkers across
models; model-swap via LM Studio REST identical to run_regression.

Usage:
  uv run python -m scripts.run_companion_eval --models ling_tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.companion.engine import CompanionEngine  # noqa: E402
from agent.companion.memory import CompanionMemory  # noqa: E402
from agent.companion.state import StateManager  # noqa: E402
from agent.core.config import load_config  # noqa: E402
from agent.core.model import ModelRouter  # noqa: E402
from agent.policy.source_policy import SourcePolicy, SourceRegistry  # noqa: E402
from agent.tools.layer import ToolLayer  # noqa: E402
from evaluation.bench.companion_runner import (  # noqa: E402
    load_suites,
    run_model,
    write_report,
)
from ingestion.hadith_ingest import HadithStore  # noqa: E402
from ingestion.quran_ingest import QuranStore  # noqa: E402
from ingestion.tafsir_en_ingest import TafsirEnStore  # noqa: E402
from ingestion.tafsir_ingest import TafsirStore  # noqa: E402
from retrieval.hybrid import RetrievalOrchestrator  # noqa: E402

try:
    from retrieval.vector_store import VectorStore  # noqa: E402
except Exception:
    VectorStore = None

from scripts.run_regression import lms_ensure_loaded  # noqa: E402


def build_engine(role: str) -> CompanionEngine:
    cfg = load_config()
    for key in list(cfg.routing):
        cfg.routing[key] = role
    router = ModelRouter(cfg)
    store = QuranStore()
    hadith = HadithStore()
    policy = SourcePolicy(SourceRegistry.load())
    memory = CompanionMemory(db_path=REPO_ROOT / "knowledge" / "processed" / "eval_companion_mem.db")
    tools = ToolLayer(policy, store=store, hadith_store=hadith, memory=memory)
    ro = RetrievalOrchestrator(
        store, hadith_store=hadith,
        tafsir_store=TafsirStore(), tafsir_en_store=TafsirEnStore(),
        vector_store=(VectorStore() if VectorStore else None),
    )
    return CompanionEngine(router, ro, tools, memory=memory,
                          state_manager=StateManager())


def main() -> None:
    parser = argparse.ArgumentParser(description="Companion evaluation")
    parser.add_argument("--models", nargs="+", default=["ling_tiny"])
    parser.add_argument("--suites", nargs="+", default=None)
    parser.add_argument("--no-swap", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    backend = cfg.defaults.get("backend", "lmstudio")
    suites = load_suites(args.suites)
    print(f"suites: {[s['name'] for s in suites]}")

    for role in args.models:
        model_id = cfg.backends[backend].models.get(role, role)
        print(f"\n=== {role}:{model_id} ===")
        if not args.no_swap:
            if not lms_ensure_loaded(model_id):
                print(f"  [unavailable] {model_id}; skipping")
                continue
        engine = build_engine(role)
        summary = run_model(engine, suites, f"{role}:{model_id}")
        md = write_report(summary)
        crisis = "OK" if summary["crisis_routing_ok"] else "!!! FAILURES"
        print(
            f"  => {summary['passed']}/{summary['cases']} passed "
            f"({summary['pass_rate']:.0%}) | crisis routing: {crisis} | "
            f"avg {summary['avg_latency_s']}s"
        )
        print(f"  report: {md}")


if __name__ == "__main__":
    main()
