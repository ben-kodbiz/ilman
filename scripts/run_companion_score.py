"""Companion score CLI (fixme_v2 §28, §44).

  uv run python -m scripts.run_companion_score --models ling_tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.companion.memory import CompanionMemory  # noqa: E402
from agent.context.builder import ContextBuilder  # noqa: E402
from agent.core.config import load_config  # noqa: E402
from agent.core.harness import CompanionHarness  # noqa: E402
from agent.core.model import ModelRouter  # noqa: E402
from agent.memory.router import MemoryRouter  # noqa: E402
from agent.policy.companion_policy import CompanionPolicyEngine  # noqa: E402
from agent.policy.source_policy import SourcePolicy, SourceRegistry  # noqa: E402
from agent.state.manager import StateManager  # noqa: E402
from agent.tools.layer import ToolLayer  # noqa: E402
from agent.validators.companion_validator import ResponseValidator  # noqa: E402
from agent.validators.pipeline import CitationValidator  # noqa: E402
from evaluation.bench.companion_score import run_all, write_score_report  # noqa: E402
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


def build_harness(role: str) -> CompanionHarness:
    cfg = load_config()
    for key in list(cfg.routing):
        cfg.routing[key] = role
    router = ModelRouter(cfg)
    store = QuranStore()
    hadith = HadithStore()
    policy = SourcePolicy(SourceRegistry.load())
    memory = MemoryRouter(
        CompanionMemory(db_path=REPO_ROOT / "knowledge" / "processed" / "eval_score_mem.db")
    )
    ToolLayer(policy, store=store, hadith_store=hadith, memory=memory.memory)
    ro = RetrievalOrchestrator(
        store, hadith_store=hadith, tafsir_store=TafsirStore(),
        tafsir_en_store=TafsirEnStore(),
        vector_store=(VectorStore() if VectorStore else None),
    )
    return CompanionHarness(
        router, retrieval=ro, memory_router=memory,
        states=StateManager(), policy_engine=CompanionPolicyEngine(),
        context_builder=ContextBuilder(), validator=ResponseValidator(),
        citation_validator=CitationValidator(),
        model_label=f"{role}:{cfg.backends[cfg.defaults['backend']].models.get(role, role)}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="fixme_v2 §28 companion score")
    parser.add_argument("--models", nargs="+", default=["ling_tiny"])
    parser.add_argument("--no-swap", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    backend = cfg.defaults.get("backend", "lmstudio")
    for role in args.models:
        model_id = cfg.backends[backend].models.get(role, role)
        print(f"\n===== {role}:{model_id} =====")
        if not args.no_swap and not lms_ensure_loaded(model_id):
            print(f"  [unavailable] {model_id}; skipping")
            continue
        harness = build_harness(role)
        summary = run_all(harness, f"{role}:{model_id}")
        report = write_score_report(
            summary, REPO_ROOT / "evaluation" / "results"
        )
        s = summary["score"]
        print(
            f"  => companion score {s['score']}/100 | "
            f"cases {s['cases_passed']}/{s['cases_total']} | "
            f"scenarios {s['scenarios_passed']}/{s['scenarios_total']}"
        )
        print(f"  report: {report}")


if __name__ == "__main__":
    main()
