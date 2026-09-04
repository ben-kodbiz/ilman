"""Model-swap regression CLI (agentodo.md §19, §26 Phase 5).

Runs the grounded suite per model, swapping LM Studio models via its REST
API (only ~12GB VRAM: never two large models at once). Swapping is best
effort: if a model cannot be loaded, its section reports unavailability and
the run continues with the next model.

Usage:
  uv run python -m scripts.run_regression --models ling_tiny gemma_e4b
  uv run python -m scripts.run_regression --models gemma_e4b --no-swap
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.core.agent import AgentOrchestrator  # noqa: E402
from agent.core.config import load_config  # noqa: E402
from agent.core.model import ModelRouter  # noqa: E402
from agent.memory.store import MemoryStore  # noqa: E402
from agent.policy.source_policy import SourcePolicy, SourceRegistry  # noqa: E402
from agent.tools.layer import ToolLayer  # noqa: E402
from evaluation.bench.grounded_runner import (  # noqa: E402
    load_suites,
    run_model,
    write_jsonl,
    write_report,
)
from ingestion.hadith_ingest import HadithStore  # noqa: E402
from ingestion.quran_ingest import QuranStore  # noqa: E402
from ingestion.tafsir_en_ingest import TafsirEnStore  # noqa: E402
from ingestion.tafsir_ingest import TafsirStore  # noqa: E402
from retrieval.hybrid import RetrievalOrchestrator  # noqa: E402

LMSTUDIO = "http://127.0.0.1:1234"


def lms_loaded_models() -> set[str]:
    try:
        r = requests.get(f"{LMSTUDIO}/api/v0/models", timeout=5)
        return {m["id"] for m in r.json()["data"] if m.get("state") == "loaded"}
    except Exception:
        return set()


def lms_unload(model_id: str) -> None:
    try:
        requests.post(
            f"{LMSTUDIO}/api/v1/models/unload",
            json={"instance_id": model_id}, timeout=10,
        )
        time.sleep(2)
    except Exception as e:
        print(f"  [warn] unload {model_id} failed: {e}")


def lms_ensure_loaded(model_id: str, timeout_s: int = 600) -> bool:
    """Ensure exactly this LLM is loaded (JIT-load via a chat call)."""
    loaded = lms_loaded_models()
    llms_loaded = {m for m in loaded if m != "text-embedding-nomic-embed-text-v1.5"}
    if model_id in llms_loaded and len(llms_loaded) == 1:
        return True
    for other in llms_loaded:
        if other != model_id:
            print(f"  unloading {other} ...")
            lms_unload(other)
    try:
        r = requests.post(
            f"{LMSTUDIO}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 50,
            },
            timeout=timeout_s,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"  [warn] load {model_id} failed: {e}")
        return False


def build_agent(role: str) -> AgentOrchestrator:
    """Build the full agent with routing overridden to `role` for every task."""
    cfg = load_config()
    # Override every routing class to the requested role (identical retrieval
    # + evidence across models — §19: "Use identical retrieval and evidence packs")
    for key in list(cfg.routing):
        cfg.routing[key] = role
    router = ModelRouter(cfg)
    store = QuranStore()
    hadith_store = HadithStore()
    policy = SourcePolicy(SourceRegistry.load())
    memory = MemoryStore(db_path=REPO_ROOT / "knowledge" / "processed" / "eval_memory.db")
    tools = ToolLayer(policy, store=store, hadith_store=hadith_store, memory=memory)
    ro = RetrievalOrchestrator(
        store, hadith_store=hadith_store,
        tafsir_store=TafsirStore(), tafsir_en_store=TafsirEnStore(),
    )
    try:
        from retrieval.vector_store import VectorStore

        vs = VectorStore()
        if vs.size:
            ro.vector_store = vs
            print(f"  vector index: {vs.size} texts")
    except Exception as e:
        print(f"  [warn] vector leg unavailable: {e}")
    return AgentOrchestrator(router, ro, tools, memory=memory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounded regression across models")
    parser.add_argument("--models", nargs="+", default=["ling_tiny"],
                        help="model roles from configs/config.yaml")
    parser.add_argument("--suites", nargs="+", default=None)
    parser.add_argument("--no-swap", action="store_true",
                        help="do not manage LM Studio load/unload")
    args = parser.parse_args()

    cfg = load_config()
    backend_name = cfg.defaults.get("backend", "lmstudio")
    suites = load_suites(args.suites)
    print(f"suites: {[s['name'] for s in suites]}")

    summaries = []
    for role in args.models:
        model_id = cfg.backends[backend_name].models.get(role, role)
        print(f"\n=== {role}:{model_id} ===")
        if not args.no_swap:
            ok = lms_ensure_loaded(model_id)
            if not ok:
                print(f"  [unavailable] could not load {model_id}; skipping")
                continue
        agent = build_agent(role)
        summary = run_model(agent, suites, f"{role}:{model_id}")
        summaries.append(summary)
        print(
            f"  => {summary['passed']}/{summary['cases']} passed "
            f"({summary['pass_rate']:.0%}), halluc {summary['hallucination_rate']:.0%}, "
            f"refusal {summary['refusal_rate']:.0%}, avg {summary['avg_latency_s']}s"
        )

    if not summaries:
        print("no models evaluated")
        return
    md = write_report(summaries)
    jl = write_jsonl(summaries)
    print(f"\nReport: {md}")
    print(f"Raw:    {jl}")


if __name__ == "__main__":
    main()
