"""Answer ratings (owner request): thumbs up/down per companion answer, in
both the Gradio console and the PWA, stored for later pipeline analysis —
especially down-rated answers (troubleshooting signal).

Privacy rules follow the chat-log policy (§31): local-only JSONL under
knowledge/processed/companion_logs/, gitignored, no chain-of-thought; only
the pipeline decision metadata the harness already exposes. A rating links
to its source turn via (session_id, turn) — the chat log's primary key —
plus a frozen snapshot of the pipeline metadata so analysis never needs to
re-join the chat logs.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RATINGS_PATH = REPO_ROOT / "knowledge" / "processed" / "companion_logs" / "ratings.jsonl"

VALID_RATINGS = {"up", "down"}


class RatingError(ValueError):
    pass


def _sanitize_session(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", (session_id or "")[:80]) or "anon"


def record_rating(session_id: str, turn: int, rating: str,
                  answer_excerpt: str = "", pipeline: dict | None = None) -> Path:
    """Append one rating record. Idempotent per (session, turn): a re-rate
    (e.g. thumb down after thumb up) appends a superseding record; analysis
    reads only the latest per turn."""
    if rating not in VALID_RATINGS:
        raise RatingError(f"rating must be 'up' or 'down', got {rating!r}")
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "session_id": session_id,
        "turn": int(turn),
        "rating": rating,
        "answer_excerpt": (answer_excerpt or "")[:200],
        "pipeline": pipeline or {},
        "schema": 1,
    }
    RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RATINGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return RATINGS_PATH


def read_ratings() -> list[dict]:
    if not RATINGS_PATH.exists():
        return []
    out = []
    for line in RATINGS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def latest_ratings() -> dict[tuple[str, int], dict]:
    """(session_id, turn) -> the most recent rating record for that turn."""
    latest: dict[tuple[str, int], dict] = {}
    for rec in read_ratings():
        key = (rec.get("session_id", ""), int(rec.get("turn", 0)))
        prev = latest.get(key)
        if prev is None or rec.get("ts", "") >= prev.get("ts", ""):
            latest[key] = rec
    return latest


def ratings_analysis() -> dict:
    """Aggregate for pipeline troubleshooting — down-rated turns are the
    interesting signal (owner's explicit purpose)."""
    ratings = latest_ratings()
    all_recs = list(ratings.values())
    up = [r for r in all_recs if r["rating"] == "up"]
    down = [r for r in all_recs if r["rating"] == "down"]

    def _bucket(recs: list[dict], field: str, sub: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in recs:
            pipe = r.get("pipeline") or {}
            val = (pipe.get(field) or {}).get(sub) if isinstance(pipe.get(field), dict) \
                else pipe.get(field)
            if isinstance(val, dict):
                val = val.get(sub)
            key = str(val or "—")[:40]
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    analysis = {
        "total_rated_turns": len(all_recs),
        "up": len(up),
        "down": len(down),
        "down_rate": round(len(down) / len(all_recs), 3) if all_recs else 0.0,
        "down_by_intent": _bucket(down, "intent", ""),
        "down_by_mode": _bucket(down, "mode", ""),
        "down_by_evidence_status": _bucket(down, "evidence_status", ""),
        "down_turns": [
            {
                "session_id": r["session_id"], "turn": r["turn"],
                "ts": r["ts"], "answer_excerpt": r.get("answer_excerpt", "")[:150],
                "intent": (r.get("pipeline") or {}).get("intent", ""),
                "mode": (r.get("pipeline") or {}).get("mode", ""),
                "evidence_status": (r.get("pipeline") or {}).get("evidence_status", ""),
                "citations": (r.get("pipeline") or {}).get("citations", []),
            }
            for r in down
        ],
    }
    return analysis


def enrich_from_chat_log(session_id: str, turn: int, pipeline: dict | None = None,
                        log_dir: Path | None = None) -> dict:
    """Build the pipeline snapshot for a rating from the chat log turn (the
    ratings API calls this so the stored record is self-contained)."""
    from agent.companion.logging import LOG_DIR as _DEFAULT_LOG_DIR
    from agent.companion.logging import CompanionLogger

    turns = CompanionLogger.read_turns(session_id, log_dir or _DEFAULT_LOG_DIR)
    snap: dict = dict(pipeline or {})
    for t in turns:
        if int(t.get("turn", -1)) == int(turn):
            comp = t.get("companion", {})
            snap = {
                "intent": comp.get("intent", ""),
                "emotion": comp.get("emotion"),
                "mode": comp.get("mode", ""),
                "risk": comp.get("risk", ""),
                "route": comp.get("route", ""),
                "citations": t.get("citations", []),
                "evidence_status": t.get("evidence", {}).get("status", ""),
                "claim_verdicts": [
                    {"verdict": c.get("verdict"), "claim_type": c.get("claim_type")}
                    for c in t.get("evidence", {}).get("claim_verdicts", [])
                ],
                "latency_s": t.get("latency_s", 0.0),
            }
            break
    return snap
