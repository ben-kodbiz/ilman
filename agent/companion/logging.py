"""Companion chat logging (user request: capture all companion chats for
later pipeline troubleshooting/enhancement).

Privacy-first design (fixme_v3 §31 + fixme_v2 §25/§31):
- Local-only: knowledge/processed/companion_logs/*.jsonl — never the repo
  (gitignored), never any network
- Structured JSONL: one record per turn-pair (user message + companion
  reply + full pipeline metadata) — machine-analyzable for troubleshooting
- NO chain-of-thought is ever logged; trace fields are decision metadata
  only (intent/emotion/risk/mode/policy/verdicts/latency), exactly what the
  harness already exposes in DebugTrace
- Crisis turns are marked sensitive: content captured (troubleshooting
  requires it) but flagged so exports/analytics can exclude them
- Per-session opt-out: `logging_enabled=false` state disables capture;
  session drop forgets nothing already written (append-only audit log) but
  a `--redact` export tool replaces content with [REDACTED]

Record shape (one line per user→companion exchange):
{
  "ts": ISO-8601,
  "session_id": str,
  "turn": int,
  "user": {"text": str},
  "companion": {"text": str, "mode": str, "intent": str, "emotion": str|null,
                "risk": str, "route": str},
  "citations": [...], "unsupported_citations": [...],
  "companion_validation": {...},
  "evidence": {"status": str, "sufficiency": float,
               "claim_verdicts": [{claim, verdict, claim_type, citation}...]},
  "policy": {...}, "planned_query": {...},
  "notes": [...], "latency_s": float,
  "sensitive": bool,        # crisis/high-risk turn
  "model": str,
  "schema": 1
}
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "knowledge" / "processed" / "companion_logs"
SCHEMA = 1

# sessions excluded from logging (evaluation runs, test fixtures). Anything
# matching these prefixes never writes a chat record — production/console
# sessions capture; eval noise doesn't.
_EVAL_PREFIXES = (
    "case-", "scen-", "eval-", "dua-test", "pillar-", "v3-", "adv-",
    "fatihah-", "s1", "s2", "s3", "s4", "s5", "s6",  # test fixtures
)
_DISABLED_SESSIONS: set[str] = set()


def disable_session(session_id: str) -> None:
    _DISABLED_SESSIONS.add(session_id)


def is_enabled(session_id: str) -> bool:
    if session_id in _DISABLED_SESSIONS:
        return False
    return not session_id.startswith(_EVAL_PREFIXES)


class CompanionLogger:
    """Append-only JSONL chat logger, one file per session per UTC day."""

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = log_dir

    def _session_file(self, session_id: str, ts: datetime) -> Path:
        # filesystem-safe session id
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:80] or "anon"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / f"{safe}-{ts:%Y%m%d}.jsonl"

    def log_turn(
        self,
        session_id: str,
        turn: int,
        user_text: str,
        reply_text: str,
        result=None,
        model_label: str = "",
    ) -> Path | None:
        """Append one turn record. `result` is the HarnessResult; on early
        exit paths (crisis) it may be None — then minimal fields are kept."""
        if not is_enabled(session_id):
            return None
        ts = datetime.now(UTC)
        path = self._session_file(session_id, ts)

        sensitive = False
        record: dict = {
            "ts": ts.isoformat(),
            "session_id": session_id,
            "turn": turn,
            "user": {"text": user_text},
            "companion": {"text": reply_text},
            "citations": [],
            "unsupported_citations": [],
            "companion_validation": {},
            "evidence": {"status": "", "sufficiency": 0.0, "claim_verdicts": []},
            "policy": {},
            "planned_query": {},
            "notes": [],
            "latency_s": 0.0,
            "sensitive": False,
            "model": model_label,
            "schema": SCHEMA,
        }
        if result is not None:
            trace = result.trace or {}
            state = result.state or {}
            record["companion"].update({
                "mode": state.get("mode", ""),
                "intent": state.get("intent", ""),
                "emotion": state.get("emotion"),
                "risk": state.get("risk", "low"),
                "route": trace.get("route", ""),
            })
            record["citations"] = list(result.citations or [])
            record["unsupported_citations"] = list(result.unsupported_citations or [])
            record["companion_validation"] = result.companion_validation or {}
            record["evidence"] = {
                "status": trace.get("evidence_status", ""),
                "sufficiency": round(float(trace.get("evidence_sufficiency", 0.0)), 3),
                "claim_verdicts": [
                    {
                        "claim": c.get("claim", "")[:160],
                        "verdict": c.get("verdict", ""),
                        "claim_type": c.get("claim_type", ""),
                        "citation": c.get("citation"),
                        "support": c.get("support", 0.0),
                    }
                    for c in (trace.get("validation_trace") or [])[:12]
                ],
            }
            record["policy"] = result.policy or {}
            record["planned_query"] = trace.get("planned_query", {})
            record["notes"] = trace.get("notes", [])
            record["latency_s"] = trace.get("latency_s", 0.0)
            record["model"] = trace.get("model", model_label)
            sensitive = (
                state.get("risk") == "high"
                or record["companion"].get("mode") == "crisis"
            )
        record["sensitive"] = sensitive

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    # ------------------------------------------------------------ reading
    @staticmethod
    def read_sessions(log_dir: Path = LOG_DIR) -> list[str]:
        """Distinct session ids present in logs."""
        sessions: set[str] = set()
        for f in sorted(log_dir.glob("*.jsonl")):
            stem = f.stem
            # <session>-YYYYMMDD
            m = re.match(r"(.*)-\d{8}$", stem)
            sessions.add(m.group(1) if m else stem)
        return sorted(sessions)

    @staticmethod
    def read_turns(session_id: str, log_dir: Path = LOG_DIR,
                   include_sensitive: bool = True) -> list[dict]:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:80] or "anon"
        records: list[dict] = []
        for f in sorted(log_dir.glob(f"{safe}-*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("session_id") != session_id:
                    continue
                if not include_sensitive and rec.get("sensitive"):
                    continue
                records.append(rec)
        records.sort(key=lambda r: (r.get("ts", ""), r.get("turn", 0)))
        return records

    # ----------------------------------------------------------- export
    @staticmethod
    def export_markdown(session_id: str, out_path: Path,
                         log_dir: Path = LOG_DIR,
                         include_sensitive: bool = True,
                         redact: bool = False) -> Path:
        """Human-readable Markdown transcript for later analysis."""
        turns = CompanionLogger.read_turns(
            session_id, log_dir, include_sensitive=include_sensitive
        )
        lines = [
            f"# Companion chat log — session `{session_id}`",
            "",
            f"- exported: {datetime.now(UTC).isoformat()}",
            f"- turns: {len(turns)}",
            f"- sensitive turns: {sum(1 for t in turns if t.get('sensitive'))}"
            + (" (excluded)" if not include_sensitive else ""),
            f"- redaction: {'ON (content replaced)' if redact else 'off'}",
            "",
            "---",
            "",
        ]

        def _clean(text: str) -> str:
            if redact:
                return "[REDACTED]"
            return (text or "").replace("\n", "\n  ")

        for t in turns:
            comp = t.get("companion", {})
            meta = [
                f"mode: **{comp.get('mode', '?')}**",
                f"intent: `{comp.get('intent', '?')}`",
            ]
            if comp.get("emotion"):
                meta.append(f"emotion: `{comp['emotion']}`")
            meta.append(f"risk: `{comp.get('risk', '?')}`")
            if t.get("latency_s"):
                meta.append(f"{t['latency_s']}s")
            if t.get("sensitive"):
                meta.append("**SENSITIVE**")
            lines.append(f"### Turn {t.get('turn', '?')} — {t.get('ts', '')}")
            lines.append("")
            lines.append("> " + " · ".join(meta))
            lines.append("")
            lines.append(f"**User:** {_clean(t.get('user', {}).get('text', ''))}")
            lines.append("")
            lines.append(f"**Ilman:** {_clean(comp.get('text', ''))}")
            if t.get("citations"):
                lines.append("")
                lines.append(f"- citations: {', '.join(t['citations'])}")
            ev = t.get("evidence", {})
            if ev.get("status"):
                lines.append(
                    f"- evidence: {ev['status']}"
                    + (f" ({ev.get('sufficiency')})" if ev.get("sufficiency") else "")
                )
                for cv in ev.get("claim_verdicts", []):
                    mark = {"supports": "✓", "partial": "~", "irrelevant": "✗"}.get(
                        cv.get("verdict"), "?"
                    )
                    lines.append(
                        f"  - {mark} [{cv.get('claim_type', '')}] {cv.get('claim', '')}"
                        f" → {cv.get('citation') or '—'}"
                    )
            notes = t.get("notes") or []
            if notes:
                lines.append(f"- notes: {'; '.join(notes)}")
            lines.append("")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path

    # ------------------------------------------------------- bulk analysis
    @staticmethod
    def stats(log_dir: Path = LOG_DIR) -> dict:
        """Aggregate pipeline stats across all logs — the troubleshooting
        overview (mode distribution, evidence status, verdict rates)."""
        agg = {
            "sessions": len(CompanionLogger.read_sessions(log_dir)),
            "turns": 0,
            "modes": {},
            "intents": {},
            "evidence_status": {},
            "verdicts": {},
            "refusals": 0,
            "cited_turns": 0,
            "sensitive_turns": 0,
            "repair_turns": 0,
            "avg_latency_s": 0.0,
        }
        latencies: list[float] = []
        for f in sorted(log_dir.glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                agg["turns"] += 1
                comp = rec.get("companion", {})
                agg["modes"][comp.get("mode", "?")] = (
                    agg["modes"].get(comp.get("mode", "?"), 0) + 1
                )
                agg["intents"][comp.get("intent", "?")] = (
                    agg["intents"].get(comp.get("intent", "?"), 0) + 1
                )
                st = rec.get("evidence", {}).get("status", "")
                if st:
                    agg["evidence_status"][st] = agg["evidence_status"].get(st, 0) + 1
                for cv in rec.get("evidence", {}).get("claim_verdicts", []):
                    v = cv.get("verdict", "?")
                    agg["verdicts"][v] = agg["verdicts"].get(v, 0) + 1
                if "could not verify" in (comp.get("text") or "").lower():
                    agg["refusals"] += 1
                if rec.get("citations"):
                    agg["cited_turns"] += 1
                if rec.get("sensitive"):
                    agg["sensitive_turns"] += 1
                if any("repair" in str(n).lower() for n in rec.get("notes", [])):
                    agg["repair_turns"] += 1
                if rec.get("latency_s"):
                    latencies.append(float(rec["latency_s"]))
        if latencies:
            agg["avg_latency_s"] = round(sum(latencies) / len(latencies), 2)
        return agg
