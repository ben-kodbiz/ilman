"""Companion log analysis/export CLI.

    uv run python -m scripts.companion_logs stats|ratings
    uv run python -m scripts.companion_logs stats
    uv run python -m scripts.companion_logs sessions
    uv run python -m scripts.companion_logs read <session_id> [--no-sensitive]
    uv run python -m scripts.companion_logs export <session_id> [-o out.md] [--redact]
    uv run python -m scripts.companion_logs export-all [-o logs_export/]
    uv run python -m scripts.companion_logs watch            # live tail
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.companion.logging import LOG_DIR, CompanionLogger  # noqa: E402


def _print_json(obj: dict) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def cmd_stats(_args) -> None:
    _print_json(CompanionLogger.stats())


def cmd_sessions(_args) -> None:
    sessions = CompanionLogger.read_sessions()
    print(f"{len(sessions)} sessions in {LOG_DIR}:")
    for s in sessions:
        turns = CompanionLogger.read_turns(s)
        sensitive = sum(1 for t in turns if t.get("sensitive"))
        print(f"  {s:40} {len(turns):4} turns" + (f" ({sensitive} sensitive)" if sensitive else ""))


def cmd_read(args) -> None:
    turns = CompanionLogger.read_turns(
        args.session, include_sensitive=not args.no_sensitive
    )
    for t in turns:
        _print_json(t)


def cmd_export(args) -> None:
    out = Path(args.out) if args.out else Path("companion_logs") / f"{args.session}.md"
    path = CompanionLogger.export_markdown(
        args.session, out,
        include_sensitive=not args.no_sensitive,
        redact=args.redact,
    )
    print(f"exported: {path}")


def cmd_export_all(args) -> None:
    out_dir = Path(args.out) if args.out else Path("companion_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for session in CompanionLogger.read_sessions():
        path = CompanionLogger.export_markdown(
            session, out_dir / f"{session}.md",
            include_sensitive=not args.no_sensitive,
            redact=args.redact,
        )
        print(f"exported: {path}")


def cmd_ratings(_args) -> None:
    """Rating aggregate: down-rated turns are the pipeline-enhancement
    signal (owner's stated purpose)."""
    from agent.companion.ratings import ratings_analysis

    a = ratings_analysis()
    print(json.dumps(a, indent=2, ensure_ascii=False))


def cmd_watch(_args) -> None:
    """Live tail: print new turn records as they are appended."""
    known_files: set[Path] = set()
    offsets: dict[Path, int] = {}
    print(f"watching {LOG_DIR} (ctrl-c to stop)…")
    try:
        while True:
            for f in sorted(LOG_DIR.glob("*.jsonl")):
                if f not in known_files:
                    known_files.add(f)
                    offsets[f] = 0
                size = f.stat().st_size
                if size > offsets[f]:
                    with open(f, encoding="utf-8") as fh:
                        fh.seek(offsets[f])
                        for line in fh:
                            if line.strip():
                                try:
                                    rec = json.loads(line)
                                    comp = rec.get("companion", {})
                                    mark = " [SENSITIVE]" if rec.get("sensitive") else ""
                                    print(
                                        f"{rec.get('ts', '?')[:19]} "
                                        f"[{comp.get('mode', '?')}/{comp.get('intent', '?')}]{mark} "
                                        f"u: {rec.get('user', {}).get('text', '')[:60]!r} -> "
                                        f"{(comp.get('text') or '')[:60]!r}"
                                    )
                                except json.JSONDecodeError:
                                    pass
                    offsets[f] = size
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats")
    sub.add_parser("sessions")
    sub.add_parser("watch")
    sub.add_parser("ratings")

    p_read = sub.add_parser("read")
    p_read.add_argument("session")
    p_read.add_argument("--no-sensitive", action="store_true")

    p_export = sub.add_parser("export")
    p_export.add_argument("session")
    p_export.add_argument("-o", "--out", default=None)
    p_export.add_argument("--no-sensitive", action="store_true")
    p_export.add_argument("--redact", action="store_true")

    p_all = sub.add_parser("export-all")
    p_all.add_argument("-o", "--out", default=None)
    p_all.add_argument("--no-sensitive", action="store_true")
    p_all.add_argument("--redact", action="store_true")

    args = parser.parse_args()
    {"stats": cmd_stats, "sessions": cmd_sessions, "read": cmd_read,
     "export": cmd_export, "export-all": cmd_export_all, "watch": cmd_watch,
     "ratings": cmd_ratings}[args.cmd](args)


if __name__ == "__main__":
    main()
