#!/usr/bin/env bash
# Dev workflow entrypoints (agentodo.md §26 Phase 0/1).
# Repo has no installed package tooling yet; uv manages the venv.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-help}" in
  install)
    uv sync
    ;;
  test)
    uv run pytest -q
    ;;
  lint)
    uv run ruff check agent evaluation tests scripts 2>/dev/null || uv run ruff check agent evaluation tests
    ;;
  serve)
    uv run uvicorn apps.api.server:app --host "${ILMAN_HOST:-127.0.0.1}" --port 8017
    ;;
  serve-lan)
    ILMAN_HOST=0.0.0.0 uv run uvicorn apps.api.server:app --host 0.0.0.0 --port 8017
    ;;
  console)
    uv run python -m apps.devui.console
    ;;
  console-lan)
    ILMAN_HOST=0.0.0.0 uv run python -m apps.devui.console
    ;;
  bench)
    shift
    uv run python -m evaluation.bench.runner "$@"
    ;;
  *)
    echo "usage: scripts/dev.sh {install|test|lint|serve|serve-lan|console|console-lan|bench [--models ROLE ...] [--suites NAME ...]}"
    ;;
esac
