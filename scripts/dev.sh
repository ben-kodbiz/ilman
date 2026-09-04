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
  bench)
    shift
    uv run python -m evaluation.bench.runner "$@"
    ;;
  *)
    echo "usage: scripts/dev.sh {install|test|lint|bench [--models ROLE ...] [--suites NAME ...]}"
    ;;
esac
