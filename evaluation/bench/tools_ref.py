"""Tool schema resolution for benchmark suites.

Suites declare `tools_ref: all` to receive the agent's real tool schemas,
keeping benchmarks in sync with the actual tool layer.
"""

from __future__ import annotations

from agent.tools.layer import TOOL_SCHEMAS


def resolve_tools(tools_ref: str | None) -> list[dict]:
    if tools_ref == "all":
        return TOOL_SCHEMAS
    return []
