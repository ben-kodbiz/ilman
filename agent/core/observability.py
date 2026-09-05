"""Structured observability (fixme_v2 §31-32).

Logs/returns decision metadata — intent, emotion, risk, mode, policy, RAG
used, memory used, model, validation, latency — NEVER conversation text and
NEVER chain-of-thought (§31). The DebugTrace is developer-facing only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DebugTrace:
    """fixme_v2 §32: the developer-only trace object."""

    intent: str = ""
    emotion: str | None = None
    risk: str = "low"
    mode: str = "qa"
    route: str = ""
    memory_hits: int = 0
    memory_saved: int = 0
    rag_used: bool = False
    followup_allowed: int = 1
    model: str = ""
    validation: str = ""
    planned_query: dict = field(default_factory=dict)
    evidence_status: str = ""
    evidence_sufficiency: float = 0.0
    policy: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    started_at: float = field(default_factory=time.time)

    def mark_validation(self, ok: bool, extra: str = "") -> None:
        self.validation = ("pass" if ok else "fail") + (f": {extra}" if extra else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "emotion": self.emotion,
            "risk": self.risk,
            "mode": self.mode,
            "route": self.route,
            "memory_hits": self.memory_hits,
            "memory_saved": self.memory_saved,
            "rag_used": self.rag_used,
            "model": self.model,
            "validation": self.validation,
            "planned_query": self.planned_query,
            "evidence_status": self.evidence_status,
            "evidence_sufficiency": self.evidence_sufficiency,
            "policy": self.policy,
            "latency_s": round(self.latency_s, 2),
        }
