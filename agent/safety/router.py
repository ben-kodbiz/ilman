"""Safety router (fixme_v2 §19). INDEPENDENT from companion policy.

Risk levels low / elevated / high. Safety policy can never be overridden by
companion policy. High -> canned supportive response + real-world contacts,
no harmful instructions, no religious guilt, model never invoked.

Reuses the v1 crisis patterns (already regression-tested) and adds the
three-level mapping + independent router API.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.companion.safety import (
    SAFETY_RESPONSE_EN,
    SAFETY_RESPONSE_MS,
    classify_safety,
)
from agent.state.models import Risk


@dataclass
class SafetyDecision:
    risk: Risk
    response: str | None = None      # canned response when high
    model_allowed: bool = True       # False -> harness must not call the model
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "risk": self.risk.value,
            "model_allowed": self.model_allowed,
            "notes": self.notes,
        }


def safety_route(message: str) -> SafetyDecision:
    """The independent safety gate; called BEFORE any companion logic."""
    result = classify_safety(message)
    from agent.companion.safety import Severity

    if result.severity is Severity.HIGH_RISK:
        return SafetyDecision(
            risk=Risk.HIGH, response=None, model_allowed=False,
            notes=f"high-risk signal: {result.matched!r}",
        )
    if result.severity is Severity.MODERATE_DISTRESS:
        return SafetyDecision(
            risk=Risk.ELEVATED, response=None, model_allowed=True,
            notes=f"elevated distress: {result.matched!r}",
        )
    return SafetyDecision(risk=Risk.LOW, model_allowed=True)


def canned_safety_response(lang: str = "en") -> str:
    return SAFETY_RESPONSE_MS if lang.startswith(("ms", "id")) else SAFETY_RESPONSE_EN
