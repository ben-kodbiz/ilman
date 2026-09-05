"""Optional local semantic entailment backend (fixme_v3.1 §23).

Interface only + heuristic first implementation. A future local NLI/LLM
backend can replace `_HeuristicEntailment.evaluate` without touching callers.
Never a cloud API. Not adopted automatically — benchmark before use (§23).

Result scale maps onto the EvidenceJudge's verdict vocabulary so the two
layers compose without conversion drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

from agent.validators.evidence_judge import _content_stems


class Entailment(StrEnum):
    ENTAILS = auto()
    PARTIALLY_ENTAILS = auto()
    NEUTRAL = auto()
    CONTRADICTS = auto()
    UNKNOWN = auto()


@dataclass
class EntailmentResult:
    verdict: Entailment
    score: float = 0.0  # 0..1 confidence in the verdict
    detail: str = ""


class _HeuristicEntailment:
    """First implementation (v3.1 §23: 'The first implementation may remain
    heuristic'). Bidirectional stem containment + negation-signal check."""

    _NEGATIONS = ("not ", "never ", "no ", "none ", "neither ")
    _ANTONYM_PAIRS = [
        ("always", "never"), ("all", "none"), ("cures", "causes"),
        ("removes", "adds"), ("heals", "harm"), ("guaranteed", "uncertain"),
        ("near", "far"), ("with", "without"),
    ]

    def evaluate(self, claim: str, evidence: str) -> EntailmentResult:
        claim_stems = _content_stems(claim)
        evidence_stems = _content_stems(evidence)
        if not claim_stems or not evidence_stems:
            return EntailmentResult(Entailment.UNKNOWN, 0.0, "no content stems")

        covered = sum(1 for t in claim_stems if t in evidence_stems)
        coverage = covered / max(len(claim_stems), 1)

        # negation-signal: an explicit polarity flip in the evidence on a
        # highly-covered claim is a contradiction signal (crude but safe)
        claim_lower, evidence_lower = claim.lower(), evidence.lower()
        flipped = any(
            (a in claim_lower and b in evidence_lower)
            or (b in claim_lower and a in evidence_lower)
            for a, b in self._ANTONYM_PAIRS
        )
        if flipped and coverage >= 0.5:
            return EntailmentResult(Entailment.CONTRADICTS, coverage, "polarity flip")

        if coverage >= 0.8:
            return EntailmentResult(Entailment.ENTAILS, coverage, "stem containment")
        if coverage >= 0.5:
            return EntailmentResult(
                Entailment.PARTIALLY_ENTAILS, coverage, "partial stem containment"
            )
        return EntailmentResult(Entailment.NEUTRAL, coverage, "low containment")


# The backend instance callers program against (swap for a local NLI later).
def get_backend() -> EntailmentBackend:
    return _HeuristicEntailment()


EntailmentBackend = _HeuristicEntailment  # duck-typed interface alias
