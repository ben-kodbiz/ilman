"""Evidence lifecycle (enhance_v1 §4) — explicit state machine.

States (§4.1):
    DISCOVERED → RETRIEVED → FILTERED → QUARANTINED → ADMITTED → USED
    → VALIDATED → FINAL
Rejected evidence: FILTERED → REJECTED (terminal — never re-enters
generation; this was already harness behavior, now formalized).

Deterministic: no model anywhere in this module (§45).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any


class EvidenceState(StrEnum):
    DISCOVERED = auto()
    RETRIEVED = auto()
    FILTERED = auto()
    QUARANTINED = auto()
    REJECTED = auto()
    ADMITTED = auto()
    USED = auto()
    VALIDATED = auto()
    FINAL = auto()


#: §4.1 explicit transition table. Anything not listed raises.
EVIDENCE_TRANSITIONS: dict[EvidenceState, set[EvidenceState]] = {
    EvidenceState.DISCOVERED: {EvidenceState.RETRIEVED},
    EvidenceState.RETRIEVED: {EvidenceState.FILTERED},
    EvidenceState.FILTERED: {EvidenceState.QUARANTINED, EvidenceState.REJECTED},
    EvidenceState.QUARANTINED: {EvidenceState.ADMITTED, EvidenceState.REJECTED},
    EvidenceState.ADMITTED: {EvidenceState.USED, EvidenceState.REJECTED},
    EvidenceState.USED: {EvidenceState.VALIDATED},
    EvidenceState.VALIDATED: {EvidenceState.FINAL},
    EvidenceState.REJECTED: set(),  # terminal
    EvidenceState.FINAL: set(),  # terminal
}


class InvalidEvidenceTransition(RuntimeError):
    """Raised on any transition outside the table (§4.1: invalid transitions
    must raise, never silently coerce)."""


@dataclass
class EvidenceItem:
    """One retrievable passage tracked through the lifecycle, with
    provenance retained end-to-end (§4 acceptance)."""

    citation_id: str
    state: EvidenceState = EvidenceState.DISCOVERED
    # provenance (§4 'evidence provenance')
    source_id: str = ""
    source_type: str = ""  # quran | hadith | tafsir | tafsir-en | external_untrusted
    tier: int = 5
    retrieval_leg: str = ""
    retrieval_score: float = 0.0
    quarantine_reason: str | None = None
    authority_level: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition(self, target: EvidenceState, reason: str = "") -> EvidenceItem:
        allowed = EVIDENCE_TRANSITIONS.get(self.state, set())
        if target not in allowed:
            raise InvalidEvidenceTransition(
                f"evidence {self.citation_id}: {self.state.value} -> "
                f"{target.value} is not an allowed transition"
            )
        self.state = target
        if target is EvidenceState.QUARANTINED and reason:
            self.quarantine_reason = reason
        return self

    @property
    def is_terminal(self) -> bool:
        return self.state in (EvidenceState.REJECTED, EvidenceState.FINAL)

    @property
    def generation_eligible(self) -> bool:
        """Only ADMITTED+ evidence may reach the model (§34: modules may not
        inject arbitrary evidence; quarantined/rejected never do)."""
        return self.state in (
            EvidenceState.ADMITTED, EvidenceState.USED,
            EvidenceState.VALIDATED, EvidenceState.FINAL,
        )

    def to_dict(self) -> dict:
        return {
            "citation_id": self.citation_id, "state": self.state.value,
            "source_id": self.source_id, "source_type": self.source_type,
            "tier": self.tier, "leg": self.retrieval_leg,
            "score": round(self.retrieval_score, 3),
            "quarantine_reason": self.quarantine_reason,
        }


def lifecycle_from_retrieval(passages: list, plan=None) -> list[EvidenceItem]:
    """Adapt existing RetrievedPassage results into lifecycle items.

    Runs the canonical sequence: every retrieved passage is RETRIEVED →
    FILTERED. The harness's quarantine step then moves FILTERED →
    QUARANTINED/ADMITTED (see harness integration).
    """
    items: list[EvidenceItem] = []
    for p in passages:
        state = EvidenceState.RETRIEVED
        item = EvidenceItem(
            citation_id=p.citation_id,
            state=state,
            source_id=p.source_id,
            source_type=_source_type_of(p.citation_id),
            tier=getattr(p, "tier", 5),
            retrieval_leg=getattr(p, "leg", ""),
            retrieval_score=float(getattr(p, "score", 0.0) or 0.0),
        )
        item.transition(EvidenceState.FILTERED)
        items.append(item)
    return items


def quarantine_filter(items: list[EvidenceItem],
                      keep_citation_ids: set[str]) -> list[EvidenceItem]:
    """FILTERED → QUARANTINED (kept) or REJECTED (dropped, terminal).

    keep_citation_ids: the citation ids the harness's relevance quarantine
    decided to keep. Everything else is REJECTED — permanently (§4:
    'rejected evidence must be terminal').
    """
    for item in items:
        if item.state is not EvidenceState.FILTERED:
            continue
        if item.citation_id in keep_citation_ids:
            item.transition(EvidenceState.QUARANTINED, reason="passed relevance")
        else:
            item.transition(
                EvidenceState.REJECTED, reason="quarantine: irrelevant to query"
            )
    return [i for i in items if i.state is not EvidenceState.REJECTED]


def admit_all(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """QUARANTINED → ADMITTED for the generation-eligible set."""
    for item in items:
        if item.state is EvidenceState.QUARANTINED:
            item.transition(EvidenceState.ADMITTED, reason="admitted to pack")
    return items


def mark_used_validated(items: list[EvidenceItem],
                         used_ids: set[str], verified_ids: set[str]) -> None:
    """ADMITTED → USED (cited by generation) → VALIDATED (passed the judge)."""
    for item in items:
        if item.state is EvidenceState.ADMITTED and item.citation_id in used_ids:
            item.transition(EvidenceState.USED)
    for item in items:
        if item.state is EvidenceState.USED and item.citation_id in verified_ids:
            item.transition(EvidenceState.VALIDATED)


def mark_final(items: list[EvidenceItem]) -> None:
    for item in items:
        if item.state in (EvidenceState.VALIDATED, EvidenceState.USED,
                          EvidenceState.ADMITTED):
            item.transition(EvidenceState.FINAL)


def _source_type_of(citation_id: str) -> str:
    if citation_id.startswith("quran:"):
        return "quran"
    if citation_id.startswith("hadith:"):
        return "hadith"
    if citation_id.startswith("tafsir-en:"):
        return "tafsir"
    if citation_id.startswith("tafsir:"):
        return "tafsir"
    return "external_untrusted"
