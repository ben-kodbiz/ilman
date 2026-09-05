"""Claim dependency graph (enhance_v1 §6-8).

Extends the existing typed claims (agent/validators/claim_policy.py —
§3.1 reuse, not replacement) with:

  §6.1  stable IDs, evidence_refs, dependencies, severity, status
  §7    explicit relations: SUPPORTS/DEPENDS_ON/INFERRED_FROM/CONTRADICTS/
        QUALIFIES/REFINES
  §7    invalidation propagation: unsupported parent → dependent claims
        INVALIDATED (transitively)
  §8    severity: LOW/MEDIUM/HIGH/CRITICAL (false Prophet-attribution,
        halal/haram declarations, medical guarantees = CRITICAL)

Deterministic graph traversal — no model (§45).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto

from agent.validators.claim_policy import ClaimType, extract_typed_claims


class ClaimStatus(StrEnum):
    PENDING = auto()
    SUPPORTED = auto()
    PARTIAL = auto()
    UNSUPPORTED = auto()
    INVALIDATED = auto()  # §7: parent claim failed → this claim dies with it
    REMOVED = auto()


class ClaimRelation(StrEnum):
    SUPPORTS = auto()
    DEPENDS_ON = auto()
    INFERRED_FROM = auto()
    CONTRADICTS = auto()
    QUALIFIES = auto()
    REFINES = auto()


class Severity(StrEnum):
    LOW = auto()      # minor unsupported descriptive statement
    MEDIUM = auto()   # potentially misleading interpretation
    HIGH = auto()     # religious ruling, attribution, causal claim, prescription
    CRITICAL = auto()  # false Allah/Prophet attribution; definitive halal/
    # haram without authority; medical guarantees as religious fact;
    # dangerous certainty in high-risk situations


#: §8 severity mapping by claim type (existing taxonomy from
#: claim_policy.py — the enhance_v1 §6.1 additions QURAN_CLAIM/HADITH_CLAIM/
#: FIQH_RULING map onto DIRECT_FACT/ATTRIBUTION/RULING which the extractor
#: actually emits)
SEVERITY_BY_TYPE: dict[ClaimType, Severity] = {
    ClaimType.DIAGNOSIS: Severity.CRITICAL,
    ClaimType.GUARANTEE: Severity.CRITICAL,
    ClaimType.RULING: Severity.CRITICAL,          # also covers FIQH_RULING
    ClaimType.PREDICTION: Severity.HIGH,
    ClaimType.CAUSAL_CLAIM: Severity.HIGH,
    ClaimType.ATTRIBUTION: Severity.CRITICAL,     # also HADITH_CLAIM:
    #  false Prophet ﷺ attribution is CRITICAL (§8)
    ClaimType.DIRECT_FACT: Severity.HIGH,          # also QURAN_CLAIM
    ClaimType.GENERALIZATION: Severity.HIGH,
    ClaimType.INFERENCE: Severity.MEDIUM,
    ClaimType.INTERPRETATION: Severity.MEDIUM,
    ClaimType.PARAPHRASE: Severity.MEDIUM,
    ClaimType.PLAIN: Severity.LOW,                # ADVICE/OPINION territory
}


@dataclass
class GraphClaim:
    """§6.1 claim model over the existing typed extraction."""

    id: str
    text: str
    claim_type: ClaimType
    severity: Severity
    evidence_refs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # parent claim ids
    relations: list[tuple[str, ClaimRelation]] = field(default_factory=list)
    status: ClaimStatus = ClaimStatus.PENDING
    # two-dimensional validation results (§11)
    entailment: str = ""  # evidence judge verdict
    authority: str = ""   # authority check result

    @property
    def is_high_risk(self) -> bool:
        return self.severity in (Severity.HIGH, Severity.CRITICAL)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text[:120],
            "type": self.claim_type.value, "severity": self.severity.value,
            "status": self.status.value, "evidence_refs": self.evidence_refs,
            "dependencies": self.dependencies,
            "relations": [(pid, r.value) for pid, r in self.relations],
            "entailment": self.entailment, "authority": self.authority,
        }


def build_claim_graph(text: str) -> list[GraphClaim]:
    """Extract typed claims → graph claims with stable c1..cN ids and
    inference-boundary dependency links (reuses extract_typed_claims' §9
    dependency_on detection from fixme_v3.1)."""
    typed = extract_typed_claims(text)
    graph: list[GraphClaim] = []
    id_by_sentence: dict[str, str] = {}
    for idx, tc in enumerate(typed, start=1):
        cid = f"c{idx}"
        severity = SEVERITY_BY_TYPE.get(tc.claim_type, Severity.MEDIUM)
        dependencies: list[str] = []
        relations: list[tuple[str, ClaimRelation]] = []
        if tc.dependency_on and tc.dependency_on in id_by_sentence:
            parent_id = id_by_sentence[tc.dependency_on]
            dependencies.append(parent_id)
            relations.append((parent_id, ClaimRelation.INFERRED_FROM))
        graph.append(GraphClaim(
            id=cid, text=tc.sentence, claim_type=tc.claim_type,
            severity=severity, evidence_refs=list(tc.citations),
            dependencies=dependencies, relations=relations,
        ))
        id_by_sentence[tc.sentence] = cid
    return graph


def propagate_invalidation(graph: list[GraphClaim]) -> list[GraphClaim]:
    """§7: unsupported parent → dependents INVALIDATED (transitively).

    Runs after entailment/authority statuses are set: any UNSUPPORTED (or
    INVALIDATED) claim invalidates everything that DEPENDS_ON / is
    INFERRED_FROM it. Deterministic fixpoint traversal."""
    changed = True
    while changed:
        changed = False
        for claim in graph:
            if claim.status is not ClaimStatus.UNSUPPORTED:
                continue
            for other in graph:
                if other.id == claim.id or other.status is ClaimStatus.INVALIDATED:
                    continue
                if claim.id in other.dependencies:
                    other.status = ClaimStatus.INVALIDATED
                    changed = True
    return graph


def invalidation_closure(graph: list[GraphClaim]) -> set[str]:
    """Ids of all claims that must be removed: UNSUPPORTED + INVALIDATED."""
    return {
        c.id for c in graph
        if c.status in (ClaimStatus.UNSUPPORTED, ClaimStatus.INVALIDATED)
    }
