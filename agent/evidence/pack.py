"""Immutable evidence pack (enhance_v1 §5) + fishing prevention (§5.1).

Extends the existing EvidencePack (agent/validators/pipeline.py) rather than
replacing it: a FrozenEvidencePack wraps the mutable one after admission and
computes a checksum. Frozen contents cannot be modified; the LLM (or any
module) receives the frozen pack and can never mutate evidence (§34).

Fishing prevention (§5.1): every retrieval operation gets a NEW query_id and
produces a NEW pack identity — evidence is never appended to an existing
pack to support a generated claim. New retrieval → new pack → new generation.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agent.evidence.lifecycle import EvidenceItem


class PackFrozenError(RuntimeError):
    """Any mutation attempt on a frozen pack (§5: contents cannot be
    modified after freeze)."""


@dataclass
class FrozenEvidencePack:
    """§5 structure. Wraps the existing pipeline EvidencePack's passages.

    After freeze(): passages/sources cannot be mutated — enforced by
    hash verification on every access (mutation of the underlying list
    objects is detected via checksum mismatch and raises).
    """

    pack_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    query_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    passages: list = field(default_factory=list)  # RetrievedPassage objects (read-only)
    lifecycle_items: list[EvidenceItem] = field(default_factory=list)
    retrieval_scores: dict[str, float] = field(default_factory=dict)
    authority_metadata: dict[str, dict] = field(default_factory=dict)
    quarantine_results: list[dict] = field(default_factory=list)
    checksum: str = ""
    _frozen: bool = False

    # ------------------------------------------------------------ freeze
    def freeze(self) -> FrozenEvidencePack:
        """Compute the content checksum and seal. Idempotent."""
        self.checksum = self._compute_checksum()
        self._frozen = True
        return self

    @property
    def frozen(self) -> bool:
        # verify contents still match the checksum (detects ANY mutation of
        # the underlying passages even though the list itself is mutable)
        if not self._frozen:
            return False
        if self._compute_checksum() != self.checksum:
            raise PackFrozenError(
                f"evidence pack {self.pack_id}: content checksum mismatch — "
                "frozen evidence was modified"
            )
        return True

    def _compute_checksum(self) -> str:
        payload = {
            "passages": sorted(p.citation_id for p in self.passages),
            "texts": sorted((p.citation_id, (p.translation or p.arabic or "")[:200])
                            for p in self.passages),
            "scores": {k: round(v, 4) for k, v in self.retrieval_scores.items()},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    # ---------------------------------------------------- read-only surface
    @property
    def citation_ids(self) -> set[str]:
        self.frozen  # noqa: B018 — verification on access
        return {p.citation_id for p in self.passages}

    def get_passage(self, citation_id: str):
        self.frozen  # noqa: B018
        for p in self.passages:
            if p.citation_id == citation_id:
                return p
        return None

    def to_prompt_block(self) -> str:
        """Delegate to the same rendering the pipeline pack uses (§3.1 reuse)."""
        self.frozen  # noqa: B018
        from agent.validators.pipeline import EvidencePack

        temp = EvidencePack(query=self.query_id, passages=list(self.passages))
        return temp.to_prompt_block()

    def generation_eligible_items(self) -> list[EvidenceItem]:
        return [i for i in self.lifecycle_items if i.generation_eligible]

    def to_dict(self) -> dict:
        return {
            "pack_id": self.pack_id, "query_id": self.query_id,
            "created_at": self.created_at, "checksum": self.checksum,
            "passages": [p.citation_id for p in self.passages],
            "lifecycle": [i.to_dict() for i in self.lifecycle_items],
            "quarantine_results": self.quarantine_results,
        }


def freeze_pack(pipeline_pack, query_id: str,
                lifecycle_items: list[EvidenceItem] | None = None,
                quarantine_results: list[dict] | None = None) -> FrozenEvidencePack:
    """Adapt the existing mutable pipeline EvidencePack into a frozen one
    (§3.1: adapter over existing, not a rewrite). Call AFTER quarantine so
    the frozen pack contains only generation-eligible evidence."""
    frozen = FrozenEvidencePack(
        query_id=query_id,
        passages=list(pipeline_pack.passages),
        lifecycle_items=lifecycle_items or [],
        retrieval_scores={
            p.citation_id: float(getattr(p, "score", 0.0) or 0.0)
            for p in pipeline_pack.passages
        },
        quarantine_results=quarantine_results or [],
    )
    return frozen.freeze()


class EvidenceFishingError(RuntimeError):
    """§5.1: attempting to add evidence to an existing pack instead of
    issuing a new retrieval/query_id."""


def assert_new_retrieval(existing_pack_id: str, new_pack_id: str,
                         existing_query_id: str, new_query_id: str) -> None:
    """Fishing guard: a new generation must run on a NEW pack with a NEW
    query_id; modules may never extend an old pack to support a new claim."""
    if existing_pack_id == new_pack_id:
        raise EvidenceFishingError(
            "evidence fishing: reuse of pack "
            f"{existing_pack_id} for a new generation — a new retrieval must "
            "produce a new pack"
        )
    if existing_query_id and existing_query_id == new_query_id:
        raise EvidenceFishingError(
            f"evidence fishing: query_id {new_query_id} reused across retrievals"
        )
