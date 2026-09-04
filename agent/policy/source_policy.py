"""Sunni source policy (agentodo.md §5, §6).

THE core safety mechanism of the whole project:

- Approved-source registry: what may ever be ingested/retrieved/cited.
- Excluded-source registry: what must never enter the pipeline.
- Ingestion gate: metadata -> Sunni registry -> license -> quality -> ALLOW /
  REJECT / MANUAL_REVIEW. Never ingest first and filter later (§5.2).
- Retrieval filter: mandatory row filters for every query (§8).
- Tiering: TIER 0 Qur'an .. TIER 5 educational (§6). Tier ordering is for
  provenance/retrieval policy, NOT an automatic authority ranking.

The LLM is not the source of Islam (§29). No model output is ever trusted as
evidence about religious content; only registry-approved corpus items are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "knowledge" / "registry"

TIER_ORDER = {
    "quran": 0,
    "sunnah": 1,
    "tafsir": 2,
    "aqidah": 3,
    "fiqh": 3,
    "seerah": 3,
    "contemporary": 4,
    "educational": 5,
}


class Decision(StrEnum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass
class SourceRecord:
    """Per-source metadata (§5.1). This is provenance, not content."""

    id: str
    title: str
    author: str
    type: str
    language: str
    tradition: str
    school: str = ""
    publisher: str = ""
    edition: str = ""
    license: str = ""
    url: str = ""
    verification_status: str = "pending"
    allowed: bool = False
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> SourceRecord:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class GateResult:
    decision: Decision
    reasons: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW


class ExcludedSourceError(Exception):
    """Raised when something on the exclusion list tries to enter the corpus."""


class SourceRegistry:
    """Loads approved + excluded source registries from knowledge/registry/."""

    def __init__(self, approved: dict[str, SourceRecord], excluded: dict[str, dict]):
        self._approved = approved
        self._excluded = excluded

    @classmethod
    def load(cls, registry_dir: Path = REGISTRY_DIR) -> SourceRegistry:
        approved_raw = _load_yaml(registry_dir / "approved_sources.yaml", default={"sources": []})
        excluded_raw = _load_yaml(registry_dir / "excluded_sources.yaml", default={"sources": []})
        approved = {s["id"]: SourceRecord.from_dict(s) for s in approved_raw.get("sources", [])}
        excluded = {s["id"]: s for s in excluded_raw.get("sources", [])}
        overlap = sorted(set(approved) & set(excluded))
        if overlap:
            raise ValueError(f"sources on both registries: {overlap}")
        return cls(approved, excluded)

    def get(self, source_id: str) -> SourceRecord:
        return self._approved[source_id]

    def all_approved(self) -> list[SourceRecord]:
        return list(self._approved.values())

    def is_excluded(self, source_id: str) -> bool:
        return source_id in self._excluded

    def exclusion_reason(self, source_id: str) -> str:
        return self._excluded.get(source_id, {}).get("reason", "")


class SourcePolicy:
    """Implements the §5.2 ingestion gate and the §8 retrieval filter."""

    SUNNI_TRADITION = "SUNNI"

    def __init__(self, registry: SourceRegistry):
        self.registry = registry

    # -- §5.2 ingestion gate -------------------------------------------------
    def ingestion_gate(self, record: SourceRecord) -> GateResult:
        """Registry -> tradition -> license -> quality, before any ingestion.

        Both the incoming record's own claims and the registry entry are
        checked: the registry is the authority for approval state, but a
        document that itself declares a non-Sunni tradition or carries
        unconfirmed licensing must never pass on the registry's say-so.
        """
        # 0. hard exclusion beats everything (§5.3)
        if self.registry.is_excluded(record.id):
            return GateResult(
                Decision.REJECT,
                [f"id '{record.id}' is on the excluded-source registry"],
            )
        # 1. on approved registry
        if record.id not in self.registry._approved:
            return GateResult(Decision.REJECT, ["not on the approved-source registry"])
        registered = self.registry.get(record.id)
        if not registered.allowed:
            return GateResult(Decision.REJECT, ["registry entry has allowed=false"])
        # 2. tradition check — both the document and the registry entry
        for label, rec in (("record", record), ("registry", registered)):
            if rec.tradition.upper() != self.SUNNI_TRADITION:
                return GateResult(
                    Decision.REJECT, [f"{label} tradition is {rec.tradition}, not SUNNI"]
                )
        # 3. verification status — registry is authoritative
        if registered.verification_status == "rejected":
            return GateResult(Decision.REJECT, ["registry verification_status is 'rejected'"])
        if record.verification_status == "rejected":
            return GateResult(Decision.REJECT, ["record verification_status is 'rejected'"])
        if registered.verification_status != "verified":
            return GateResult(
                Decision.MANUAL_REVIEW,
                [f"registry verification_status is '{registered.verification_status}'"],
            )
        # 4. license check — incoming claim and registry entry
        for label, rec in (("record", record), ("registry", registered)):
            if not rec.license.strip():
                return GateResult(Decision.MANUAL_REVIEW, [f"{label} license missing/empty"])
            if rec.license.strip().lower() in {"unknown", "unverified", "todo", "review_required", ""}:
                return GateResult(
                    Decision.MANUAL_REVIEW, [f"{label} license not confirmed: '{rec.license}'"]
                )
        # 5. quality check: provenance metadata must be present on both
        for label, rec in (("record", record), ("registry", registered)):
            missing = [
                f for f in ("title", "author", "type", "language")
                if not getattr(rec, f).strip()
            ]
            if missing:
                return GateResult(
                    Decision.MANUAL_REVIEW, [f"{label} metadata incomplete: missing {missing}"]
                )
        return GateResult(Decision.ALLOW, ["passed registry, tradition, license, quality gates"])

    # -- §8 retrieval filter --------------------------------------------------
    def retrieval_filter(self, record: SourceRecord) -> bool:
        """Mandatory filters: allowed && tradition==SUNNI && not rejected."""
        return (
            record.allowed
            and record.tradition.upper() == self.SUNNI_TRADITION
            and record.verification_status != "rejected"
        )

    def must_not_retrieve(self, source_id: str) -> bool:
        """Excluded material must never be retrieved/cited/merged (§5.3)."""
        return self.registry.is_excluded(source_id)

    # -- §6 tiering -------------------------------------------------------------
    def tier(self, record: SourceRecord) -> int:
        return TIER_ORDER.get(record.type, 5)

    def assert_ingestible(self, record: SourceRecord) -> None:
        """Strict helper: ingestion code must call this before storing anything.

        Raises instead of returning a soft decision because ingestion of
        non-approved material is a hard error (never ingest first and filter
        later). MANUAL_REVIEW also blocks automated ingestion until a human
        resolves it.
        """
        result = self.ingestion_gate(record)
        if result.decision is not Decision.ALLOW:
            raise ExcludedSourceError(
                f"ingestion blocked for '{record.id}': {result.decision} ({'; '.join(result.reasons)})"
            )


def _load_yaml(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
