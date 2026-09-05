"""Model roles (enhance_v1 §25-26) + retention config (§30).

Roles: ROUTER / GENERATOR / VALIDATOR / EMBEDDER / CLASSIFIER.
A generator is not automatically trusted as a validator (§26) — the role
table is explicit. Model replacement = configuration change only; no
architecture code references a provider identity.

Retention (§30): configurable LOG/SESSION/MEMORY retention days +
ENABLE_CONTENT_LOGGING; sensitive-conversation handling (§31) with
normal/sensitive/restricted classes; training-data isolation enforced by
design (chat logs are JSONL diagnostics — there is NO code path from logs
to any training pipeline, §29).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


class ModelRole:
    ROUTER = "router"
    GENERATOR = "generator"
    VALIDATOR = "validator"
    EMBEDDER = "embedder"
    CLASSIFIER = "classifier"


@dataclass
class ModelRoleTable:
    """§25 role declarations loaded from config (never hard-coded)."""

    roles: dict[str, str]

    @classmethod
    def load(cls, config_path: Path = DEFAULT_CONFIG) -> ModelRoleTable:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls(roles=dict(data.get("model_roles", {})))

    def model_for(self, role: str) -> str:
        return self.roles.get(role, "")

    def replace(self, role: str, model: str) -> None:
        """Runtime model swap for one role (§26: config change, no rewrite)."""
        self.roles[role] = model


@dataclass
class RetentionPolicy:
    """§30 configurable retention. 0 days = keep until explicitly deleted."""

    log_retention_days: int = 180
    session_retention_days: int = 2
    memory_retention_days: int = 90
    enable_content_logging: bool = True

    @classmethod
    def load(cls, config_path: Path = DEFAULT_CONFIG) -> RetentionPolicy:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        r = data.get("retention", {}) or {}
        return cls(
            log_retention_days=int(r.get("log_retention_days", 180)),
            session_retention_days=int(r.get("session_retention_days", 2)),
            memory_retention_days=int(r.get("memory_retention_days", 90)),
            enable_content_logging=bool(r.get("enable_content_logging", True)),
        )


class ConversationClass:
    """§31 sensitive-conversation handling classes."""

    NORMAL = "normal"
    SENSITIVE = "sensitive"      # crisis turns: flagged, excluded from default exports
    RESTRICTED = "restricted"    # minimal persistent logging (metadata only)

    @staticmethod
    def logging_allows(conversation_class: str, enable_content_logging: bool) -> bool:
        if conversation_class == ConversationClass.RESTRICTED:
            return False  # restricted: never persist content
        return enable_content_logging
