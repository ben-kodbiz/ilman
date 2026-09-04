"""Config loading for the Huurs agent core.

Runtime model routing and backend endpoints are ALWAYS loaded from YAML config,
never hard-coded (agentodo.md §3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


@dataclass
class BackendConfig:
    name: str
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"
    models: dict[str, str] = field(default_factory=dict)

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "not-needed")


@dataclass
class AppConfig:
    backends: dict[str, BackendConfig]
    routing: dict[str, str]
    defaults: dict[str, str] = field(default_factory=dict)

    def backend_for(self, role: str) -> tuple[BackendConfig, str]:
        model_role = self.routing.get(role) or self.defaults.get("model_role", "")
        if not model_role:
            raise ValueError(f"no routing configured for task class '{role}'")
        backend_name = self.defaults.get("backend", "")
        if not backend_name or backend_name not in self.backends:
            raise ValueError(f"default backend '{backend_name}' not defined")
        model_id = self.backends[backend_name].models.get(model_role, "")
        if not model_id:
            raise ValueError(f"backend '{backend_name}' has no model for role '{model_role}'")
        return self.backends[backend_name], model_id


def load_config(path: Path | None = None) -> AppConfig:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    backends = {}
    for name, spec in raw.get("backends", {}).items():
        backends[name] = BackendConfig(
            name=name,
            base_url=spec["base_url"],
            api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
            models=dict(spec.get("models", {})),
        )
    return AppConfig(
        backends=backends,
        routing=dict(raw.get("routing", {})),
        defaults=dict(raw.get("defaults", {})),
    )
