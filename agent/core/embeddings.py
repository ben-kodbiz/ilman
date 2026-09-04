"""Embedding client for the vector retrieval leg (agentodo.md §8).

OpenAI-compatible `/v1/embeddings` — same backend abstraction rule as chat
(§3): the embedding model is runtime config, never hard-coded. Batched with
retry; failures raise so callers can degrade the leg rather than serve wrong
vectors.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from agent.core.config import AppConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "knowledge" / "processed" / "embeddings.npz"


class EmbeddingClient:
    """nomic-embed-text-v1.5 uses task prefixes for asymmetric retrieval:
    queries get 'search_query: ', corpus documents get 'search_document: '.
    apply() enforces them; raw embed() is available for probes."""

    QUERY_PREFIX = "search_query: "
    DOC_PREFIX = "search_document: "

    def __init__(self, app_config: AppConfig | None = None, batch_size: int = 64):
        cfg = app_config or load_config()
        backend_name = cfg.defaults.get("backend", "")
        self.base_url = cfg.backends[backend_name].base_url
        self.api_key = cfg.backends[backend_name].api_key()
        self.model_id = cfg.backends[backend_name].models.get("embeddings", "")
        if not self.model_id:
            raise ValueError("no 'embeddings' model configured (configs/config.yaml)")
        self.batch_size = batch_size

    def embed_query(self, text: str) -> list[float]:
        return self.embed_one(self.QUERY_PREFIX + text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed([self.DOC_PREFIX + t for t in texts])

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            out.extend(self._embed_batch(chunk))
        return out

    def embed_one(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model_id, "input": texts},
                    timeout=120,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"embeddings HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()["data"]
                if len(data) != len(texts):
                    raise RuntimeError(f"embeddings returned {len(data)} of {len(texts)}")
                return [d["embedding"] for d in data]
            except Exception as e:  # noqa: BLE001 - degrade leg, not crash search
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"embeddings failed after 3 attempts: {last_err}")
