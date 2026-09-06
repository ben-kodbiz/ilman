"""Vector store for the semantic retrieval leg (agentodo.md §8).

Corpus enumeration + embedding cache + top-k cosine search. The cache is a
single .npz (ids, matrix) keyed by citation id; embedding a text happens once
and survives restarts. Source filtering (§8) is enforced by the caller on
the SAME registry as every other leg — vector hits are never trusted blindly.

The texts embedded are the *retrieval views* (English translation for Qur'an,
English for hadith, tafsir body) — never the Arabic Uthmani text itself, which
must stay byte-exact and serves users, not matchers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from agent.core.config import load_config
from agent.core.embeddings import DEFAULT_CACHE, EmbeddingClient

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "knowledge" / "processed" / "knowledge.db"


class VectorStore:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        cache_path: Path = DEFAULT_CACHE,
        client: EmbeddingClient | None = None,
    ):
        self.db_path = db_path
        self.cache_path = cache_path
        self.client = client
        self._ids: list[str] | None = None
        self._matrix: np.ndarray | None = None
        self._index: dict[str, int] | None = None

    # -- corpus enumeration -------------------------------------------------

    @staticmethod
    def _iter_corpus(con: sqlite3.Connection, batch: int = 512):
        """Yield (citation_id, text_to_embed) for every embeddable passage.

        Order is deterministic (citation_id) so cache rebuilds are stable.
        Only tables that exist are queried (partial corpora in tests).
        """
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        selects: list[str] = []
        if {"quran", "quran_translations"} <= tables:
            selects.append(
                "SELECT 'quran:' || surah || ':' || ayah, "
                " (SELECT translation FROM quran_translations t "
                "  WHERE t.surah=q.surah AND t.ayah=q.ayah AND t.lang='en') "
                " FROM quran q WHERE q.source_id='quran-uthmani-json' "
                " AND EXISTS (SELECT 1 FROM quran_translations t2 "
                "  WHERE t2.surah=q.surah AND t2.ayah=q.ayah AND t2.lang='en')"
            )
        if "hadith" in tables:
            selects.append(
                "SELECT 'hadith:' || source_id || ':' || hadithnumber, COALESCE(english,'') "
                " FROM hadith WHERE COALESCE(english,'') != ''"
            )
        if "tafsir" in tables:
            selects.append("SELECT citation_id, tafsir FROM tafsir")
        if "tafsir_en" in tables:
            selects.append("SELECT 'tafsir-en:' || chunk_id, text FROM tafsir_en")
        if "web_fatwas" in tables:
            selects.append(
                "SELECT citation_id, COALESCE(NULLIF(title,'') || ' ', '') || body FROM web_fatwas"
            )
        if not selects:
            return
        rows = con.execute(" UNION ALL ".join(selects) + " ORDER BY 1")
        while True:
            chunk = rows.fetchmany(batch)
            if not chunk:
                return
            yield from chunk

    def corpus_count(self, con: sqlite3.Connection | None = None) -> int:
        own = con is None
        if own:
            con = sqlite3.connect(self.db_path)
        try:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            parts = []
            if {"quran", "quran_translations"} <= tables:
                parts.append(
                    "(SELECT COUNT(*) FROM quran q WHERE EXISTS "
                    "(SELECT 1 FROM quran_translations t WHERE t.surah=q.surah "
                    " AND t.ayah=q.ayah AND t.lang='en'))"
                )
            if "hadith" in tables:
                parts.append("(SELECT COUNT(*) FROM hadith WHERE COALESCE(english,'')!='')")
            if "tafsir" in tables:
                parts.append("(SELECT COUNT(*) FROM tafsir)")
            if "tafsir_en" in tables:
                parts.append("(SELECT COUNT(*) FROM tafsir_en)")
            if "web_fatwas" in tables:
                parts.append("(SELECT COUNT(*) FROM web_fatwas)")
            if not parts:
                return 0
            return con.execute(" + ".join(parts)).fetchone()[0]
        finally:
            if own:
                con.close()

    # -- cache -----------------------------------------------------------------

    def load(self) -> bool:
        """Load the cache; True if usable. Missing/corrupt cache -> False."""
        if not self.cache_path.exists():
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=False)
            self._ids = [str(x) for x in data["ids"]]
            self._matrix = data["matrix"].astype(np.float32)
            if len(self._ids) != self._matrix.shape[0]:
                raise ValueError("ids/matrix length mismatch")
            self._index = {cid: i for i, cid in enumerate(self._ids)}
            return True
        except Exception:
            self._ids = self._matrix = self._index = None
            return False

    def build(self, progress=None) -> int:
        """(Re)build the cache from the corpus. Returns embedded count."""
        if self.client is None:
            self.client = EmbeddingClient(load_config())
        con = sqlite3.connect(self.db_path)
        try:
            items = list(self._iter_corpus(con))
        finally:
            con.close()
        ids: list[str] = []
        vecs: list[list[float]] = []
        total = len(items)
        done = 0
        batch = 128
        for i in range(0, total, batch):
            chunk = items[i : i + batch]
            embedded = self.client.embed_documents([text[:4000] for _, text in chunk])
            for (cid, _), v in zip(chunk, embedded):
                ids.append(cid)
                vecs.append(v)
            done += len(chunk)
            if progress:
                progress(done, total)
        matrix = np.array(vecs, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12  # unit rows
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # unicode ids (not object arrays) so the cache loads with allow_pickle=False
        np.savez_compressed(
            self.cache_path,
            ids=np.array(ids, dtype=np.str_),
            matrix=matrix,
        )
        self._ids = ids
        self._matrix = matrix
        self._index = {cid: i for i, cid in enumerate(ids)}
        return len(ids)

    def is_current(self) -> bool:
        """True when the cache covers the current corpus exactly."""
        if self._ids is None and not self.load():
            return False
        con = sqlite3.connect(self.db_path)
        try:
            expected = {cid for cid, _ in self._iter_corpus(con)}
        finally:
            con.close()
        return set(self._ids or []) == expected

    @property
    def size(self) -> int:
        if self._ids is None and not self.load():
            return 0
        return len(self._ids)

    # -- search ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 12) -> list[dict]:
        """Top-k cosine search. Returns [{'citation_id':..., 'score':...}]."""
        if self._ids is None and not self.load():
            return []
        if self.client is None:
            self.client = EmbeddingClient(load_config())
        q = np.array(self.client.embed_query(query), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        scores = self._matrix @ q  # rows are unit -> this is cosine
        top = np.argsort(-scores)[:top_k]
        return [
            {"citation_id": self._ids[i], "score": float(scores[i])}
            for i in top
        ]

    def has(self, citation_id: str) -> bool:
        if self._index is None and not self.load():
            return False
        return citation_id in self._index
