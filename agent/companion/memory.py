"""Companion memory: user facts + controls + relevant retrieval (fix_me.md §5, §6).

Extends MemoryStore (which keeps profile/study/history/bookmarks). Adds:

- user_facts: ONLY useful, stable, explicitly-shared facts (§5C). A saving
  policy gate rejects transient feelings — emotions are conversation state,
  NOT memory (§4).
- expiration: facts carry a last-referenced timestamp; stale facts expire.
- controls: list / forget / clear (§25).
- relevant retrieval: keyword-overlap scoring against the current message —
  never the whole memory database into the prompt (§6).

Emotional content is NEVER auto-saved as a fact. "User feels lonely" is state,
not a durable fact; saving it would violate §25 ("Never store emotional state
permanently without a clear product reason").
"""

from __future__ import annotations

import re
import time
from datetime import UTC
from pathlib import Path

from agent.memory.store import DEFAULT_DB, MemoryStore

FACT_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    fact TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    last_referenced REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'user',
    -- enhance_v1 §18-20 memory provenance columns
    source_class TEXT NOT NULL DEFAULT 'explicit_user',
    confidence REAL NOT NULL DEFAULT 1.0,
    expires_at REAL,
    updated_at REAL
);
"""

# Words that mark transient emotional content — never a durable fact.
_TRANSIENT_RE = re.compile(
    r"\b(today|tonight|right now|currently|at the moment|temporarily)\b"
    r"|\b(i feel|i felt|feeling)\b",
    re.IGNORECASE,
)


class FactRejected(ValueError):
    """Raised when a fact fails the §5C saving policy."""


class CompanionMemory(MemoryStore):
    def __init__(self, db_path: Path = DEFAULT_DB, memory_enabled: bool = True):
        super().__init__(db_path)
        import sqlite3

        with sqlite3.connect(self.db_path) as con:
            con.executescript(FACT_SCHEMA)
            self._migrate(con)
        self.memory_enabled = memory_enabled
        self.fact_ttl_days = 90

    @staticmethod
    def _migrate(con) -> None:
        """Backward-compatible column migration: older DBs created before
        enhance_v1 §18 lack the provenance columns; add them in place."""
        cols = {r[1] for r in con.execute("PRAGMA table_info(user_facts)")}
        migrations = {
            "source_class": "ALTER TABLE user_facts ADD COLUMN source_class TEXT NOT NULL DEFAULT 'explicit_user'",
            "confidence": "ALTER TABLE user_facts ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0",
            "expires_at": "ALTER TABLE user_facts ADD COLUMN expires_at REAL",
            "updated_at": "ALTER TABLE user_facts ADD COLUMN updated_at REAL",
        }
        for col, ddl in migrations.items():
            if col not in cols:
                con.execute(ddl)

    # -- §5C policy gate --------------------------------------------------
    def save_fact(self, fact: str, category: str = "general",
                  explicit: bool = False, confidence: float = 1.0,
                  expires_at: float | None = None,
                  source_class: str | None = None) -> int:
        """Save a durable user fact with provenance (enhance_v1 §18).

        Source classes (§19): EXPLICIT_USER (persistable per policy) /
        SYSTEM / IMPORTED / INFERRED (must NOT silently become permanent —
        inferred facts require lower confidence + expiry) / TEMPORARY.
        Rejects transient emotional statements unless the user EXPLICITLY
        asked to remember. INFERRED ≠ FACT (§19 critical rule): inferred
        memories carry confidence < 1.0 and an expiry by default."""
        fact = (fact or "").strip()
        if not fact:
            raise FactRejected("empty fact")
        if not self.memory_enabled:
            raise FactRejected("memory is disabled by user setting")
        if not explicit and _TRANSIENT_RE.search(fact):
            raise FactRejected(
                "transient emotional content is conversation state, not memory (§5C)"
            )
        from datetime import datetime

        # §19 provenance: source class + confidence + expiry
        if source_class is None:
            source_class = "explicit_user" if explicit else "inferred"
        source_class = source_class.lower()
        if source_class not in ("explicit_user", "system", "imported",
                                "inferred", "temporary"):
            raise FactRejected(f"unknown source class: {source_class}")
        if source_class == "inferred":
            # §19: INFERRED must NOT automatically become permanent
            confidence = min(confidence, 0.7)
            if expires_at is None:
                expires_at = time.time() + 14 * 86400  # 14-day default expiry
        if source_class == "temporary":
            if expires_at is None:
                expires_at = time.time() + 86400  # session-scale default

        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO user_facts (created_at, fact, category, last_referenced, source, "
                "source_class, confidence, expires_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now(UTC).isoformat(), fact, category, time.time(),
                 "user" if explicit else "inferred",
                 source_class, confidence, expires_at, time.time()),
            )
            return cur.lastrowid

    def facts(self, limit: int = 50) -> list[dict]:
        self._expire_facts()
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, created_at, fact, category, source, source_class, "
                "confidence, expires_at FROM user_facts "
                "ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def forget_fact(self, fact_id: int) -> bool:
        with self._connect() as con:
            cur = con.execute("DELETE FROM user_facts WHERE id=?", (fact_id,))
            return cur.rowcount > 0

    def clear_facts(self) -> int:
        with self._connect() as con:
            cur = con.execute("DELETE FROM user_facts")
            return cur.rowcount

    def _expire_facts(self) -> None:
        """§20 retention: last-referenced TTL + explicit expires_at both
        purge; deterministic sweep on every read."""
        now = time.time()
        ttl_cutoff = now - self.fact_ttl_days * 86400
        with self._connect() as con:
            con.execute("DELETE FROM user_facts WHERE last_referenced < ?", (ttl_cutoff,))
            con.execute(
                "DELETE FROM user_facts WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )

    # -- §6 relevant retrieval -------------------------------------------
    def relevant_facts(self, message: str, limit: int = 3) -> list[dict]:
        """Keyword-overlap retrieval: only memories relevant to the current
        message (§6: 'Do NOT send the entire memory database to the model')."""
        self._expire_facts()
        stop = {
            "the", "a", "an", "i", "my", "me", "and", "or", "to", "of", "in",
            "is", "are", "was", "feel", "feeling", "about", "that", "this",
        }
        tokens = {t for t in re.findall(r"[a-zA-Z']{3,}", message.lower())} - stop
        if not tokens:
            return []
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, fact, category, last_referenced FROM user_facts"
            ).fetchall()
        scored: list[tuple[float, dict]] = []
        now = time.time()
        for r in rows:
            fact_tokens = {t for t in re.findall(r"[a-zA-Z']{3,}", r["fact"].lower())} - stop
            if not fact_tokens:
                continue
            overlap = len(tokens & fact_tokens) / len(fact_tokens)
            if overlap <= 0:
                continue
            recency = max(0.0, 1.0 - (now - r["last_referenced"]) / (self.fact_ttl_days * 86400))
            scored.append((overlap * 0.8 + recency * 0.2, dict(r)))
        scored.sort(key=lambda x: -x[0])
        top = [fact for score, fact in scored[:limit] if score > 0.15]
        if top:
            with self._connect() as con:
                for fact in top:
                    con.execute(
                        "UPDATE user_facts SET last_referenced=? WHERE id=?",
                        (now, fact["id"]),
                    )
        return [
            {"id": f["id"], "fact": f["fact"], "category": f["category"]} for f in top
        ]

    # -- §25 controls ------------------------------------------------------
    def memory_view(self) -> dict:
        """'View memories' control."""
        return {
            "memory_enabled": self.memory_enabled,
            "facts": self.facts(),
            "study_notes": self.notes(limit=20),
            "bookmarks": self.bookmarks(),
        }

    def set_memory_enabled(self, enabled: bool) -> None:
        self.memory_enabled = enabled

    def clear_all(self) -> dict:
        """'Clear memories' control: wipes facts + notes + history + bookmarks."""
        cleared = {"facts": self.clear_facts()}
        with self._connect() as con:
            cleared["notes"] = con.execute("DELETE FROM study_notes").rowcount
            cleared["history"] = con.execute("DELETE FROM study_history").rowcount
            cleared["bookmarks"] = con.execute("DELETE FROM bookmarks").rowcount
        return cleared
