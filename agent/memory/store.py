"""Agent memory (agentodo.md §15).

Four SEPARATE memory types; user memory is never merged into authoritative
knowledge (§15: "Never mix user memory with authoritative knowledge"):

- profile: language/UI/study preferences
- study: topics/ayahs/hadiths studied, notes, bookmarks (persisted SQLite)
- conversation: short-lived, in-process only, capped
- knowledge: the canonical corpus (NOT stored here — that is the knowledge DB)

No chain-of-thought is ever stored: notes keep content only, decisions and
confidence where provided (§0).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "knowledge" / "processed" / "memory.db"

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS study_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    citation_id TEXT,
    note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS study_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    query TEXT NOT NULL,
    intent TEXT,
    citations TEXT  -- JSON array of citation_ids actually verified
);
CREATE TABLE IF NOT EXISTS bookmarks (
    citation_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
"""


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ConversationMemory:
    """Short-lived conversational context, in-process only (§15)."""

    max_turns: int = 12
    turns: list[ConversationTurn] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append(ConversationTurn(role, content))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def as_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def clear(self) -> None:
        self.turns = []


class MemoryStore:
    """Persisted profile + study memory. Separate DB from the knowledge store
    so user data can never leak into the corpus."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(MEMORY_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    # -- profile ----------------------------------------------------------
    def set_profile(self, key: str, value: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO profile VALUES (?,?)", (key, value)
            )

    def get_profile(self, key: str, default: str = "") -> str:
        with self._connect() as con:
            row = con.execute("SELECT value FROM profile WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    # -- study notes --------------------------------------------------------
    def save_note(self, note: str, citation_id: str | None = None) -> int:
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO study_notes (created_at, citation_id, note) VALUES (?,?,?)",
                (datetime.now(UTC).isoformat(), citation_id, note),
            )
            return cur.lastrowid

    def notes(self, limit: int = 20, citation_id: str | None = None) -> list[dict]:
        sql = "SELECT * FROM study_notes"
        params: list = []
        if citation_id:
            sql += " WHERE citation_id = ?"
            params.append(citation_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # -- study history ------------------------------------------------------
    def record_query(self, query: str, intent: str, citations: list[str]) -> int:
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO study_history (created_at, query, intent, citations) VALUES (?,?,?,?)",
                (
                    datetime.now(UTC).isoformat(),
                    query, intent, json.dumps(citations),
                ),
            )
            return cur.lastrowid

    def history(self, limit: int = 20) -> list[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM study_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["citations"] = json.loads(item["citations"] or "[]")
            out.append(item)
        return out

    # -- bookmarks -----------------------------------------------------------
    def bookmark(self, citation_id: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO bookmarks VALUES (?,?)",
                (citation_id, datetime.now(UTC).isoformat()),
            )

    def bookmarks(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM bookmarks ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
