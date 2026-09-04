"""Kutub al-Sittah ingestion (agentodo.md §7, §13, §26 Phase 2).

Ingests the six canonical Sunni hadith collections from the approved
fawazahmed0/hadith-api dataset: Arabic + English per hadith, with grading
metadata preserved verbatim (§6: hadith authenticity metadata must be
preserved; never let the LLM manufacture gradings, §13).

Citation IDs are stable: hadith:<collection>:<hadithnumber>. The Arabic and
English editions are aligned by hadithnumber (validated at ingest); a
misalignment is a hard error, never a silent shift.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent.policy.source_policy import SourcePolicy, SourceRegistry
from ingestion.arabic_norm import search_form
from ingestion.quran_ingest import (
    SCHEMA as _QURAN_SCHEMA,
)
from ingestion.quran_ingest import (
    sha256_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "knowledge" / "hadith" / "raw"
DEFAULT_DB = REPO_ROOT / "knowledge" / "processed" / "knowledge.db"

# registry source_id -> (dataset book key, display name, tier-1 collection)
KUTUB_AL_SITTAH = {
    "sahih-bukhari": ("bukhari", "Sahih al-Bukhari"),
    "sahih-muslim": ("muslim", "Sahih Muslim"),
    "sunan-abu-dawud": ("abudawud", "Sunan Abu Dawud"),
    "jami-at-tirmidhi": ("tirmidhi", "Jami' at-Tirmidhi"),
    "sunan-an-nasai": ("nasai", "Sunan an-Nasa'i"),
    "sunan-ibn-majah": ("ibnmajah", "Sunan Ibn Majah"),
}

HADITH_SCHEMA = """
CREATE TABLE IF NOT EXISTS hadith_collections (
    source_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    book_key TEXT NOT NULL,
    hadith_count INTEGER NOT NULL,
    grading_basis TEXT NOT NULL,
    sha256_ara TEXT,
    sha256_eng TEXT
);
CREATE TABLE IF NOT EXISTS hadith (
    citation_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES hadith_collections(source_id),
    hadithnumber INTEGER NOT NULL,
    arabicnumber INTEGER,
    book_number INTEGER,
    book_hadith_number INTEGER,
    arabic TEXT NOT NULL,
    arabic_search TEXT NOT NULL,
    english TEXT,
    english_search TEXT,
    grades_json TEXT,
    UNIQUE (source_id, hadithnumber)
);
CREATE VIRTUAL TABLE IF NOT EXISTS hadith_fts USING fts5(
    arabic_search,
    english_search,
    content='hadith', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS h_ai AFTER INSERT ON hadith BEGIN
    INSERT INTO hadith_fts(rowid, arabic_search, english_search)
    VALUES (new.rowid, new.arabic_search, COALESCE(new.english_search,''));
END;
CREATE TRIGGER IF NOT EXISTS h_ad AFTER DELETE ON hadith BEGIN
    INSERT INTO hadith_fts(hadith_fts, rowid, arabic_search, english_search)
    VALUES ('delete', old.rowid, old.arabic_search, COALESCE(old.english_search,''));
END;
"""


def hadith_citation_id(source_id: str, hadithnumber: int) -> str:
    return f"hadith:{source_id}:{hadithnumber}"


@dataclass
class HadithIngestResult:
    source_id: str
    hadith_count: int
    run_id: str
    deterministic_replay: bool


class HadithIngestor:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        raw_dir: Path = RAW_DIR,
        policy: SourcePolicy | None = None,
    ):
        self.db_path = db_path
        self.raw_dir = raw_dir
        self.policy = policy or SourcePolicy(SourceRegistry.load())

    def ingest_all(self) -> list[HadithIngestResult]:
        return [self.ingest(source_id) for source_id in sorted(KUTUB_AL_SITTAH)]

    def ingest(self, source_id: str) -> HadithIngestResult:
        book_key, title = KUTUB_AL_SITTAH[source_id]
        record = self.policy.registry.get(source_id)
        self.policy.assert_ingestible(record)  # §5.2 gate before any write
        ara_path = self.raw_dir / f"hadith-ara-{book_key}.json"
        eng_path = self.raw_dir / f"hadith-eng-{book_key}.json"
        digest_ara = sha256_of(ara_path)
        digest_eng = sha256_of(eng_path)
        # hash pinning: both files must match the registry notes
        registered = _registered_hash_multi(record)
        for file_digest in (digest_ara, digest_eng):
            if registered and file_digest not in registered:
                raise ValueError(
                    f"source hash mismatch for {source_id}: got {file_digest}, "
                    f"registry pins {registered}"
                )
        ara = json.loads(ara_path.read_text(encoding="utf-8"))
        eng = json.loads(eng_path.read_text(encoding="utf-8"))
        ara_h, eng_h = ara["hadiths"], eng["hadiths"]
        if len(ara_h) != len(eng_h):
            raise ValueError(f"{source_id}: AR/EN hadith count mismatch {len(ara_h)} vs {len(eng_h)}")
        rows: list[tuple] = []
        for a, e in zip(ara_h, eng_h):
            if a["hadithnumber"] != e["hadithnumber"]:
                raise ValueError(
                    f"{source_id}: numbering misaligned at AR {a['hadithnumber']} vs EN {e['hadithnumber']}"
                )
            arabic = (a.get("text") or "").strip()
            if not arabic:
                continue  # dataset has a few empty-side entries; skip, never invent
            english = (e.get("text") or "").strip()
            grades = e.get("grades") or []
            ref = e.get("reference") or {}
            rows.append((
                hadith_citation_id(source_id, a["hadithnumber"]),
                source_id,
                a["hadithnumber"],
                a.get("arabicnumber"),
                ref.get("book"),
                ref.get("hadith"),
                arabic,
                search_form(arabic),
                english or None,
                search_form(english) if english else None,
                json.dumps(grades, ensure_ascii=False) if grades else None,
            ))
        # grading basis: per-hadith grades present, or collection-level sahih
        n_graded = sum(1 for r in rows if r[10])
        basis = (
            "per-hadith (dataset carries grader metadata)"
            if n_graded == len(rows)
            else "collection-level sahih (Sahih collection; no per-hadith grades in dataset)"
            if source_id in ("sahih-bukhari", "sahih-muslim")
            else "mixed (dataset grades present for most, not all)"
        )
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(_QURAN_SCHEMA)
            con.executescript(HADITH_SCHEMA)
            replay_row = con.execute(
                "SELECT hadith_count FROM hadith_collections WHERE source_id=?", (source_id,)
            ).fetchone()
            replay = bool(replay_row) and replay_row[0] == len(rows)
            con.execute("DELETE FROM hadith WHERE source_id=?", (source_id,))
            con.execute(
                "INSERT OR REPLACE INTO hadith_collections VALUES (?,?,?,?,?,?,?,?)",
                (source_id, title, record.author, book_key, len(rows), basis,
                 digest_ara, digest_eng),
            )
            con.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
                (source_id, record.title, record.author, record.type, record.language,
                 record.tradition, record.license, record.verification_status, digest_ara),
            )
            con.executemany(
                "INSERT OR REPLACE INTO hadith (citation_id, source_id, hadithnumber, arabicnumber, "
                "book_number, book_hadith_number, arabic, arabic_search, english, english_search, "
                "grades_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            run_id = f"ingest:{source_id}:{digest_ara[:12]}"
            con.execute(
                "INSERT OR REPLACE INTO ingestion_log VALUES (?,?,?,?,?)",
                (run_id, source_id, digest_ara, len(rows), replay),
            )
            con.commit()
        finally:
            con.close()
        return HadithIngestResult(source_id, len(rows), run_id, replay)


def _registered_hash_multi(record) -> set[str]:
    """All 64-hex sha256 pins in a registry record's notes.

    YAML folding may merge the notes onto one line or keep per-line layout;
    scan the whole blob so either shape yields every pin.
    """
    import re

    blob = record.notes or ""
    return set(re.findall(r"sha256=([0-9a-f]{64})", blob))


class HadithStore:
    """Read-side access with preserved grading metadata (§6, §13)."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def get_hadith(self, source_id: str, hadithnumber: int) -> dict | None:
        citation = hadith_citation_id(source_id, hadithnumber)
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM hadith WHERE citation_id=?", (citation,)
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["grades"] = json.loads(item.pop("grades_json") or "[]")
        return item

    def collections(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM hadith_collections ORDER BY title").fetchall()
        return [dict(r) for r in rows]

    def hadith_count(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM hadith").fetchone()[0]

    def search_fts(self, query: str, source_id: str | None = None, limit: int = 12) -> list[dict]:
        """FTS across both Arabic and English hadith text. Query routed by script."""
        from ingestion.quran_ingest import _content_words_exist, _fts_escape

        con = self._connect()
        try:
            arabic_query = any(0x0600 <= ord(c) <= 0x06FF for c in query)
            if arabic_query:
                escaped = _fts_escape(search_form(query))
            else:
                escaped = _fts_escape(query)
            sql = (
                "SELECT h.citation_id, h.source_id, h.hadithnumber, h.arabic, h.english, "
                "h.grades_json, h.book_number, h.book_hadith_number, "
                "bm25(hadith_fts) AS rank FROM hadith_fts f JOIN hadith h ON h.rowid = f.rowid "
                "WHERE hadith_fts MATCH ? "
            )
            params: list = [escaped]
            if source_id:
                sql += "AND h.source_id = ? "
                params.append(source_id)
            sql += "ORDER BY rank LIMIT ?"
            params.append(limit)
            rows = con.execute(sql, params).fetchall()
            if not rows and " " in escaped.strip() and _content_words_exist(
                con, "hadith_fts", escaped
            ):
                or_query = escaped.replace('" ', '" OR ')
                params[0] = or_query
                rows = con.execute(sql, params).fetchall()
        finally:
            con.close()
        out = []
        for r in rows:
            item = dict(r)
            item["grades"] = json.loads(item.pop("grades_json") or "[]")
            out.append(item)
        return out
