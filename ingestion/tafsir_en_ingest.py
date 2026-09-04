"""Classic English tafsir ingestion from quran_campaign's extracted chunks.

agentodo.md §7, §6 (TIER 2). Source: knowledge/tafsir/raw/classic_tafsir_en_chunks.db
(quran_campaign extraction of born-digital PDFs: as-Sa'di, Ibn Kathir
abridged, al-Qurtubi). Only the `chunks` table is imported — the source DB's
own quran/translations tables are truncated and are NEVER ingested; Qur'an
text in this project remains the hash-verified quran-uthmani-json.

Chunk rows keep their original chunk_id (stable), carry scholar/work/volume/
page provenance, and are validated against the Uthmani ayah grid. Text with
out-of-range surah/ayah tags is a hard error, never silently dropped.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent.policy.source_policy import SourcePolicy, SourceRegistry
from ingestion.arabic_norm import search_form
from ingestion.quran_ingest import SCHEMA as QURAN_SCHEMA
from ingestion.quran_ingest import _registered_hash, sha256_of

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DB = REPO_ROOT / "knowledge" / "tafsir" / "raw" / "classic_tafsir_en_chunks.db"
DEFAULT_DB = REPO_ROOT / "knowledge" / "processed" / "knowledge.db"

# registry source_id -> quran_campaign source_id
SOURCE_MAP = {
    "tafsir-sadi-en": "tafsir_sadi_en_001",
    "tafsir-ibn-kathir-en": "tafsir_ibn_kathir_en_001",
    "tafsir-qurtubi-en": "tafsir_qurtubi_en_001",
}

TAFSIR_EN_SCHEMA = """
CREATE TABLE IF NOT EXISTS tafsir_en (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    scholar TEXT NOT NULL DEFAULT '',
    work TEXT NOT NULL DEFAULT '',
    volume TEXT NOT NULL DEFAULT '',
    page_start INTEGER,
    page_end INTEGER,
    surah INTEGER NOT NULL,
    ayah_start INTEGER NOT NULL,
    ayah_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_search TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tafsir_en_ayah ON tafsir_en (surah, ayah_start, ayah_end);
-- Quarantine: chunks whose source-pipeline ayah tags fail grid validation.
-- Never silently dropped OR silently retagged (§7): preserved with the reason,
-- excluded from retrieval until a human fixes the tag.
CREATE TABLE IF NOT EXISTS tafsir_en_quarantine (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    surah_tag INTEGER,
    ayah_start_tag INTEGER,
    ayah_end_tag INTEGER,
    text TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS tafsir_en_fts USING fts5(
    text_search,
    content='tafsir_en', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS te_ai AFTER INSERT ON tafsir_en BEGIN
    INSERT INTO tafsir_en_fts(rowid, text_search) VALUES (new.rowid, new.text_search);
END;
CREATE TRIGGER IF NOT EXISTS te_ad AFTER DELETE ON tafsir_en BEGIN
    INSERT INTO tafsir_en_fts(tafsir_en_fts, rowid, text_search)
    VALUES ('delete', old.rowid, old.text_search);
END;
"""


@dataclass
class TafsirEnIngestResult:
    source_id: str
    chunk_count: int
    run_id: str
    deterministic_replay: bool


class TafsirEnIngestor:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        source_db: Path = DEFAULT_SOURCE_DB,
        policy: SourcePolicy | None = None,
    ):
        self.db_path = db_path
        self.source_db = source_db
        self.policy = policy or SourcePolicy(SourceRegistry.load())

    def ingest_all(self) -> list[TafsirEnIngestResult]:
        return [self.ingest(sid) for sid in sorted(SOURCE_MAP)]

    def ingest(self, source_id: str) -> TafsirEnIngestResult:
        if source_id not in SOURCE_MAP:
            raise ValueError(f"unknown classic tafsir source '{source_id}'")
        record = self.policy.registry.get(source_id)
        self.policy.assert_ingestible(record)  # §5.2 gate before any write
        digest = sha256_of(self.source_db)
        if _registered_hash(record) and digest != _registered_hash(record):
            raise ValueError(
                f"source hash mismatch for {source_id}: got {digest}, "
                f"registry says {_registered_hash(record)}"
            )
        rows, scholar, work = self._read_source(source_id)
        run_id = f"ingest:{source_id}:{digest[:12]}"
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(QURAN_SCHEMA)
            con.executescript(TAFSIR_EN_SCHEMA)
            grid = {tuple(r) for r in con.execute("SELECT surah, ayah FROM quran")}
            if len(grid) != 6236:
                raise RuntimeError("Uthmani corpus not ingested — run QuranIngestor first")
            # Grid validation with quarantine: bad tags preserved, never
            # silently dropped or retagged (§7 quality validation).
            good_rows: list[tuple] = []
            quarantined: list[tuple] = []
            for r in rows:
                surah, a_start, a_end = r[7], r[8], r[9]
                if (
                    surah is None or a_start is None or a_end is None
                    or (surah, a_start) not in grid
                ):
                    reason = (
                        "null ayah tag (front matter/TOC)"
                        if surah is None or a_start is None or a_end is None
                        else f"ayah tag {surah}:{a_start} not in Uthmani grid "
                        "(source pipeline misparse)"
                    )
                    quarantined.append((
                        r[0], source_id, surah, a_start, a_end, r[10], reason,
                    ))
                else:
                    good_rows.append(r)
            replay_row = con.execute(
                "SELECT sha256, ayah_count FROM ingestion_log WHERE source_id=? "
                "ORDER BY run_id DESC LIMIT 1",
                (source_id,),
            ).fetchone()
            replay = bool(replay_row) and replay_row[0] == digest and replay_row[1] == len(rows)
            con.execute("DELETE FROM tafsir_en WHERE source_id=?", (source_id,))
            con.execute("DELETE FROM tafsir_en_quarantine WHERE source_id=?", (source_id,))
            con.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
                (source_id, record.title, record.author, record.type, record.language,
                 record.tradition, record.license, record.verification_status, digest),
            )
            con.executemany(
                "INSERT OR REPLACE INTO tafsir_en (chunk_id, source_id, scholar, work, "
                "volume, page_start, page_end, surah, ayah_start, ayah_end, text, "
                "text_search) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                good_rows,
            )
            con.executemany(
                "INSERT OR REPLACE INTO tafsir_en_quarantine VALUES (?,?,?,?,?,?,?)",
                quarantined,
            )
            con.execute(
                "INSERT OR REPLACE INTO ingestion_log VALUES (?,?,?,?,?)",
                (run_id, source_id, digest, len(rows), replay),
            )
            con.commit()
        finally:
            con.close()
        return TafsirEnIngestResult(source_id, len(good_rows), run_id, replay)

    def _read_source(self, source_id: str) -> tuple[list[tuple], str, str]:
        """Read + validate chunks from the quran_campaign DB (read-only)."""
        campaign_id = SOURCE_MAP[source_id]
        src = sqlite3.connect(f"file:{self.source_db}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        try:
            meta = src.execute(
                "SELECT scholar, work FROM sources WHERE source_id=?", (campaign_id,)
            ).fetchone()
            if not meta:
                raise ValueError(f"{campaign_id} missing from source DB sources table")
            rows: list[tuple] = []
            for r in src.execute(
                "SELECT chunk_id, document_id, volume, page_start, page_end, "
                "surah_number, ayah_start, ayah_end, text FROM chunks "
                "WHERE source_id=? ORDER BY chunk_id",
                (campaign_id,),
            ):
                text = (r["text"] or "").strip()
                if not text:
                    continue  # 3 empty chunks exist; skip rather than invent
                # invalid/null tags are NOT dropped here — grid validation in
                # ingest() quarantines them with reasons (§7)
                rows.append((
                    r["chunk_id"], source_id, meta["scholar"], meta["work"],
                    r["volume"] or "", r["page_start"], r["page_end"],
                    r["surah_number"], r["ayah_start"], r["ayah_end"],
                    text, search_form(text),
                ))
        finally:
            src.close()
        return rows, meta["scholar"], meta["work"]


class TafsirEnStore:
    """Read-side access to classic English tafsir chunks (TIER 2)."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def chunk_count(self, source_id: str | None = None) -> int:
        with self._connect() as con:
            if source_id:
                return con.execute(
                    "SELECT COUNT(*) FROM tafsir_en WHERE source_id=?", (source_id,)
                ).fetchone()[0]
            return con.execute("SELECT COUNT(*) FROM tafsir_en").fetchone()[0]

    def get_chunk(self, chunk_id: str) -> dict | None:
        """One chunk by its stable id (vector-hit resolution)."""
        with self._connect() as con:
            row = con.execute(
                "SELECT chunk_id, source_id, scholar, work, volume, page_start, "
                "surah, ayah_start, ayah_end, text FROM tafsir_en WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_for_ayah(self, surah: int, ayah: int, limit: int = 6) -> list[dict]:
        """All classic tafsir commentary covering a specific ayah."""
        with self._connect() as con:
            rows = con.execute(
                "SELECT chunk_id, source_id, scholar, work, volume, page_start, "
                "surah, ayah_start, ayah_end, text FROM tafsir_en "
                "WHERE surah=? AND ayah_start<=? AND ayah_end>=? "
                "ORDER BY source_id, ayah_start LIMIT ?",
                (surah, ayah, ayah, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_fts(self, query: str, source_id: str | None = None, limit: int = 10) -> list[dict]:
        from ingestion.quran_ingest import _content_words_exist, _fts_escape

        sql = (
            "SELECT c.chunk_id, c.source_id, c.scholar, c.surah, c.ayah_start, "
            "c.ayah_end, c.text, bm25(tafsir_en_fts) AS rank "
            "FROM tafsir_en_fts f JOIN tafsir_en c ON c.rowid = f.rowid "
            "WHERE tafsir_en_fts MATCH ? "
        )
        if source_id:
            sql += "AND c.source_id = ? "
        sql += "ORDER BY rank LIMIT ?"
        with self._connect() as con:
            escaped = _fts_escape(query)
            rows = con.execute(sql, (escaped, source_id, limit) if source_id
                               else (escaped, limit)).fetchall()
            if not rows and " " in escaped.strip() and _content_words_exist(
                con, "tafsir_en_fts", escaped
            ):
                or_query = escaped.replace('" ', '" OR ')
                rows = con.execute(sql, (or_query, source_id, limit) if source_id
                                   else (or_query, limit)).fetchall()
        return [dict(r) for r in rows]
