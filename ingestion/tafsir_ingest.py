"""Kutub tafsir ingestion: Kemenag per-ayah tafsir + Indonesian translation.

agentodo.md §7 (ingestion), §6 (tiers): tafsir is TIER 2 — interpretation,
NEVER Qur'an text. Tafsir rows attach to (surah, ayah) of the hash-verified
Uthmani corpus; the Arabic inside the Kemenag files is NOT served and NOT
ingested (wrong script variant).

Citation IDs: tafsir:<source_id>:<surah>:<ayah>, stable.
"""

from __future__ import annotations

import json
import sqlite3
import tarfile
from dataclasses import dataclass
from pathlib import Path

from agent.policy.source_policy import SourcePolicy, SourceRegistry
from ingestion.arabic_norm import search_form
from ingestion.quran_ingest import (
    _registered_hash,
    sha256_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ARCHIVE = REPO_ROOT / "knowledge" / "tafsir" / "raw" / "kemenag-tafsir.json.tar.gz"
DEFAULT_DB = REPO_ROOT / "knowledge" / "processed" / "knowledge.db"

TAFSIR_SOURCE_ID = "tafsir-kemenag"
TRANSLATION_SOURCE_ID = "translation-kemenag-id"

TAFSIR_SCHEMA = """
CREATE TABLE IF NOT EXISTS tafsir (
    citation_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    tafsir TEXT NOT NULL,
    tafsir_search TEXT NOT NULL,
    UNIQUE (source_id, surah, ayah)
);
CREATE VIRTUAL TABLE IF NOT EXISTS tafsir_fts USING fts5(
    tafsir_search,
    content='tafsir', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS t_ai AFTER INSERT ON tafsir BEGIN
    INSERT INTO tafsir_fts(rowid, tafsir_search) VALUES (new.rowid, new.tafsir_search);
END;
CREATE TRIGGER IF NOT EXISTS t_ad AFTER DELETE ON tafsir BEGIN
    INSERT INTO tafsir_fts(tafsir_fts, rowid, tafsir_search)
    VALUES ('delete', old.rowid, old.tafsir_search);
END;
"""


def tafsir_citation_id(source_id: str, surah: int, ayah: int) -> str:
    return f"tafsir:{source_id}:{surah}:{ayah}"


@dataclass
class TafsirIngestResult:
    run_id: str
    tafsir_count: int
    translation_count: int
    sha256: str
    deterministic_replay: bool


class TafsirIngestor:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        raw_archive: Path = RAW_ARCHIVE,
        policy: SourcePolicy | None = None,
    ):
        self.db_path = db_path
        self.raw_archive = raw_archive
        self.policy = policy or SourcePolicy(SourceRegistry.load())

    def ingest(self) -> TafsirIngestResult:
        # §5.2 gate for BOTH registry entries before any write
        tafsir_record = self.policy.registry.get(TAFSIR_SOURCE_ID)
        translation_record = self.policy.registry.get(TRANSLATION_SOURCE_ID)
        self.policy.assert_ingestible(tafsir_record)
        self.policy.assert_ingestible(translation_record)
        # hash pinning (both entries pin the same archive hash)
        digest = sha256_of(self.raw_archive)
        for record in (tafsir_record, translation_record):
            if _registered_hash(record) and digest != _registered_hash(record):
                raise ValueError(
                    f"source hash mismatch for {record.id}: got {digest}, "
                    f"registry says {_registered_hash(record)}"
                )
        # extract + load all 114 surah files
        tafsir_rows: list[tuple] = []
        translation_rows: list[tuple] = []
        with tarfile.open(self.raw_archive, "r:gz") as tar:
            for member in sorted(tar.getmembers(), key=lambda m: m.name):
                if not member.name.endswith(".json"):
                    continue
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                data = json.loads(fh.read().decode("utf-8"))
                surah_num = int(Path(member.name).stem)
                surah = data.get(str(surah_num))
                if not surah:
                    raise ValueError(f"{member.name}: missing surah key {surah_num}")
                n_ayah = int(surah["number_of_ayah"])
                tafsir_text = surah["tafsir"]["id"]["kemenag"]["text"]
                trans_text = surah["translations"]["id"]["text"]
                if len(tafsir_text) != n_ayah or len(trans_text) != n_ayah:
                    raise ValueError(
                        f"surah {surah_num}: tafsir/translation count mismatch "
                        f"({len(tafsir_text)}/{len(trans_text)} vs {n_ayah})"
                    )
                for ayah_str, text in tafsir_text.items():
                    ayah = int(ayah_str)
                    body = (text or "").strip()
                    if not body:
                        raise ValueError(f"empty tafsir at {surah_num}:{ayah}")
                    tafsir_rows.append((
                        tafsir_citation_id(TAFSIR_SOURCE_ID, surah_num, ayah),
                        TAFSIR_SOURCE_ID, surah_num, ayah, body, search_form(body),
                    ))
                for ayah_str, text in trans_text.items():
                    ayah = int(ayah_str)
                    body = (text or "").strip()
                    if not body:
                        raise ValueError(f"empty translation at {surah_num}:{ayah}")
                    translation_rows.append((surah_num, ayah, body, search_form(body)))
        if len(tafsir_rows) != 6236 or len(translation_rows) != 6236:
            raise ValueError(
                f"expected 6236 tafsir/translation rows, got {len(tafsir_rows)}/{len(translation_rows)}"
            )
        # every tafsir row must attach to an ingested Uthmani ayah
        con = sqlite3.connect(self.db_path)
        try:
            from ingestion.quran_ingest import SCHEMA as QURAN_SCHEMA
            con.executescript(QURAN_SCHEMA)
            con.executescript(TAFSIR_SCHEMA)
            quran_grid = {
                tuple(r) for r in con.execute("SELECT surah, ayah FROM quran")
            }
            if len(quran_grid) != 6236:
                raise RuntimeError("Uthmani corpus not ingested — run QuranIngestor first")
            for row in tafsir_rows:
                if (row[2], row[3]) not in quran_grid:
                    raise ValueError(f"tafsir row {row[2]}:{row[3]} has no Qur'an counterpart")
            replay_row = con.execute(
                "SELECT sha256, ayah_count FROM ingestion_log WHERE source_id=? "
                "ORDER BY run_id DESC LIMIT 1",
                (TAFSIR_SOURCE_ID,),
            ).fetchone()
            replay = bool(replay_row) and replay_row[0] == digest and replay_row[1] == len(tafsir_rows)
            con.execute("DELETE FROM tafsir WHERE source_id=?", (TAFSIR_SOURCE_ID,))
            for source_id, record in (
                (TAFSIR_SOURCE_ID, tafsir_record),
                (TRANSLATION_SOURCE_ID, translation_record),
            ):
                con.execute(
                    "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
                    (source_id, record.title, record.author, record.type, record.language,
                     record.tradition, record.license, record.verification_status, digest),
                )
            con.executemany(
                "INSERT OR REPLACE INTO tafsir (citation_id, source_id, surah, ayah, tafsir, "
                "tafsir_search) VALUES (?,?,?,?,?,?)",
                tafsir_rows,
            )
            # Indonesian translation rides the existing quran_translations table
            con.execute(
                "DELETE FROM quran_translations WHERE source_id=?", (TRANSLATION_SOURCE_ID,)
            )
            con.executemany(
                "INSERT OR REPLACE INTO quran_translations (surah, ayah, lang, translation, "
                "search, source_id) VALUES (?,?,?,?,?,?)",
                [(s, a, "id", t, sf, TRANSLATION_SOURCE_ID) for (s, a, t, sf) in translation_rows],
            )
            run_id = f"ingest:{TAFSIR_SOURCE_ID}:{digest[:12]}"
            con.execute(
                "INSERT OR REPLACE INTO ingestion_log VALUES (?,?,?,?,?)",
                (run_id, TAFSIR_SOURCE_ID, digest, len(tafsir_rows), replay),
            )
            con.commit()
        finally:
            con.close()
        return TafsirIngestResult(run_id, len(tafsir_rows), len(translation_rows), digest, replay)


class TafsirStore:
    """Read-side access to TIER 2 tafsir with per-ayah provenance."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def get_tafsir(self, surah: int, ayah: int, source_id: str = TAFSIR_SOURCE_ID) -> dict | None:
        citation = tafsir_citation_id(source_id, surah, ayah)
        with self._connect() as con:
            row = con.execute("SELECT * FROM tafsir WHERE citation_id=?", (citation,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item.pop("tafsir_search", None)
        return item

    def tafsir_count(self, source_id: str = TAFSIR_SOURCE_ID) -> int:
        with self._connect() as con:
            return con.execute(
                "SELECT COUNT(*) FROM tafsir WHERE source_id=?", (source_id,)
            ).fetchone()[0]

    def search_fts(self, query: str, source_id: str = TAFSIR_SOURCE_ID, limit: int = 10) -> list[dict]:
        """FTS over tafsir body text (Indonesian)."""
        from ingestion.quran_ingest import _content_words_exist, _fts_escape

        sql = (
            "SELECT t.citation_id, t.source_id, t.surah, t.ayah, t.tafsir, "
            "bm25(tafsir_fts) AS rank FROM tafsir_fts f "
            "JOIN tafsir t ON t.rowid = f.rowid "
            "WHERE tafsir_fts MATCH ? AND t.source_id = ? ORDER BY rank LIMIT ?"
        )
        with self._connect() as con:
            escaped = _fts_escape(query)
            rows = con.execute(sql, (escaped, source_id, limit)).fetchall()
            if not rows and " " in escaped.strip() and _content_words_exist(
                con, "tafsir_fts", escaped
            ):
                rows = con.execute(sql, (escaped.replace('" ', '" OR '), source_id, limit)).fetchall()
        return [dict(r) for r in rows]
