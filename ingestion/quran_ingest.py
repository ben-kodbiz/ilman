"""Qur'an ingestion (agentodo.md §7, §14, §26 Phase 2).

Deterministic, repeatable ingestion of the approved Uthmani Qur'an dataset
into SQLite + FTS5 with stable IDs (`quran:<surah>:<ayah>`) and per-ayah
provenance. The source-policy gate MUST pass before any row is written:
ingestion is blocked otherwise (§5.2 — never ingest first, filter later).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent.policy.source_policy import SourcePolicy, SourceRegistry
from agent.tools.quran_refs import normalize_reference
from ingestion.arabic_norm import search_form

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "knowledge" / "quran" / "quran-uthmani-raw.json"
DEFAULT_EN_RAW = REPO_ROOT / "knowledge" / "quran" / "quran-en-translation-raw.json"
DEFAULT_DB = REPO_ROOT / "knowledge" / "processed" / "knowledge.db"
QURAN_SOURCE_ID = "quran-uthmani-json"
QURAN_EN_SOURCE_ID = "quran-en-saheeh-json"
EXPECTED_AYAHS = 6236

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    type TEXT NOT NULL,
    language TEXT,
    tradition TEXT NOT NULL,
    license TEXT,
    verification_status TEXT,
    sha256 TEXT
);
CREATE TABLE IF NOT EXISTS quran (
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    arabic TEXT NOT NULL,
    search TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    PRIMARY KEY (surah, ayah)
);
CREATE VIRTUAL TABLE IF NOT EXISTS quran_fts USING fts5(
    search,
    content='quran', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS quran_ai AFTER INSERT ON quran BEGIN
    INSERT INTO quran_fts(rowid, search) VALUES (new.rowid, new.search);
END;
CREATE TRIGGER IF NOT EXISTS quran_ad AFTER DELETE ON quran BEGIN
    INSERT INTO quran_fts(quran_fts, rowid, search) VALUES ('delete', old.rowid, old.search);
END;
CREATE TRIGGER IF NOT EXISTS quran_au AFTER UPDATE ON quran BEGIN
    INSERT INTO quran_fts(quran_fts, rowid, search) VALUES ('delete', old.rowid, old.search);
    INSERT INTO quran_fts(rowid, search) VALUES (new.rowid, new.search);
END;
CREATE TABLE IF NOT EXISTS quran_translations (
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    lang TEXT NOT NULL,
    translation TEXT NOT NULL,
    search TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    PRIMARY KEY (surah, ayah, lang)
);
CREATE VIRTUAL TABLE IF NOT EXISTS quran_translations_fts USING fts5(
    search,
    content='quran_translations', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS qt_ai AFTER INSERT ON quran_translations BEGIN
    INSERT INTO quran_translations_fts(rowid, search) VALUES (new.rowid, new.search);
END;
CREATE TRIGGER IF NOT EXISTS qt_ad AFTER DELETE ON quran_translations BEGIN
    INSERT INTO quran_translations_fts(quran_translations_fts, rowid, search)
    VALUES ('delete', old.rowid, old.search);
END;
CREATE TABLE IF NOT EXISTS ingestion_log (
    run_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    ayah_count INTEGER NOT NULL,
    deterministic_replay BOOLEAN NOT NULL
);
"""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def citation_id(surah: int, ayah: int) -> str:
    """Stable citation ID (§7): quran:<surah>:<ayah>."""
    return f"quran:{surah}:{ayah}"


@dataclass
class QuranIngestResult:
    run_id: str
    ayah_count: int
    sha256: str
    deterministic_replay: bool


class QuranIngestor:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        raw_path: Path = DEFAULT_RAW,
        policy: SourcePolicy | None = None,
    ):
        self.db_path = db_path
        self.raw_path = raw_path
        self.policy = policy or SourcePolicy(SourceRegistry.load())

    def ingest(self) -> QuranIngestResult:
        """Gate -> hash -> load -> validate -> write. Raises on any failure."""
        # 1. Source gate: registry says this exact dataset is approved (§5.2).
        record = self.policy.registry.get(QURAN_SOURCE_ID)
        self.policy.assert_ingestible(record)
        # 2. Hash the actual file: bytes on disk must match the registered hash.
        digest = sha256_of(self.raw_path)
        registered_hash = _registered_hash(record)
        if registered_hash and digest != registered_hash:
            raise ValueError(
                f"source hash mismatch for {self.raw_path}: got {digest}, registry says {registered_hash}"
            )
        # 3. Load + structural validation (deterministic checks, §7).
        chapters = _load_and_validate_chapters(self.raw_path)
        rows: list[tuple[int, int, str, str]] = []
        for surah in sorted(chapters):
            for v in chapters[surah]:
                text = v["text"].strip()
                if not text:
                    raise ValueError(f"empty ayah text at {surah}:{v['verse']}")
                rows.append((surah, v["verse"], text, search_form(text)))
        if len(rows) != EXPECTED_AYAHS:
            raise ValueError(f"expected {EXPECTED_AYAHS} ayahs, got {len(rows)}")
        # 4. Write (idempotent full replay).
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(SCHEMA)
            replay = self._was_replayed(con, QURAN_SOURCE_ID, digest, len(rows))
            con.execute("DELETE FROM quran WHERE source_id = ?", (QURAN_SOURCE_ID,))
            con.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    QURAN_SOURCE_ID, record.title, record.author, record.type,
                    record.language, record.tradition, record.license,
                    record.verification_status, digest,
                ),
            )
            con.executemany(
                "INSERT INTO quran (surah, ayah, arabic, search, source_id) VALUES (?,?,?,?,?)",
                [(s, a, t, sf, QURAN_SOURCE_ID) for (s, a, t, sf) in rows],
            )
            run_id = f"ingest:{QURAN_SOURCE_ID}:{digest[:12]}"
            con.execute(
                "INSERT OR REPLACE INTO ingestion_log VALUES (?,?,?,?,?)",
                (run_id, QURAN_SOURCE_ID, digest, len(rows), replay),
            )
            con.commit()
        finally:
            con.close()
        return QuranIngestResult(run_id, len(rows), digest, replay)

    def _was_replayed(self, con: sqlite3.Connection, source_id: str, digest: str, count: int) -> bool:
        row = con.execute(
            "SELECT sha256, ayah_count FROM ingestion_log WHERE source_id = ? ORDER BY run_id DESC LIMIT 1",
            (source_id,),
        ).fetchone()
        return bool(row) and row[0] == digest and row[1] == count


def _registered_hash(record) -> str | None:
    """Extract the 64-hex sha256 from a registry record's notes."""
    for line in (record.notes or "").splitlines():
        if "sha256=" in line:
            candidate = line.split("sha256=", 1)[1].strip().split()
            candidate = candidate[0].rstrip(".,;:") if candidate else ""
            if len(candidate) == 64 and all(c in "0123456789abcdef" for c in candidate):
                return candidate
    return None


def _load_and_validate_chapters(raw_path: Path) -> dict[int, list[dict]]:
    """Load a quran-json shaped file and validate structure deterministically."""
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    chapters = {int(k): v for k, v in data.items()}
    if sorted(chapters) != list(range(1, 115)):
        raise ValueError("dataset must contain exactly surahs 1-114")
    for surah, verses in chapters.items():
        if [v["verse"] for v in verses] != list(range(1, len(verses) + 1)):
            raise ValueError(f"non-contiguous verse numbers in surah {surah}")
    return chapters


class TranslationIngestor:
    """Ingests a per-ayah translation aligned with the ingested Uthmani text.

    The gate, hash pinning, and structural validation mirror QuranIngestor.
    Additionally the verse grid MUST align exactly with the Arabic corpus —
    a translation is stored as an interpretation of a specific ayah, so any
    misalignment is a hard error, never a silent shift.
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        raw_path: Path = DEFAULT_EN_RAW,
        lang: str = "en",
        source_id: str = QURAN_EN_SOURCE_ID,
        policy: SourcePolicy | None = None,
    ):
        self.db_path = db_path
        self.raw_path = raw_path
        self.lang = lang
        self.source_id = source_id
        self.policy = policy or SourcePolicy(SourceRegistry.load())

    def ingest(self) -> QuranIngestResult:
        record = self.policy.registry.get(self.source_id)
        self.policy.assert_ingestible(record)  # §5.2 gate before any write
        digest = sha256_of(self.raw_path)
        registered = _registered_hash(record)
        if registered and digest != registered:
            raise ValueError(
                f"source hash mismatch for {self.raw_path}: got {digest}, registry says {registered}"
            )
        chapters = _load_and_validate_chapters(self.raw_path)
        # Alignment with the ingested Arabic corpus is mandatory.
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            arabic_rows = con.execute("SELECT surah, ayah FROM quran").fetchall()
        if not arabic_rows:
            raise RuntimeError("Uthmani corpus not ingested yet — run QuranIngestor first")
        arabic_grid = {(r["surah"], r["ayah"]) for r in arabic_rows}
        rows: list[tuple[int, int, str, str]] = []
        for surah in sorted(chapters):
            for v in chapters[surah]:
                key = (surah, v["verse"])
                if key not in arabic_grid:
                    raise ValueError(f"translation verse {surah}:{v['verse']} has no Arabic counterpart")
                text = v["text"].strip()
                if not text:
                    raise ValueError(f"empty translation at {surah}:{v['verse']}")
                rows.append((surah, v["verse"], text, search_form(text)))
        if len(rows) != len(arabic_grid):
            raise ValueError(f"translation covers {len(rows)} ayahs, corpus has {len(arabic_grid)}")
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(SCHEMA)
            replay = self._was_replayed(con, digest, len(rows))
            con.execute("DELETE FROM quran_translations WHERE source_id = ?", (self.source_id,))
            con.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    self.source_id, record.title, record.author, record.type,
                    record.language, record.tradition, record.license,
                    record.verification_status, digest,
                ),
            )
            con.executemany(
                "INSERT INTO quran_translations (surah, ayah, lang, translation, search, source_id) "
                "VALUES (?,?,?,?,?,?)",
                [(s, a, self.lang, t, sf, self.source_id) for (s, a, t, sf) in rows],
            )
            run_id = f"ingest:{self.source_id}:{digest[:12]}"
            con.execute(
                "INSERT OR REPLACE INTO ingestion_log VALUES (?,?,?,?,?)",
                (run_id, self.source_id, digest, len(rows), replay),
            )
            con.commit()
        finally:
            con.close()
        return QuranIngestResult(run_id, len(rows), digest, replay)

    def _was_replayed(self, con: sqlite3.Connection, digest: str, count: int) -> bool:
        row = con.execute(
            "SELECT sha256, ayah_count FROM ingestion_log WHERE source_id = ? ORDER BY run_id DESC LIMIT 1",
            (self.source_id,),
        ).fetchone()
        return bool(row) and row[0] == digest and row[1] == count


class QuranStore:
    """Read-side access with deterministic provenance (§14)."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def get_ayah(self, surah: int, ayah: int, lang: str | None = None) -> dict | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT surah, ayah, arabic, source_id FROM quran WHERE surah=? AND ayah=?",
                (surah, ayah),
            ).fetchone()
            translation = None
            if row is not None and lang:
                trow = con.execute(
                    "SELECT translation, source_id AS translation_source_id FROM quran_translations "
                    "WHERE surah=? AND ayah=? AND lang=?",
                    (surah, ayah, lang),
                ).fetchone()
                translation = dict(trow) if trow else None
        if not row:
            return None
        item = dict(row)
        item["citation_id"] = citation_id(item["surah"], item["ayah"])
        if translation:
            item["translation"] = translation["translation"]
            item["translation_source_id"] = translation["translation_source_id"]
        return item

    def get_by_reference(self, reference: str) -> dict | None:
        """Deterministic: '2:255', 'Al-Baqarah 255', 'Ayat al-Kursi' -> row."""
        try:
            ref = normalize_reference(reference)
        except ValueError:
            return None
        return self.get_ayah(ref["surah"], ref["ayah"])

    def surah_ayah_count(self, surah: int) -> int:
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM quran WHERE surah=?", (surah,)
            ).fetchone()
        return row[0]

    def search_fts(self, query: str, limit: int = 20) -> list[dict]:
        """Arabic-corpus FTS (§8). AND first; OR fallback only when a content
        word actually exists in the index (else OR matches pure noise)."""
        sql = (
            "SELECT q.surah AS surah, q.ayah AS ayah, q.arabic AS arabic, "
            "q.source_id AS source_id, bm25(quran_fts) AS rank "
            "FROM quran_fts JOIN quran AS q ON q.rowid = quran_fts.rowid "
            "WHERE quran_fts MATCH ? ORDER BY rank LIMIT ?"
        )
        with self._connect() as con:
            escaped = _fts_escape(search_form(query))
            rows = con.execute(sql, (escaped, limit)).fetchall()
            if not rows and " " in escaped.strip() and _content_words_exist(con, "quran_fts", escaped):
                rows = con.execute(sql, (escaped.replace('" ', '" OR '), limit)).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["citation_id"] = citation_id(r["surah"], r["ayah"])
            out.append(item)
        return out

    def search_translation_fts(self, query: str, lang: str = "en", limit: int = 20) -> list[dict]:
        """FTS over translations; returns Arabic rows joined with the match text."""
        sql = (
            "SELECT q.surah AS surah, q.ayah AS ayah, q.arabic AS arabic, "
            "q.source_id AS source_id, t.translation AS translation, "
            "t.source_id AS translation_source_id, bm25(quran_translations_fts) AS rank "
            "FROM quran_translations_fts f "
            "JOIN quran_translations t ON t.rowid = f.rowid "
            "JOIN quran q ON q.surah = t.surah AND q.ayah = t.ayah "
            "WHERE quran_translations_fts MATCH ? AND t.lang = ? ORDER BY rank LIMIT ?"
        )
        with self._connect() as con:
            escaped = _fts_escape(query)
            rows = con.execute(sql, (escaped, lang, limit)).fetchall()
            if not rows and " " in escaped.strip() and _content_words_exist(
                con, "quran_translations_fts", escaped
            ):
                rows = con.execute(sql, (escaped.replace('" ', '" OR '), lang, limit)).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["citation_id"] = citation_id(r["surah"], r["ayah"])
            out.append(item)
        return out

    def ayah_count_total(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM quran").fetchone()[0]

    def translation_count(self, lang: str = "en") -> int:
        with self._connect() as con:
            return con.execute(
                "SELECT COUNT(*) FROM quran_translations WHERE lang=?", (lang,)
            ).fetchone()[0]


def _fts_escape(query: str) -> str:
    """Escape user input into a safe FTS5 query (quoted phrase per token)."""
    tokens = [t for t in query.split() if t.strip()]
    if not tokens:
        return '""'
    return " ".join(f'"{t.replace(chr(34), "")}"' for t in tokens)


_EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "with", "for", "as", "by", "at", "from", "not", "no", "but", "if",
    "then", "than", "so", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "all", "any", "some", "there", "here", "his", "her",
    "its", "their", "our", "your", "my", "me", "us", "them",
}


def _content_words_exist(con: sqlite3.Connection, fts_table: str, escaped_query: str) -> bool:
    """True if at least one non-stopword of the query exists in the FTS index.

    Guards the OR fallback: if only stopwords exist in the index ('quantum' OR
    'in' OR 'this'), every verse matches and the results are pure noise.
    """
    tokens = [t.strip('"') for t in escaped_query.split()]
    content = [t for t in tokens if t.lower() not in _EN_STOPWORDS]
    if not content:
        return False
    for token in content:
        try:
            hit = con.execute(
                f"SELECT rowid FROM {fts_table} WHERE {fts_table} MATCH ? LIMIT 1",
                (f'"{token}"',),
            ).fetchone()
        except sqlite3.OperationalError:
            continue
        if hit:
            return True
    return False
