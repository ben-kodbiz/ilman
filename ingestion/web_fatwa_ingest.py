"""Web fatwa ingestion from harvested public fatwa JSON files.

TIER 4 (contemporary scholarship, agentodo §6). Source:
knowledge/web/raw/islamqa_en/<id>.json — one file per harvested answer
from islamqa.info (Sheikh Muhammad Saalih al-Munajjid, Zad Foundation),
harvested by scripts/harvest_islamqa.py (polite crawl of the public site).

A fatwa is contemporary scholarly opinion subordinate to TIER 0-2 sources;
it never overrides Qur'an/hadith/tafsir. Ingested rows keep full web
provenance (answer id, url, title, scholar) so citations can point back to
the original page. Question text stays with the answer (the fatwa is the
Q&A pair, not a free-floating text).
"""

from __future__ import annotations

import html as html_mod
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from agent.policy.source_policy import SourcePolicy, SourceRegistry
from ingestion.quran_ingest import SCHEMA as QURAN_SCHEMA  # ingestion_log lives here

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "knowledge" / "web" / "raw" / "islamqa_en"
DEFAULT_DB = REPO_ROOT / "knowledge" / "processed" / "knowledge.db"

SOURCE_ID = "islamqa-info-en"
MIN_BODY_CHARS = 80  # a real fatwa answer is at least a paragraph
MAX_ROW_CHARS = 12000  # cap absurdly long answers (site junk), keep provenance

WEB_FATWA_SCHEMA = """
CREATE TABLE IF NOT EXISTS web_fatwas (
    citation_id TEXT PRIMARY KEY,          -- "webfatwa:islamqa-info-en:<answer_id>"
    source_id TEXT NOT NULL REFERENCES sources(source_id),
    answer_id TEXT NOT NULL,              -- site answer id (stable)
    url TEXT NOT NULL,
    title TEXT NOT NULL,                  -- the question asked
    summary TEXT NOT NULL DEFAULT '',     -- site-provided answer summary
    body TEXT NOT NULL,                   -- answer text (HTML stripped)
    body_search TEXT NOT NULL,            -- FTS-normalized
    scholar TEXT NOT NULL,
    harvested_at TEXT NOT NULL,
    UNIQUE (source_id, answer_id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS web_fatwas_fts USING fts5(
    body_search,
    content='web_fatwas', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS wf_ai AFTER INSERT ON web_fatwas BEGIN
    INSERT INTO web_fatwas_fts(rowid, body_search) VALUES (new.rowid, new.body_search);
END;
CREATE TRIGGER IF NOT EXISTS wf_ad AFTER DELETE ON web_fatwas BEGIN
    INSERT INTO web_fatwas_fts(web_fatwas_fts, rowid, body_search)
    VALUES ('delete', old.rowid, old.body_search);
END;
"""

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def html_to_text(raw: str) -> str:
    """Strip tags, unescape entities, collapse whitespace."""
    text = _TAG_RE.sub(" ", raw)
    text = html_mod.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _digest_dir(raw_dir: Path) -> str:
    """Deterministic content hash over the raw harvest (sorted ids)."""
    h = sha256()
    for f in sorted(raw_dir.glob("*.json"), key=lambda p: p.name):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@dataclass
class WebFatwaIngestResult:
    source_id: str
    fatwa_count: int
    skipped: int
    run_id: str
    deterministic_replay: bool


class WebFatwaIngestor:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        raw_dir: Path = DEFAULT_RAW_DIR,
        policy: SourcePolicy | None = None,
    ):
        self.db_path = db_path
        self.raw_dir = raw_dir
        self.policy = policy or SourcePolicy(SourceRegistry.load())

    def ingest(self) -> WebFatwaIngestResult:
        record = self.policy.registry.get(SOURCE_ID)
        self.policy.assert_ingestible(record)  # §5.2 gate before any write
        files = sorted(self.raw_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        # index.json from the crawler is metadata, not an answer
        files = [f for f in files if f.name != "index.json"]
        if not files:
            raise ValueError(f"no harvested fatwa files in {self.raw_dir}")
        digest = _digest_dir(self.raw_dir)

        rows: list[tuple] = []
        skipped = 0
        for f in files:
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                skipped += 1
                continue
            body = html_to_text(rec.get("body_html", ""))
            if len(body) < MIN_BODY_CHARS:
                skipped += 1
                continue  # too short to be a real answer; never pad
            title = (rec.get("title") or "").strip() or f"Answer {rec['id']}"
            title = (rec.get("title") or "").strip() or f"Answer {rec['id']}"
            rows.append((
                f"webfatwa:{SOURCE_ID}:{rec['id']}",
                SOURCE_ID,
                str(rec["id"]),
                rec.get("url", f"https://islamqa.info/en/answers/{rec['id']}"),
                title,
                (rec.get("summary") or "").strip(),
                body[:MAX_ROW_CHARS],
                f"{title} {html_to_text(rec.get('summary') or '')} {body}"[:MAX_ROW_CHARS * 2],
                rec.get("scholar") or "Muhammad Saalih al-Munajjid",
                rec.get("harvested_at") or datetime.now(UTC).isoformat(),
            ))

        run_id = f"ingest:{SOURCE_ID}:{digest[:12]}"
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(QURAN_SCHEMA)  # ensures sources + ingestion_log exist
            con.executescript(WEB_FATWA_SCHEMA)
            replay_row = con.execute(
                # replay = this exact content was ingested before (any prior
                # run, not just the lexicographically-latest run_id)
                "SELECT 1 FROM ingestion_log WHERE source_id=? AND sha256=? "
                "AND ayah_count=? LIMIT 1",
                (SOURCE_ID, digest, len(rows)),
            ).fetchone()
            replay = replay_row is not None
            con.execute("DELETE FROM web_fatwas WHERE source_id=?", (SOURCE_ID,))
            con.execute(
                "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?)",
                (SOURCE_ID, record.title, record.author, record.type, record.language,
                 record.tradition, record.license, record.verification_status, digest),
            )
            con.executemany(
                "INSERT OR REPLACE INTO web_fatwas (citation_id, source_id, answer_id, url, "
                "title, summary, body, body_search, scholar, harvested_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            con.execute(
                "INSERT OR REPLACE INTO ingestion_log VALUES (?,?,?,?,?)",
                (run_id, SOURCE_ID, digest, len(rows), replay),
            )
            con.commit()
        finally:
            con.close()
        return WebFatwaIngestResult(SOURCE_ID, len(rows), skipped, run_id, replay)


class WebFatwaStore:
    """Read-side access to web fatwas (TIER 4)."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def fatwa_count(self) -> int:
        try:
            with self._connect() as con:
                return con.execute("SELECT COUNT(*) FROM web_fatwas").fetchone()[0]
        except sqlite3.OperationalError:
            return 0

    def get_fatwa(self, answer_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT citation_id, source_id, answer_id, url, title, summary, body, "
                "scholar FROM web_fatwas WHERE answer_id=?",
                (str(answer_id),),
            ).fetchone()
        return dict(row) if row else None

    def search_fts(self, query: str, limit: int = 10) -> list[dict]:
        """BM25 search over fatwa bodies + titles (title prepended to the
        body_search index text so question-phrasing matches too)."""
        from ingestion.quran_ingest import _content_words_exist, _fts_escape

        sql = (
            "SELECT c.citation_id, c.source_id, c.answer_id, c.url, c.title, c.summary, "
            "c.body, c.scholar, bm25(web_fatwas_fts) AS rank "
            "FROM web_fatwas_fts f JOIN web_fatwas c ON c.rowid = f.rowid "
            "WHERE web_fatwas_fts MATCH ? ORDER BY rank LIMIT ?"
        )
        with self._connect() as con:
            escaped = _fts_escape(query)
            rows = con.execute(sql, (escaped, limit)).fetchall()
            if not rows and " " in escaped.strip() and _content_words_exist(
                con, "web_fatwas_fts", escaped
            ):
                or_query = escaped.replace('" ', '" OR ')
                rows = con.execute(sql, (or_query, limit)).fetchall()
        return [dict(r) for r in rows]
