from __future__ import annotations

import json

import pytest

from ingestion.web_fatwa_ingest import (
    SOURCE_ID,
    WebFatwaIngestor,
    WebFatwaStore,
    html_to_text,
)

RAW = {
    "1": {
        "id": 1,
        "url": "https://islamqa.info/en/answers/1",
        "title": "Interruption of Wudu",
        "summary": "Continuity would not be disrupted.",
        "body_html": "<p>Continuity would not be disrupted by such an action "
        "(according to the most viable opinion) even if his body had dried, "
        "because he was delayed due to an action required for his purity.</p>",
        "scholar": "Muhammad Saalih al-Munajjid",
        "harvested_at": "2026-09-06T00:00:00+00:00",
    },
    "300": {
        "id": 300,
        "url": "https://islamqa.info/en/answers/300",
        "title": "Who Are Ahlul Kitab (People of the Book)?",
        "summary": "1- Ahlul Kitab consist of both believers and disbelievers.",
        "body_html": "<p>Ahlul Kitab (People of the Book) consist of both "
        "believers and disbelievers as indicated in the Quran. The disbelief "
        "of the disbelieving People of the Scripture does not expel them "
        "from being People of the Scripture.</p>",
        "scholar": "Muhammad Saalih al-Munajjid",
        "harvested_at": "2026-09-06T00:00:00+00:00",
    },
    # junk records that must be skipped, never padded
    "9": {"id": 9, "title": "stub", "body_html": "<p>too short</p>"},
    "bad": {"id": 7, "title": "unparseable"},
}


@pytest.fixture(scope="module")
def fatwa_db(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    for name, rec in RAW.items():
        (raw / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")
    db = tmp_path_factory.mktemp("wfdb") / "k.db"
    WebFatwaIngestor(db_path=db, raw_dir=raw).ingest()
    return db


@pytest.fixture(scope="module")
def store(fatwa_db):
    return WebFatwaStore(db_path=fatwa_db)


class TestHtmlToText:
    def test_strips_tags_and_entities(self):
        assert html_to_text("<p>A &amp; B &nbsp; <b>C</b></p>") == "A & B C"

    def test_collapses_whitespace(self):
        assert html_to_text("<p>one\ntwo   three</p>") == "one two three"


class TestIngestion:
    def test_ingests_valid_skips_junk(self, store):
        # 2 valid fatwas; stub (short body) + unparseable JSON name skipped
        assert store.fatwa_count() == 2

    def test_citation_format_and_provenance(self, store):
        rec = store.get_fatwa("1")
        assert rec["citation_id"] == "webfatwa:islamqa-info-en:1"
        assert rec["source_id"] == SOURCE_ID
        assert rec["url"] == "https://islamqa.info/en/answers/1"
        assert rec["scholar"] == "Muhammad Saalih al-Munajjid"
        assert "<p>" not in rec["body"]  # HTML stripped

    def test_body_search_includes_title(self, store):
        # question phrasing must match, not just the answer body
        hits = store.search_fts("Interruption of Wudu")
        assert any(h["answer_id"] == "1" for h in hits)

    def test_fts_matches_answer_content(self, store):
        hits = store.search_fts("breastfeeding")
        assert not any(h["answer_id"] == "1" for h in hits)
        hits2 = store.search_fts("People of the Book believers disbelievers")
        assert any(h["answer_id"] == "300" for h in hits2)

    def test_gate_blocks_unapproved_source(self, tmp_path, monkeypatch):
        """A source failing the §5.2 gate must never write."""
        import ingestion.web_fatwa_ingest as wf

        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "1.json").write_text(json.dumps(RAW["1"]), encoding="utf-8")

        class BlockedPolicy(wf.SourcePolicy):
            def assert_ingestible(self, record):
                raise RuntimeError("gate blocked")

        monkeypatch.setattr(
            wf, "SourcePolicy", lambda reg: BlockedPolicy(reg)
        )
        with pytest.raises(RuntimeError, match="gate blocked"):
            WebFatwaIngestor(db_path=tmp_path / "k.db", raw_dir=raw).ingest()

    def test_empty_raw_dir_is_error(self, tmp_path):
        with pytest.raises(ValueError, match="no harvested"):
            WebFatwaIngestor(db_path=tmp_path / "k.db", raw_dir=tmp_path).ingest()

    def test_idempotent_replay(self, fatwa_db, tmp_path_factory):
        raw = tmp_path_factory.mktemp("raw2")
        for name, rec in RAW.items():
            if name in ("9", "bad"):
                continue
            (raw / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")
        # same content, fresh dir: first run seeds, second replays deterministically
        WebFatwaIngestor(db_path=fatwa_db, raw_dir=raw).ingest()
        res = WebFatwaIngestor(db_path=fatwa_db, raw_dir=raw).ingest()
        assert res.deterministic_replay is True
        assert WebFatwaStore(db_path=fatwa_db).fatwa_count() == 2


class TestRetrievalLeg:
    def test_web_fatwa_leg_fuses_with_quran(self, fatwa_db):
        from ingestion.quran_ingest import QuranStore
        from retrieval.hybrid import RetrievalOrchestrator

        orch = RetrievalOrchestrator(
            QuranStore(db_path=fatwa_db), web_fatwa_store=WebFatwaStore(db_path=fatwa_db)
        )
        hits = orch.search("interruption of wudu purity", limit=6)
        legs = {h.leg for h in hits}
        assert "web_fatwa" in legs
        wf = [h for h in hits if h.leg == "web_fatwa"][0]
        assert wf.tier == 4
        assert wf.source_id == SOURCE_ID

    def test_web_fatwa_excluded_for_semantic_only(self, fatwa_db):
        """Emotional statements must not lexically match fatwas (noise)."""
        from ingestion.quran_ingest import QuranStore
        from retrieval.hybrid import RetrievalOrchestrator

        orch = RetrievalOrchestrator(
            QuranStore(db_path=fatwa_db), web_fatwa_store=WebFatwaStore(db_path=fatwa_db)
        )
        hits = orch.search("wudu", limit=6, semantic_only=True)
        assert all(h.leg != "web_fatwa" for h in hits)
