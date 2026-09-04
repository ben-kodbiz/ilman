from __future__ import annotations

import pytest

from ingestion.quran_ingest import QuranIngestor, QuranStore
from ingestion.tafsir_en_ingest import TafsirEnIngestor, TafsirEnStore


@pytest.fixture(scope="module")
def en_tafsir_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("tedb") / "k.db"
    QuranIngestor(db_path=db).ingest()
    TafsirEnIngestor(db_path=db).ingest_all()
    return db


@pytest.fixture(scope="module")
def store(en_tafsir_db):
    return TafsirEnStore(db_path=en_tafsir_db)


class TestIngestion:
    def test_all_three_scholars_ingested(self, store):
        assert store.chunk_count() == 18940

    def test_quarantine_preserved_not_dropped(self, en_tafsir_db):
        """Bad-tag chunks must be quarantined with reasons, never deleted."""
        import sqlite3

        con = sqlite3.connect(en_tafsir_db)
        n = con.execute("SELECT COUNT(*) FROM tafsir_en_quarantine").fetchone()[0]
        reasons = {r[0] for r in con.execute("SELECT reason FROM tafsir_en_quarantine")}
        con.close()
        assert n == 164
        assert any("not in Uthmani grid" in r for r in reasons)
        assert any("null ayah tag" in r for r in reasons)

    def test_ayat_kursi_commentary_from_ibn_kathir(self, store):
        chunks = store.get_for_ayah(2, 255)
        scholars = {c["scholar"] for c in chunks}
        assert "Ibn Kathir" in scholars
        # chunks covering 2:255 must reference the ayah inline
        assert any("2:255" in c["text"] for c in chunks)

    def test_qurtubi_missing_surahs_documented(self, store):
        """Qurtubi covers surahs 1-94 only (source PDFs lack 95-114)."""
        chunks = store.get_for_ayah(112, 1)
        assert all(c["source_id"] != "tafsir-qurtubi-en" for c in chunks)
        assert any(c["source_id"] == "tafsir-sadi-en" for c in chunks)  # 112 covered by sadi

    def test_source_hash_mismatch_blocks(self, tmp_path, monkeypatch):
        """Tampered source DB must never ingest (hash pinning)."""
        import ingestion.tafsir_en_ingest as te

        db = tmp_path / "k.db"
        QuranIngestor(db_path=db).ingest()
        monkeypatch.setattr(te, "sha256_of", lambda p: "0" * 64)
        with pytest.raises(ValueError, match="hash mismatch"):
            TafsirEnIngestor(db_path=db).ingest("tafsir-sadi-en")


class TestRetrieval:
    def test_fts_finds_ibn_kathir_on_intentions(self, store):
        hits = store.search_fts("deeds depend on the intentions")
        assert hits and hits[0]["scholar"] in ("Ibn Kathir", "Abd al-Rahman al-Sa'di")

    def test_fts_filter_by_source(self, store):
        hits = store.search_fts("mercy", source_id="tafsir-qurtubi-en")
        assert hits and all(h["source_id"] == "tafsir-qurtubi-en" for h in hits)

    def test_hybrid_includes_classic_tafsir_for_reference_query(self, en_tafsir_db):
        """2:255 query must surface multi-scholar tafsir evidence (§6)."""
        from retrieval.hybrid import RetrievalOrchestrator

        ro = RetrievalOrchestrator(
            QuranStore(db_path=en_tafsir_db), tafsir_en_store=TafsirEnStore(db_path=en_tafsir_db)
        )
        hits = ro.search("What does 2:255 say?", limit=8)
        en_tafsir = [h for h in hits if h.citation_id.startswith("tafsir-en:")]
        assert en_tafsir, "classic tafsir must appear in reference-query evidence"
        assert en_tafsir[0].tier == 2
        assert en_tafsir[0].scholar  # scholar attribution preserved

    def test_hybrid_tier2_source_diversity(self, en_tafsir_db):
        """TIER 2 quota prefers distinct tafsir works (scholarly perspectives)."""
        from retrieval.hybrid import RetrievalOrchestrator

        ro = RetrievalOrchestrator(
            QuranStore(db_path=en_tafsir_db), tafsir_en_store=TafsirEnStore(db_path=en_tafsir_db)
        )
        hits = ro.search("What does 2:255 say?", limit=8)
        t2_sources = {h.source_id for h in hits if h.tier == 2}
        assert len(t2_sources) >= 2  # kemenag + at least one classic scholar

    def test_tafsir_en_citation_validation(self):
        """tafsir-en:<chunk_id> citations validate against the evidence pack."""
        from agent.validators.pipeline import CitationValidator, EvidencePack
        from retrieval.hybrid import RetrievedPassage

        passage = RetrievedPassage(
            citation_id="tafsir-en:tafsir_ibn_kathir_en_001_d98dc204_00511",
            surah=2, ayah=255, arabic="", source_id="tafsir-ibn-kathir-en",
            tier=2, leg="tafsir", score=-1.0,
            translation="Ibn Kathir's commentary...", scholar="Ibn Kathir",
        )
        pack = EvidencePack(query="q", passages=[passage])
        v = CitationValidator().validate(
            "As Ibn Kathir explains [tafsir-en:tafsir_ibn_kathir_en_001_d98dc204_00511]...",
            pack,
        )
        assert v.ok
        assert v.verified_citations == ["tafsir-en:tafsir_ibn_kathir_en_001_d98dc204_00511"]
        # chunk not in pack -> unsupported
        v2 = CitationValidator().validate("[tafsir-en:tafsir_ibn_kathir_en_001_d98dc204_99999]", pack)
        assert not v2.ok

    def test_evidence_pack_names_scholar(self):
        from agent.validators.pipeline import EvidencePack
        from retrieval.hybrid import RetrievedPassage

        passage = RetrievedPassage(
            citation_id="tafsir-en:tafsir_ibn_kathir_en_001_d98dc204_00511",
            surah=2, ayah=255, arabic="", source_id="tafsir-ibn-kathir-en",
            tier=2, leg="tafsir", score=-1.0,
            translation="commentary text", scholar="Ibn Kathir",
        )
        block = EvidencePack(query="q", passages=[passage]).to_prompt_block()
        assert "Ibn Kathir" in block
        assert "interpretation" in block  # never presented as Qur'an text
