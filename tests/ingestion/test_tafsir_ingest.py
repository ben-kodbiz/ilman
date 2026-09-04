from __future__ import annotations

import pytest

from agent.policy.source_policy import SourceRegistry
from ingestion.quran_ingest import QuranIngestor, QuranStore
from ingestion.tafsir_ingest import (
    TAFSIR_SOURCE_ID,
    TRANSLATION_SOURCE_ID,
    TafsirIngestor,
    TafsirStore,
    tafsir_citation_id,
)


@pytest.fixture(scope="module")
def tafsir_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("tdb") / "k.db"
    QuranIngestor(db_path=db).ingest()
    TafsirIngestor(db_path=db).ingest()
    return db


@pytest.fixture(scope="module")
def store(tafsir_db):
    return TafsirStore(db_path=tafsir_db)


@pytest.fixture(scope="module")
def quran_store(tafsir_db):
    return QuranStore(db_path=tafsir_db)


class TestIngestion:
    def test_full_tafsir_coverage(self, store):
        assert store.tafsir_count() == 6236

    def test_indonesian_translation_rides_quran_table(self, quran_store):
        assert quran_store.translation_count("id") == 6236
        row = quran_store.get_ayah(1, 1, lang="id")
        assert "Allah" in row["translation"]

    def test_citation_id_format(self):
        assert tafsir_citation_id(TAFSIR_SOURCE_ID, 2, 255) == "tafsir:tafsir-kemenag:2:255"

    def test_get_tafsir_ayat_kursi(self, store):
        row = store.get_tafsir(2, 255)
        assert row["citation_id"] == "tafsir:tafsir-kemenag:2:255"
        assert "Allah" in row["tafsir"]

    def test_gate_blocks_unregistered_tafsir(self, tmp_path):
        """Ingestion of an unregistered tafsir source is a hard error (§5.2)."""
        import json as _json
        import tarfile

        db = tmp_path / "k.db"
        QuranIngestor(db_path=db).ingest()
        # build a fake archive
        fake = tmp_path / "fake.tar.gz"
        with tarfile.open(fake, "w:gz") as tar:
            import io

            data = _json.dumps({
                "1": {"number_of_ayah": "7",
                      "tafsir": {"id": {"kemenag": {"text": {str(i): "x" for i in range(1, 8)}}}},
                      "translations": {"id": {"text": {str(i): "y" for i in range(1, 8)}}}}
            }).encode()
            info = tarfile.TarInfo("1.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        # raw archive for an id that isn't registered -> registry lookup fails
        with pytest.raises(Exception):
            ing = TafsirIngestor(db_path=db, raw_archive=fake)
            ing.TAFSIR_SOURCE_ID = "fake-tafsir"
            ing.TRANSLATION_SOURCE_ID = "fake-translation"
            ing.ingest()

    def test_tafsir_count_mismatch_is_hard_error(self, tmp_path):
        """A surah whose tafsir count disagrees with ayah count never ingests."""
        import io
        import json as _json
        import tarfile

        db = tmp_path / "k.db"
        QuranIngestor(db_path=db).ingest()
        fake = tmp_path / "bad.tar.gz"
        with tarfile.open(fake, "w:gz") as tar:
            data = _json.dumps({
                "1": {"number_of_ayah": "7",
                      "tafsir": {"id": {"kemenag": {"text": {str(i): "x" for i in range(1, 6)}}}},
                      "translations": {"id": {"text": {str(i): "y" for i in range(1, 8)}}}}
            }).encode()
            info = tarfile.TarInfo("1.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        with pytest.raises(ValueError, match="count mismatch"):
            ing = TafsirIngestor(db_path=db, raw_archive=fake)
            # bypass the hash pin (registry-pin test covers that layer) so the
            # count-mismatch validation is exercised directly
            import re as _re

            import ingestion.tafsir_ingest as ti

            record = SourceRegistry.load().get(TAFSIR_SOURCE_ID)
            m = _re.search(r"sha256=([0-9a-f]{64})", record.notes or "")
            ti.sha256_of = lambda path: m.group(1)
            try:
                ing.ingest()
            finally:
                ti.sha256_of = TafsirIngestor.__module__ and __import__(
                    "ingestion.quran_ingest", fromlist=["sha256_of"]
                ).sha256_of


class TestRetrieval:
    def test_tafsir_fts_indonesian(self, store):
        hits = store.search_fts("niat amal")
        assert hits and all(h["citation_id"].startswith("tafsir:") for h in hits)

    def test_tafsir_never_in_quran_table(self, store, quran_store):
        """Tafsir is interpretation; it must not leak into the Qur'an table."""
        import sqlite3

        con = sqlite3.connect(store.db_path)
        n = con.execute(
            "SELECT COUNT(*) FROM quran WHERE source_id IN (?,?)",
            (TAFSIR_SOURCE_ID, TRANSLATION_SOURCE_ID),
        ).fetchone()[0]
        con.close()
        assert n == 0

    def test_tier_is_2_in_hybrid(self, tafsir_db, quran_store):
        from retrieval.hybrid import RetrievalOrchestrator

        ro = RetrievalOrchestrator(quran_store, tafsir_store=TafsirStore(db_path=tafsir_db))
        hits = ro.search("What does 2:255 mean according to tafsir?")
        tafsir_hits = [h for h in hits if h.citation_id.startswith("tafsir:")]
        assert tafsir_hits and all(h.tier == 2 for h in tafsir_hits)
        # the ayah itself is TIER 0 and precedes its tafsir
        assert any(h.citation_id == "quran:2:255" and h.tier == 0 for h in hits)

    def test_reference_query_seeds_paired_tafsir(self, tafsir_db, quran_store):
        from retrieval.hybrid import RetrievalOrchestrator

        ro = RetrievalOrchestrator(quran_store, tafsir_store=TafsirStore(db_path=tafsir_db))
        hits = ro.search("What does 2:255 say?")
        ids = [h.citation_id for h in hits]
        assert "quran:2:255" in ids
        assert "tafsir:tafsir-kemenag:2:255" in ids  # paired explanation


class TestEvidencePack:
    def test_tafsir_prompt_block_renders_as_interpretation(self):
        from agent.validators.pipeline import EvidencePack
        from retrieval.hybrid import RetrievedPassage

        passage = RetrievedPassage(
            citation_id="tafsir:tafsir-kemenag:2:255", surah=2, ayah=255,
            arabic="", source_id=TAFSIR_SOURCE_ID, tier=2,
            leg="tafsir", score=-1.0, translation="Allah adalah Tuhan Yang Maha Esa...",
        )
        block = EvidencePack(query="q", passages=[passage]).to_prompt_block()
        assert "tafsir:tafsir-kemenag:2:255" in block
        assert "interpretation" in block  # model told it is TIER 2, not Qur'an text

    def test_tafsir_citation_validation(self):
        from agent.validators.pipeline import CitationValidator, EvidencePack
        from retrieval.hybrid import RetrievedPassage

        passage = RetrievedPassage(
            citation_id="tafsir:tafsir-kemenag:2:255", surah=2, ayah=255,
            arabic="", source_id=TAFSIR_SOURCE_ID, tier=2,
            leg="tafsir", score=-1.0, translation="...",
        )
        pack = EvidencePack(query="q", passages=[passage])
        v = CitationValidator().validate(
            "According to tafsir [tafsir:tafsir-kemenag:2:255], ...", pack
        )
        assert v.ok and v.verified_citations == ["tafsir:tafsir-kemenag:2:255"]
        # tafsir citation not in pack -> unsupported
        v2 = CitationValidator().validate(
            "According to [tafsir:tafsir-kemenag:9:9], ...", pack
        )
        assert not v2.ok
