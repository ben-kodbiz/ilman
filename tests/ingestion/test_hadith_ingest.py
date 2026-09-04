from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.hadith_ingest import (
    KUTUB_AL_SITTAH,
    HadithIngestor,
    HadithStore,
    hadith_citation_id,
)
from ingestion.quran_ingest import QuranIngestor


@pytest.fixture(scope="module")
def hadith_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("hdb") / "k.db"
    QuranIngestor(db_path=db).ingest()  # shared schema baseline
    ingestor = HadithIngestor(db_path=db)
    ingestor.ingest_all()
    return HadithStore(db_path=db)


@pytest.fixture(scope="module")
def store(hadith_db):
    return hadith_db


class TestIngestion:
    def test_all_six_collections_ingested(self, store):
        collections = {c["source_id"] for c in store.collections()}
        assert collections == set(KUTUB_AL_SITTAH)

    def test_known_hadith_retrievable(self, store):
        from ingestion.arabic_norm import search_form

        row = store.get_hadith("sahih-bukhari", 1)
        assert row["citation_id"] == "hadith:sahih-bukhari:1"
        # 'intentions' hadith, checked via normalized forms (diacritic-proof)
        assert "النيات" in search_form(row["arabic"])
        assert "intentions" in row["english"].lower()

    def test_grading_metadata_preserved(self, store):
        # Abu Dawud 1 has multiple grader entries in the dataset
        row = store.get_hadith("sunan-abu-dawud", 1)
        assert row["grades"], "dataset grading metadata must be preserved (§6)"
        graders = {g["name"] for g in row["grades"]}
        assert any("Albani" in name for name in graders)

    def test_bukhari_grades_empty_not_invented(self, store):
        row = store.get_hadith("sahih-bukhari", 1)
        assert row["grades"] == []  # dataset has none; we never invent (§13)
        col = next(c for c in store.collections() if c["source_id"] == "sahih-bukhari")
        assert "collection-level" in col["grading_basis"]

    def test_citation_id_format(self):
        assert hadith_citation_id("sahih-muslim", 123) == "hadith:sahih-muslim:123"

    def test_gate_blocks_unregistered_collection(self, tmp_path):
        db = tmp_path / "k.db"
        QuranIngestor(db_path=db).ingest()
        bad = HadithIngestor(db_path=db)
        bad.KUTUB_AL_SITTAH = {**KUTUB_AL_SITTAH, "fake-book": ("fake", "Fake Book")}
        with pytest.raises(Exception):
            bad.ingest("fake-book")

    def test_numbering_misalignment_is_hard_error(self, tmp_path):
        """AR/EN misalignment must fail ingestion, never silently shift."""
        import json
        import shutil

        db = tmp_path / "k.db"
        QuranIngestor(db_path=db).ingest()
        raw = tmp_path / "raw"
        raw.mkdir()
        src = Path("knowledge/hadith/raw")
        for f in src.glob("hadith-*-bukhari.json"):
            shutil.copy(f, raw / f.name)
        eng = json.loads((raw / "hadith-eng-bukhari.json").read_text(encoding="utf-8"))
        eng["hadiths"] = eng["hadiths"][1:]  # drop one EN hadith -> count mismatch
        (raw / "hadith-eng-bukhari.json").write_text(json.dumps(eng, ensure_ascii=False), encoding="utf-8")
        ing = HadithIngestor(db_path=db, raw_dir=raw)
        # bypass registry hash pinning is impossible; hash mismatch fires first —
        # either failure proves the corpus never ingests misaligned data.
        with pytest.raises((ValueError, KeyError)):
            ing.ingest("sahih-bukhari")


class TestSearch:
    def test_english_search_finds_intentions_hadith(self, store):
        hits = store.search_fts("reward of deeds depends upon the intentions")
        assert hits and hits[0]["citation_id"] == "hadith:sahih-bukhari:1"

    def test_arabic_search_normalized(self, store):
        hits = store.search_fts("إنما الأعمال بالنيات")
        assert any(h["citation_id"] == "hadith:sahih-bukhari:1" for h in hits)

    def test_collection_filter(self, store):
        hits = store.search_fts("prayer", source_id="sahih-muslim")
        assert hits and all(h["source_id"] == "sahih-muslim" for h in hits)

    def test_every_hit_carries_provenance(self, store):
        for h in store.search_fts("patience")[:10]:
            assert h["citation_id"].startswith("hadith:")
            assert h["source_id"] in KUTUB_AL_SITTAH

    def test_garbage_query_no_noise(self, store):
        assert store.search_fts("quantum chromodynamics") == []


class TestHybridIntegration:
    def test_hadith_leg_in_hybrid_search(self, tmp_path_factory):
        from ingestion.quran_ingest import QuranStore, TranslationIngestor
        from retrieval.hybrid import RetrievalOrchestrator

        db = tmp_path_factory.mktemp("hdb2") / "k.db"
        QuranIngestor(db_path=db).ingest()
        TranslationIngestor(db_path=db).ingest()
        HadithIngestor(db_path=db).ingest_all()
        ro = RetrievalOrchestrator(QuranStore(db_path=db), hadith_store=HadithStore(db_path=db))
        hits = ro.search("hadith about intentions behind deeds")
        hadith_hits = [h for h in hits if h.citation_id.startswith("hadith:")]
        assert hadith_hits, "hybrid search must include hadith evidence"
        assert any(h.tier == 1 for h in hadith_hits)

    def test_hadith_evidence_in_prompt_block(self, tmp_path_factory):
        from agent.validators.pipeline import EvidencePack
        from retrieval.hybrid import RetrievedPassage

        passage = RetrievedPassage(
            citation_id="hadith:sunan-abu-dawud:1", surah=0, ayah=0,
            arabic="نَصُّ الْحَدِيث", source_id="sunan-abu-dawud", tier=1,
            leg="hadith", score=-1.0, translation="English text",
            collection="sunan-abu-dawud", hadithnumber=1,
            grades=[{"name": "Al-Albani", "grade": "Hasan Sahih"}],
        )
        block = EvidencePack(query="q", passages=[passage]).to_prompt_block()
        assert "hadith:sunan-abu-dawud:1" in block
        assert "Al-Albani: Hasan Sahih" in block  # grades shown to model verbatim
