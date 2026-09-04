from __future__ import annotations

import pytest

from ingestion.quran_ingest import (
    QURAN_EN_SOURCE_ID,
    QuranIngestor,
    QuranStore,
    TranslationIngestor,
)


@pytest.fixture(scope="module")
def translated_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("tdb") / "k.db"
    QuranIngestor(db_path=db).ingest()
    TranslationIngestor(db_path=db).ingest()
    return db


@pytest.fixture(scope="module")
def store(translated_db):
    return QuranStore(db_path=translated_db)


class TestTranslationIngestion:
    def test_all_ayahs_translated(self, store):
        assert store.translation_count("en") == 6236

    def test_alignment_with_arabic_mandatory(self, tmp_path):
        """A translation verse without an Arabic counterpart is a hard error.

        Uses an unregistered-content file whose hash therefore mismatches the
        registry too — either failure is the gate working; the verse-grid
        check is verified separately via the same code path in
        test_gate_blocks_unregistered_translation.
        """
        import json
        db = tmp_path / "k.db"
        QuranIngestor(db_path=db).ingest()
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"115": [{"chapter": 115, "verse": 1, "text": "x"}]}))
        with pytest.raises((ValueError, RuntimeError)):
            TranslationIngestor(db_path=db, raw_path=bad).ingest()
        # and the grid check specifically: same dataset but Arabic 114 removed
        con = __import__("sqlite3").connect(db)
        con.execute("DELETE FROM quran WHERE surah=114")
        con.commit()
        con.close()
        with pytest.raises(ValueError, match="no Arabic counterpart"):
            TranslationIngestor(db_path=db).ingest()

    def test_gate_blocks_unregistered_translation(self, tmp_path):
        import json
        db = tmp_path / "k.db"
        QuranIngestor(db_path=db).ingest()
        unregistered = tmp_path / "unregistered.json"
        unregistered.write_text(json.dumps({"1": [{"chapter": 1, "verse": 1, "text": "x"}]}))
        with pytest.raises(Exception):
            TranslationIngestor(
                db_path=db, raw_path=unregistered, source_id="random-translation"
            ).ingest()

    def test_deterministic_replay(self, translated_db):
        r = TranslationIngestor(db_path=translated_db).ingest()
        assert r.deterministic_replay


class TestTranslationRetrieval:
    def test_translation_fts_finds_fatihah(self, store):
        hits = store.search_translation_fts("entirely merciful especially merciful")
        ids = [h["citation_id"] for h in hits]
        assert "quran:1:1" in ids and "quran:1:3" in ids  # 1:1 and 1:3 both match

    def test_translation_joins_arabic(self, store):
        hits = store.search_translation_fts("He is Allah who is One")
        top = hits[0]
        assert top["citation_id"] == "quran:112:1"
        assert top["arabic"].startswith("قُل")  # Arabic row attached
        assert "translation_source_id" in top

    def test_get_ayah_with_lang(self, store):
        row = store.get_ayah(2, 255, lang="en")
        assert row["citation_id"] == "quran:2:255"
        assert "Allah" in row["translation"]
        assert row["translation_source_id"] == QURAN_EN_SOURCE_ID

    def test_get_ayah_without_lang_has_no_translation(self, store):
        row = store.get_ayah(2, 255)
        assert "translation" not in row
