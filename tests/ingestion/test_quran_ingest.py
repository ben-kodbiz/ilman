from __future__ import annotations

import pytest

from ingestion.quran_ingest import (
    DEFAULT_DB,
    QURAN_SOURCE_ID,
    QuranIngestor,
    QuranStore,
    citation_id,
)


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    """Ingest the real approved dataset into a temp DB once for this module."""
    db = tmp_path_factory.mktemp("kdb") / "knowledge.db"
    QuranIngestor(db_path=db).ingest()
    return QuranStore(db_path=db)


@pytest.fixture()
def store(ingested):
    return ingested


class TestIngestion:
    def test_full_quran_ingested(self, store):
        assert store.ayah_count_total() == 6236

    def test_deterministic_replay_flagged(self, tmp_path):
        db = tmp_path / "k.db"
        first = QuranIngestor(db_path=db).ingest()
        assert not first.deterministic_replay
        second = QuranIngestor(db_path=db).ingest()
        assert second.deterministic_replay
        assert second.ayah_count == 6236  # idempotent, no duplicates

    def test_gate_blocks_unknown_dataset(self, tmp_path):
        """Ingestion MUST refuse a dataset not on the approved registry."""
        bad = tmp_path / "fake.json"
        bad.write_text('{"1": [{"chapter": 1, "verse": 1, "text": "مصطنع"}]}')
        with pytest.raises(Exception):
            QuranIngestor(db_path=tmp_path / "k.db", raw_path=bad).ingest()

    def test_hash_mismatch_blocks_ingestion(self, tmp_path, monkeypatch):
        """Tampered file bytes must never reach the DB."""
        import ingestion.quran_ingest as qi

        def tampered(path):
            return "0" * 64

        monkeypatch.setattr(qi, "sha256_of", tampered)
        with pytest.raises(ValueError, match="hash mismatch"):
            QuranIngestor(db_path=tmp_path / "k.db").ingest()
        assert not (tmp_path / "k.db").exists()


class TestDeterministicRetrieval:
    def test_numeric_reference(self, store):
        row = store.get_by_reference("2:255")
        assert row["citation_id"] == "quran:2:255"
        # robust check via the same normalizer the index uses
        from ingestion.arabic_norm import search_form

        assert search_form(row["arabic"]).startswith("الله لا اله الا هو الحي القيوم")

    def test_named_reference(self, store):
        assert store.get_by_reference("Al-Baqarah 255")["ayah"] == 255

    def test_alias_reference(self, store):
        assert store.get_by_reference("Ayat al-Kursi")["surah"] == 2

    def test_invalid_reference_is_none_not_error(self, store):
        assert store.get_by_reference("nonsense") is None

    def test_out_of_range_is_none(self, store):
        assert store.get_ayah(200, 1) is None
        assert store.get_by_reference("1:999") is None

    def test_citation_id_format(self):
        assert citation_id(2, 255) == "quran:2:255"
        assert citation_id(114, 6) == "quran:114:6"


class TestFTS:
    def test_plain_arabic_query_finds_uthmani_text(self, store):
        hits = store.search_fts("قل هو الله أحد")
        assert hits and hits[0]["citation_id"] == "quran:112:1"

    def test_diacritic_insensitive(self, store):
        assert store.search_fts("الرحمن")[0]["surah"] == 55

    def test_multiword_phrase(self, store):
        hits = store.search_fts("الله الصمد")
        assert any(h["citation_id"] == "quran:112:2" for h in hits)

    def test_no_fts_injection_via_quotes(self, store):
        # user input must not be able to break/mutate the MATCH query
        hits = store.search_fts('"قل" OR 1=1')
        assert isinstance(hits, list)  # no error raised

    def test_empty_query_safe(self, store):
        assert store.search_fts("") == []

    def test_every_hit_has_provenance(self, store):
        for h in store.search_fts("الله")[:10]:
            assert h["source_id"] == QURAN_SOURCE_ID
            assert h["citation_id"].startswith("quran:")


class TestSchemaIntegrity:
    def test_arabic_column_untouched_by_normalization(self, store):
        row = store.get_ayah(112, 1)
        # original Uthmani marks preserved exactly as ingested (§7)
        assert "ٱ" in row["arabic"]
        assert row["arabic"].startswith("قُلۡ")

    def test_sources_row_registered(self, store):
        with store._connect() as con:
            row = con.execute(
                "SELECT source_id, tradition, verification_status FROM sources WHERE source_id=?",
                (QURAN_SOURCE_ID,),
            ).fetchone()
        assert row["tradition"] == "SUNNI"
        assert row["verification_status"] == "verified"

    def test_production_db_exists_and_is_complete(self):
        if not DEFAULT_DB.exists():
            pytest.skip("production DB not ingested yet")
        store = QuranStore()
        assert store.ayah_count_total() == 6236
