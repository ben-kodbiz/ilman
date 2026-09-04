from __future__ import annotations

import pytest

from agent.policy.source_policy import SourcePolicy, SourceRegistry
from ingestion.quran_ingest import QuranStore
from retrieval.hybrid import RetrievalOrchestrator, RetrievedPassage


@pytest.fixture(scope="module")
def orchestrator(tmp_path_factory):
    db = tmp_path_factory.mktemp("rdb") / "k.db"
    from ingestion.quran_ingest import QuranIngestor, TranslationIngestor
    QuranIngestor(db_path=db).ingest()
    TranslationIngestor(db_path=db).ingest()
    return RetrievalOrchestrator(QuranStore(db_path=db))


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    db = tmp_path_factory.mktemp("sdb") / "k.db"
    from ingestion.quran_ingest import QuranIngestor, TranslationIngestor
    QuranIngestor(db_path=db).ingest()
    TranslationIngestor(db_path=db).ingest()
    return QuranStore(db_path=db)


class TestHybridSearch:
    def test_reference_leg_is_deterministic(self, orchestrator):
        hits = orchestrator.search("What does 2:255 say?")
        assert hits and hits[0].citation_id == "quran:2:255"
        assert hits[0].leg == "reference"

    def test_fts_leg_ranks_relevant_surah_first(self, orchestrator):
        hits = orchestrator.search("قل هو الله أحد")
        assert hits and hits[0].citation_id == "quran:112:1"

    def test_english_query_uses_translation_leg(self, store):
        """English queries must retrieve via the translation corpus."""
        from retrieval.hybrid import RetrievalOrchestrator
        ro = RetrievalOrchestrator(store)
        hits = ro.search("who is Allah")
        assert hits and hits[0].translation
        assert hits[0].leg in ("translation", "reference")

    def test_fusion_merges_legs(self, orchestrator):
        hits = orchestrator.search("2:255 الله الحي القيوم")
        ids = [h.citation_id for h in hits]
        assert "quran:2:255" in ids
        top = next(h for h in hits if h.citation_id == "quran:2:255")
        assert top.leg == "reference"  # exact reference always survives on top

    def test_every_passage_carries_provenance(self, orchestrator):
        for h in orchestrator.search("الرحمن")[:10]:
            assert h.citation_id.startswith("quran:")
            assert h.source_id == "quran-uthmani-json"
            assert h.tier == 0

    def test_no_evidence_for_garbage(self, orchestrator):
        assert orchestrator.search("xyzzy frobnicate") == []

    def test_limit_respected(self, orchestrator):
        assert len(orchestrator.search("الله", limit=3)) <= 3


class TestSourceFilterIntegration:
    def test_filter_blocks_excluded_source(self, store=None):
        """A passage from an excluded source must never survive search."""
        ro = RetrievalOrchestrator.__new__(RetrievalOrchestrator)
        ro.store = None
        ro.policy = SourcePolicy(SourceRegistry.load())
        passage = RetrievedPassage(
            citation_id="quran:1:1", surah=1, ayah=1, arabic="x",
            source_id="web-uncategorized-fatwa-compilation", tier=5, leg="fts", score=-1,
        )
        assert not ro._filter_passes(passage)

    def test_filter_blocks_unregistered_source(self):
        ro = RetrievalOrchestrator.__new__(RetrievalOrchestrator)
        ro.store = None
        ro.policy = SourcePolicy(SourceRegistry.load())
        passage = RetrievedPassage(
            citation_id="quran:1:1", surah=1, ayah=1, arabic="x",
            source_id="not-in-registry", tier=0, leg="fts", score=-1,
        )
        assert not ro._filter_passes(passage)

    def test_filter_passes_approved_source(self):
        ro = RetrievalOrchestrator.__new__(RetrievalOrchestrator)
        ro.store = None
        ro.policy = SourcePolicy(SourceRegistry.load())
        passage = RetrievedPassage(
            citation_id="quran:1:1", surah=1, ayah=1, arabic="x",
            source_id="quran-uthmani-json", tier=0, leg="fts", score=-1,
        )
        assert ro._filter_passes(passage)


class TestRRF:
    def test_rrf_orders_by_fused_score(self):
        legs = [
            [RetrievedPassage("quran:1:1", 1, 1, "a", "quran-uthmani-json", 0, "fts", -1),
             RetrievedPassage("quran:2:255", 2, 255, "b", "quran-uthmani-json", 0, "fts", -1)],
            [RetrievedPassage("quran:2:255", 2, 255, "b", "quran-uthmani-json", 0, "fts", -1)],
        ]
        fused = RetrievalOrchestrator._rrf(legs)
        assert fused[0].citation_id == "quran:2:255"  # found in both legs
        assert all(p.score > 0 for p in fused)

    def test_rrf_prefers_reference_passage(self):
        legs = [
            [RetrievedPassage("quran:112:1", 112, 1, "a", "quran-uthmani-json", 0, "fts", -1, "")],
            [RetrievedPassage("quran:112:1", 112, 1, "a", "quran-uthmani-json", 0, "reference", 1.0, "")],
        ]
        fused = RetrievalOrchestrator._rrf(legs)
        assert fused[0].leg == "reference"

    def test_rrf_keeps_translation_text(self):
        legs = [
            [RetrievedPassage("quran:112:1", 112, 1, "a", "quran-uthmani-json", 0,
                              "translation", -1, "Say, He is One")],
            [RetrievedPassage("quran:112:1", 112, 1, "a", "quran-uthmani-json", 0,
                              "fts", -2, "")],
        ]
        fused = RetrievalOrchestrator._rrf(legs)
        assert fused[0].translation == "Say, He is One"
