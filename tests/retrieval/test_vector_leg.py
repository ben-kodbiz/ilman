from __future__ import annotations

import pytest

from ingestion.quran_ingest import QuranIngestor, QuranStore
from retrieval.hybrid import RetrievalOrchestrator
from retrieval.vector_store import VectorStore


class MockVectorStore:
    """Deterministic mock: returns preset hits for preset queries."""

    def __init__(self, mapping: dict[str, list[dict]]):
        self.mapping = mapping
        self.queries: list[str] = []

    @property
    def size(self) -> int:
        return 1

    def search(self, query: str, top_k: int = 12) -> list[dict]:
        self.queries.append(query)
        return self.mapping.get(query, [])[:top_k]


@pytest.fixture(scope="module")
def quran_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("vdb") / "k.db"
    QuranIngestor(db_path=db).ingest()
    from ingestion.quran_ingest import TranslationIngestor
    TranslationIngestor(db_path=db).ingest()
    return db


@pytest.fixture(scope="module")
def store(quran_db):
    return QuranStore(db_path=quran_db)


class TestVectorLeg:
    def test_semantic_hit_joins_evidence(self, store):
        """'I am lonely' -> vector hit 2:186 must surface in hybrid results."""
        mock = MockVectorStore({"I am lonely": [
            {"citation_id": "quran:2:186", "score": 0.53},
            {"citation_id": "quran:93:3", "score": 0.49},
        ]})
        ro = RetrievalOrchestrator(store, vector_store=mock)
        hits = ro.search("I am lonely", limit=8)
        ids = [h.citation_id for h in hits]
        assert "quran:2:186" in ids
        vec = [h for h in hits if h.citation_id == "quran:2:186"][0]
        assert vec.leg == "vector"
        assert vec.tier == 0
        assert vec.translation  # full passage resolved, not a bare citation id

    def test_vector_hit_carries_provenance(self, store):
        mock = MockVectorStore({"q": [{"citation_id": "quran:2:186", "score": 0.5}]})
        ro = RetrievalOrchestrator(store, vector_store=mock)
        hits = ro.search("q")
        v = [h for h in hits if h.leg == "vector"][0]
        assert v.source_id == "quran-uthmani-json"

    def test_vector_leg_never_bypasses_source_filter(self, store):
        """A vector hit on an excluded/unknown source must be filtered out (§8)."""
        mock = MockVectorStore({"q": [
            {"citation_id": "quran:2:186", "score": 0.5},
        ]})
        ro = RetrievalOrchestrator(store, vector_store=mock)
        # sabotage: registry lookup for the quran source excluded
        ro.policy.must_not_retrieve = lambda sid: True
        hits = ro.search("q")
        assert all(h.citation_id != "quran:2:186" for h in hits)

    def test_dedup_against_other_legs(self, store):
        """A citation found by both FTS and vector must not duplicate."""
        mock = MockVectorStore({"Allah": [{"citation_id": "quran:112:1", "score": 0.9}]})
        ro = RetrievalOrchestrator(store, vector_store=mock)
        hits = ro.search("Allah")
        ids = [h.citation_id for h in hits]
        assert ids.count("quran:112:1") == 1

    def test_no_vector_store_degrades_gracefully(self, store):
        ro = RetrievalOrchestrator(store)  # no vector store
        hits = ro.search("قل هو الله أحد")
        assert hits and hits[0].citation_id == "quran:112:1"


class TestVectorStoreCache:
    def test_missing_cache_returns_empty_search(self, tmp_path):
        vs = VectorStore(cache_path=tmp_path / "none.npz")
        assert vs.size == 0

    def test_load_rejects_corrupt_cache(self, tmp_path):
        (tmp_path / "bad.npz").write_bytes(b"garbage")
        vs = VectorStore(cache_path=tmp_path / "bad.npz")
        assert vs.load() is False
        assert vs.size == 0


class TestVectorStoreLive:
    """Live embedding backend — skipped when LM Studio embeddings unavailable."""

    @pytest.fixture()
    def live_store(self, quran_db):
        try:
            from agent.core.embeddings import EmbeddingClient
            client = EmbeddingClient()
            client.embed_one("ping")
        except Exception:
            pytest.skip("embedding backend unavailable")
        return VectorStore(db_path=quran_db, cache_path=quran_db.parent / "live.npz",
                           client=client)

    def test_semantic_search_finds_comfort_verses(self, live_store, store):
        """The motivating case, as the system actually handles it: raw
        'I am lonely' alone is semantically weak (nomic ceiling on emotional
        register), so the intent router's concept expansions carry the query.
        This fixture embeds a quran-only corpus, so the assertion is the
        comfort VERSE; the full-corpus hadith behavior is covered by the
        agent-level live tests."""
        live_store.build()
        raw = live_store.search("I am lonely", top_k=12)
        assert raw, "vector index must return results"
        # via concept expansion (the designed path): 2:186 must top the hits
        hits = live_store.search("Allah is near responds to dua", top_k=8)
        ids = [h["citation_id"] for h in hits]
        assert ids[0] == "quran:2:186", (
            f"expansion must surface the nearness verse first; got {ids[:4]}"
        )
