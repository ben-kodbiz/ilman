from __future__ import annotations

import pytest

from agent.validators.pipeline import (
    UNVERIFIABLE_NOTICE,
    CitationValidator,
    EvidencePack,
    ResponsePipeline,
)
from ingestion.quran_ingest import QuranStore
from retrieval.hybrid import RetrievalOrchestrator, RetrievedPassage


@pytest.fixture(scope="module")
def store():
    return QuranStore()


@pytest.fixture(scope="module")
def orchestrator(store):
    return RetrievalOrchestrator(store)


def _passage(citation="quran:112:1", surah=112, ayah=1, source_id="quran-uthmani-json"):
    return RetrievedPassage(
        citation_id=citation, surah=surah, ayah=ayah,
        arabic="قُلۡ هُوَ ٱللَّهُ أَحَدٌ",
        source_id=source_id, tier=0, leg="fts", score=-1.0,
    )


class TestCitationValidator:
    def test_quran_prefixed_citation_verified(self):
        pack = EvidencePack(query="q", passages=[_passage()])
        v = CitationValidator().validate("As stated [quran:112:1], He is One.", pack)
        assert v.ok and v.verified_citations == ["quran:112:1"]

    def test_plain_ref_citation_verified(self):
        pack = EvidencePack(query="q", passages=[_passage()])
        v = CitationValidator().validate("As stated (112:1), He is One.", pack)
        assert v.ok

    def test_citation_outside_pack_is_unsupported(self):
        pack = EvidencePack(query="q", passages=[_passage()])
        v = CitationValidator().validate("See [quran:2:255] for details.", pack)
        assert not v.ok
        assert v.unsupported_citations == ["quran:2:255"]

    def test_no_citations_at_all(self):
        pack = EvidencePack(query="q", passages=[_passage()])
        v = CitationValidator().validate("He is One, with no reference.", pack)
        assert v.ok  # nothing fabricated; caller decides if citations required
        assert not v.had_any_citation

    def test_empty_pack_all_citations_unsupported(self):
        pack = EvidencePack(query="q", passages=[])
        v = CitationValidator().validate("See [quran:112:1].", pack)
        assert not v.ok


class TestEvidencePack:
    def test_prompt_block_lists_all_passages(self, orchestrator):
        passages = orchestrator.search("قل هو الله أحد", limit=3)
        pack = EvidencePack(query="tawhid", passages=passages)
        block = pack.to_prompt_block()
        assert "[quran:112:1]" in block
        assert "قُلۡ" in block or "قل" in block

    def test_empty_pack_prompts_no_evidence(self):
        assert EvidencePack(query="q", passages=[]).to_prompt_block() == "NO EVIDENCE AVAILABLE."


class TestGroundedRefusal:
    def test_no_evidence_returns_unverifiable_notice(self, store):
        """§12: no evidence -> DO NOT GUESS, no model call at all."""
        pipeline = ResponsePipeline(router=None)
        result = pipeline.answer("quantum chromodynamics in fiqh", RetrievalOrchestrator(store))
        assert result.refused
        assert result.answer == UNVERIFIABLE_NOTICE
        assert result.evidence.passages == []


class TestEndToEndWithModel:
    """Live model tests — skipped when the routed backend model is unavailable."""

    @pytest.fixture()
    def pipeline(self):
        from agent.core.config import load_config
        from agent.core.model import ModelRouter
        try:
            router = ModelRouter(load_config())
            backend, model_id = router.resolve("complex_rag")
            backend.chat(model_id, [__import__("agent.core.model", fromlist=["ChatMessage"]).ChatMessage(
                role="user", content="Say OK")], max_tokens=100)
        except Exception:
            pytest.skip("routed model not loaded in backend")
        return ResponsePipeline(router)

    def test_grounded_answer_cites_evidence(self, pipeline, store):
        result = pipeline.answer("What does Surah 112 say about Allah being One?",
                                 RetrievalOrchestrator(store))
        assert result.evidence.passages, "expected 112 material"
        if not result.refused:
            assert result.verified, (
                f"validator rejected answer: {result.validation.unsupported_citations}"
            )
            assert result.validation.verified_citations

    def test_fabrication_trap_hits_notice(self, pipeline, store):
        """No matching evidence for a fake verse -> unverifiable notice."""
        result = pipeline.answer(
            "Explain the verse 'and whoever reads 3 pages on Friday gets Paradise' "
            "and cite its surah and ayah",
            RetrievalOrchestrator(store),
        )
        # Either the pipeline refused pre-model, or the answer came back and
        # every citation in it must still be validated against the pack.
        if not result.refused:
            assert result.verified
