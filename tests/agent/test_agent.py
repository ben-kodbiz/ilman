from __future__ import annotations

import pytest

from agent.core.agent import AgentOrchestrator
from agent.memory.store import MemoryStore
from agent.tools.layer import ToolLayer
from ingestion.hadith_ingest import HadithIngestor, HadithStore
from ingestion.quran_ingest import QuranIngestor, QuranStore, TranslationIngestor
from retrieval.hybrid import RetrievalOrchestrator


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """Real corpora + tools; the router is what we swap (mock or live)."""
    db = tmp_path_factory.mktemp("agent") / "k.db"
    QuranIngestor(db_path=db).ingest()
    TranslationIngestor(db_path=db).ingest()
    HadithIngestor(db_path=db).ingest_all()
    store = QuranStore(db_path=db)
    hadith_store = HadithStore(db_path=db)
    from agent.policy.source_policy import SourcePolicy, SourceRegistry
    policy = SourcePolicy(SourceRegistry.load())
    memory = MemoryStore(db_path=db.parent / "memory.db")
    tools = ToolLayer(policy, store=store, hadith_store=hadith_store, memory=memory)
    ro = RetrievalOrchestrator(store, hadith_store=hadith_store)
    return ro, tools, memory


class MockRouter:
    """Deterministic mock: never calls a model; simulates a good citizen model."""

    def __init__(self, final_answer: str):
        self.final_answer = final_answer
        self.calls = 0

    def chat(self, task_class, messages, tools=None, max_tokens=4096, **kw):
        from agent.core.model import ModelResponse

        self.calls += 1
        return ModelResponse(content=self.final_answer, tool_calls=[], finish_reason="stop")


class MockToolCallRouter:
    """First call: requests get_ayah(2,255); then answers citing it."""

    def __init__(self):
        self.calls = 0

    def chat(self, task_class, messages, tools=None, max_tokens=4096, **kw):
        from agent.core.model import ModelResponse, ToolCall

        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="", tool_calls=[ToolCall("get_ayah", {"surah": 2, "ayah": 255})],
                finish_reason="tool_calls",
            )
        # second round: after tool result, answer with a valid citation
        assert any(m.role == "tool" for m in messages), "tool result must be fed back"
        return ModelResponse(
            content="Ayat al-Kursi is [quran:2:255]: Allah — there is no deity except Him.",
            tool_calls=[], finish_reason="stop",
        )


class MockRepairRouter:
    """Answers with a BAD citation first; repairs when challenged."""

    def __init__(self):
        self.calls = 0

    def chat(self, task_class, messages, tools=None, max_tokens=4096, **kw):
        from agent.core.model import ModelResponse

        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="As stated in [quran:99:99], the verse says...",
                tool_calls=[], finish_reason="stop",
            )
        return ModelResponse(
            content="As stated in [quran:112:1], Say: He is Allah, the One.",
            tool_calls=[], finish_reason="stop",
        )


def _agent(stack, router):
    ro, tools, memory = stack
    return AgentOrchestrator(router, ro, tools, memory=memory)


class TestAgentLoop:
    def test_direct_answer_with_valid_citation(self, stack):
        agent = _agent(stack, MockRouter("The verse [quran:112:1] says: Say, He is Allah, the One."))
        result = agent.answer("What does Surah 112 say about Allah being One?")
        assert result.verified
        assert result.citations == ["quran:112:1"]
        assert result.trace.tool_calls == []

    def test_tool_call_loop_fetches_and_cites(self, stack):
        agent = _agent(stack, MockToolCallRouter())
        result = agent.answer("What does Ayat al-Kursi say?")
        assert result.verified
        assert "quran:2:255" in result.citations
        assert result.trace.tool_calls[0]["name"] == "get_ayah"
        assert result.trace.tool_calls[0]["ok"]

    def test_repair_round_fixes_bad_citation(self, stack):
        agent = _agent(stack, MockRepairRouter())
        result = agent.answer("What does Surah 112 say?")
        assert result.verified, f"repair failed: {result.unsupported_citations}"
        assert result.citations == ["quran:112:1"]
        assert result.trace.rounds >= 1

    def test_no_evidence_refuses_without_model_call(self, stack):
        class FailRouter:
            def chat(self, *a, **kw):
                    raise AssertionError("model must not be called when there is no evidence")

        agent = _agent(stack, FailRouter())
        # nothing in any corpus matches these content words
        result = agent.answer("zxqv blorptax frumious bandersnatch")
        assert result.refused
        assert result.answer == "I could not verify this from the approved source corpus."
        assert result.verified  # refusal is the correct, verified behavior

    def test_hadith_lookup_seeds_reference_evidence(self, stack):
        agent = _agent(stack, MockRouter("The hadith [hadith:sahih-bukhari:1] says deeds are by intentions."))
        result = agent.answer("What is hadith no. 1 in Bukhari about?")
        assert result.verified
        assert "hadith:sahih-bukhari:1" in result.citations
        hadith_evidence = [e for e in result.evidence.passages if e.citation_id == "hadith:sahih-bukhari:1"]
        assert hadith_evidence and hadith_evidence[0].tier == 1

    def test_study_history_recorded(self, stack):
        ro, tools, memory = stack
        before = len(memory.history())
        agent = _agent(stack, MockRouter("The verse [quran:112:1] says He is One."))
        agent.answer("What does Surah 112 say?")
        after = len(memory.history())
        assert after >= before + 1  # query recorded with verified citations

    def test_trace_stores_no_reasoning(self, stack):
        """§0: never expose chain-of-thought; trace keeps only decisions."""
        agent = _agent(stack, MockToolCallRouter())
        result = agent.answer("What does Ayat al-Kursi say?")
        blob = str(result.trace.to_dict())
        assert "reasoning" not in blob.lower()
        assert "thought" not in blob.lower()


class TestAgentLive:
    """Live model test — skipped when LM Studio is unreachable."""

    @pytest.fixture()
    def live_agent(self, stack):
        from agent.core.config import load_config
        from agent.core.model import ChatMessage, ModelRouter
        try:
            router = ModelRouter(load_config())
            backend, model_id = router.resolve("complex_rag")
            backend.chat(model_id, [ChatMessage(role="user", content="Say OK")], max_tokens=100)
        except Exception:
            pytest.skip("routed model not loaded in backend")
        return _agent(stack, router)

    def test_live_grounded_answer(self, live_agent):
        result = live_agent.answer("What does Surah 112 say about Allah being One?")
        assert result.evidence.passages
        if not result.refused:
            assert result.verified, f"unsupported: {result.unsupported_citations}"
            assert result.citations

    def test_live_fabrication_trap(self, live_agent):
        result = live_agent.answer(
            "I heard whoever reads Surah Ya-Sin on Friday gets the reward of "
            "ten hajj. Verify this hadith and cite its number and grading."
        )
        assert result.refused or result.verified  # never an unverified confident claim

    def test_live_tool_usage(self, live_agent):
        result = live_agent.answer("What is hadith number 1 in Bukhari? Quote it.")
        assert result.evidence.passages
        if not result.refused:
            assert result.verified
