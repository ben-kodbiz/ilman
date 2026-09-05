from __future__ import annotations

import json

import pytest

from agent.companion.logging import (
    LOG_DIR,
    CompanionLogger,
    disable_session,
    is_enabled,
)
from agent.companion.memory import CompanionMemory
from agent.context.builder import ContextBuilder
from agent.core.harness import CompanionHarness
from agent.memory.router import MemoryRouter
from agent.policy.companion_policy import CompanionPolicyEngine
from agent.state.manager import StateManager
from agent.validators.companion_validator import ResponseValidator
from agent.validators.pipeline import CitationValidator
from retrieval.hybrid import RetrievedPassage


@pytest.fixture()
def logger(tmp_path):
    return CompanionLogger(log_dir=tmp_path / "logs")


def _hadith(cid, en):
    return RetrievedPassage(
        citation_id=cid, surah=0, ayah=0, arabic="", translation=en,
        source_id=cid.split(":")[1], tier=1, leg="hadith", score=-1.0,
        collection=cid.split(":")[1], hadithnumber=int(cid.split(":")[2]),
    )


class ScriptedRouter:
    def __init__(self, answers):
        self.answers = list(answers)

    def chat(self, task, messages, tools=None, max_tokens=1200, **kw):
        from agent.core.model import ModelResponse

        return ModelResponse(
            content=self.answers.pop(0) if self.answers else "I hear you.",
            tool_calls=[], finish_reason="stop",
        )


class FakeRetrieval:
    hadith_store = None  # noqa: RUF012
    store = None  # noqa: RUF012
    vector_store = None  # noqa: RUF012

    def search(self, query, limit=6, concept_expansions=None, semantic_only=False):
        return [_hadith(
            "hadith:sahih-bukhari:6369",
            'Allah\'s Messenger used to seek refuge saying: "O Allah! I seek '
            'refuge in You from worry and grief."',
        )]


def _harness(router, tmp_path, logger):
    memory = MemoryRouter(CompanionMemory(db_path=tmp_path / "m.db"))
    return CompanionHarness(
        router, retrieval=FakeRetrieval(), memory_router=memory,
        states=StateManager(), policy_engine=CompanionPolicyEngine(),
        context_builder=ContextBuilder(), validator=ResponseValidator(),
        citation_validator=CitationValidator(), chat_logger=logger,
    )


class TestLoggingCapture:
    def test_normal_turn_logged(self, tmp_path, logger):
        h = _harness(ScriptedRouter(["I hear you. That sounds heavy."]), tmp_path, logger)
        h.respond("prod-session-1", "I feel lonely.")
        files = list((tmp_path / "logs").glob("*.jsonl"))
        assert len(files) == 1
        rec = json.loads(files[0].read_text().splitlines()[0])
        assert rec["user"]["text"] == "I feel lonely."
        assert rec["companion"]["text"].startswith("I hear you")
        assert rec["companion"]["mode"] == "companion"
        assert rec["schema"] == 1
        assert rec["sensitive"] is False

    def test_crisis_turn_marked_sensitive(self, tmp_path, logger):
        class FailRouter:
            def chat(self, *a, **kw):
                raise AssertionError("model must not be called")

        h = _harness(FailRouter(), tmp_path, logger)
        h.respond("prod-session-2", "I want to kill myself")
        rec = json.loads(
            list((tmp_path / "logs").glob("*.jsonl"))[0].read_text().splitlines()[0]
        )
        assert rec["sensitive"] is True
        assert rec["companion"]["mode"] == "crisis"
        assert rec["companion"]["risk"] == "high"

    def test_rag_turn_captures_evidence_metadata(self, tmp_path, logger):
        answer = (
            'The Prophet ﷺ taught: "O Allah! I seek refuge in You from worry '
            'and grief" [hadith:sahih-bukhari:6369].'
        )
        h = _harness(ScriptedRouter([answer]), tmp_path, logger)
        h.respond("prod-session-3", "Is there any dua for grief?")
        rec = json.loads(
            list((tmp_path / "logs").glob("*.jsonl"))[0].read_text().splitlines()[0]
        )
        assert rec["citations"] == ["hadith:sahih-bukhari:6369"]
        assert rec["companion"]["intent"] == "dua_request"
        assert rec["evidence"]["status"]  # judge status recorded
        assert rec["policy"]  # policy decision recorded

    def test_no_cot_logged(self, tmp_path, logger):
        """§31 invariant: chain-of-thought never appears in records — only
        the trace's decision fields. The DebugTrace has no CoT field, and the
        record schema whitelists fields explicitly."""
        h = _harness(ScriptedRouter(["I hear you."]), tmp_path, logger)
        h.respond("prod-session-4", "I feel anxious.")
        raw = list((tmp_path / "logs").glob("*.jsonl"))[0].read_text()
        assert "reasoning" not in raw.lower()
        assert "<think" not in raw.lower()

    def test_eval_sessions_excluded(self, tmp_path, logger):
        h = _harness(ScriptedRouter(["I hear you."]), tmp_path, logger)
        h.respond("case-lonely_001", "I feel lonely.")
        h.respond("scen-x", "I feel lonely.")
        h.respond("eval-model", "I feel lonely.")
        assert not list((tmp_path / "logs").glob("*.jsonl"))

    def test_session_optout(self, tmp_path, logger):
        disable_session("opted-out")
        assert not is_enabled("opted-out")
        h = _harness(ScriptedRouter(["ok"]), tmp_path, logger)
        h.respond("opted-out", "I feel lonely.")
        assert not list((tmp_path / "logs").glob("*.jsonl"))


class TestLogReading:
    def test_read_sessions_and_turns(self, tmp_path, logger):
        h = _harness(ScriptedRouter(["ok", "ok2"]), tmp_path, logger)
        h.respond("prod-a", "first")
        h.respond("prod-a", "second")
        sessions = logger.read_sessions(tmp_path / "logs")
        assert sessions == ["prod-a"]
        turns = logger.read_turns("prod-a", tmp_path / "logs")
        assert len(turns) == 2
        assert turns[0]["user"]["text"] == "first"

    def test_sensitive_exclusion(self, tmp_path, logger):
        class FailRouter:
            def chat(self, *a, **kw):
                raise AssertionError("no model in crisis")

        h = _harness(FailRouter(), tmp_path, logger)
        h.respond("prod-b", "I want to die")  # sensitive
        h = _harness(ScriptedRouter(["ok"]), tmp_path, logger)
        h.respond("prod-b", "hello again")
        all_turns = logger.read_turns("prod-b", tmp_path / "logs")
        safe_turns = logger.read_turns(
            "prod-b", tmp_path / "logs", include_sensitive=False
        )
        assert len(all_turns) == 2
        assert len(safe_turns) == 1


class TestMarkdownExport:
    def test_export_creates_readable_transcript(self, tmp_path, logger):
        h = _harness(ScriptedRouter(["I hear you. Want to talk about it?"]), tmp_path, logger)
        h.respond("prod-c", "I feel lonely today.")
        out = tmp_path / "export" / "prod-c.md"
        path = CompanionLogger.export_markdown("prod-c", out, tmp_path / "logs")
        content = path.read_text()
        assert "session `prod-c`" in content
        assert "**User:** I feel lonely today." in content
        assert "**Ilman:** I hear you." in content
        assert "mode: **companion**" in content

    def test_export_redacted(self, tmp_path, logger):
        h = _harness(ScriptedRouter(["I hear you."]), tmp_path, logger)
        h.respond("prod-d", "my secret feeling")
        out = tmp_path / "export" / "prod-d.md"
        content = CompanionLogger.export_markdown(
            "prod-d", out, tmp_path / "logs", redact=True
        ).read_text()
        assert "my secret feeling" not in content
        assert "[REDACTED]" in content

    def test_stats_aggregation(self, tmp_path, logger):
        h = _harness(ScriptedRouter(["I hear you.", "I hear you."]), tmp_path, logger)
        h.respond("prod-e", "I feel lonely.")
        h.respond("prod-e", "still lonely.")
        stats = CompanionLogger.stats(tmp_path / "logs")
        assert stats["sessions"] == 1
        assert stats["turns"] == 2
        assert stats["modes"]["companion"] == 2


class TestProductionLoggerDefault:
    def test_default_logger_writes_to_knowledge_dir(self, tmp_path, monkeypatch):
        """The production logger writes under knowledge/processed — verify
        path construction only (no writes in tests)."""
        prod = CompanionLogger()
        assert prod.log_dir == LOG_DIR
        assert "companion_logs" in str(LOG_DIR)
