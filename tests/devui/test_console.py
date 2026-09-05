# ruff: noqa: E402  (gradio import guard precedes repo imports)
from __future__ import annotations

import pytest

gradio = pytest.importorskip("gradio")

from agent.companion.memory import CompanionMemory
from agent.context.builder import ContextBuilder
from agent.core.harness import CompanionHarness
from agent.memory.router import MemoryRouter
from agent.policy.companion_policy import CompanionPolicyEngine
from agent.state.manager import StateManager
from agent.validators.companion_validator import ResponseValidator


class ScriptedRouter:
    def chat(self, task, messages, tools=None, max_tokens=1200, **kw):
        from agent.core.model import ModelResponse

        return ModelResponse(
            content="I hear you. That sounds heavy.",
            tool_calls=[], finish_reason="stop",
        )


@pytest.fixture()
def harness(tmp_path):
    memory = MemoryRouter(CompanionMemory(db_path=tmp_path / "m.db"))
    return CompanionHarness(
        ScriptedRouter(), retrieval=None, memory_router=memory,
        states=StateManager(), policy_engine=CompanionPolicyEngine(),
        context_builder=ContextBuilder(), validator=ResponseValidator(),
    )


class TestConsoleCompanionFlow:
    """The DevUI companion path: the exact generator the UI calls."""

    def _run(self, harness, message, history=None):
        from apps.devui import console

        # patch the module-level harness factory to our scripted one
        console._HARNESSES["test-role"] = harness
        out_history = []
        trace_text = status = None
        # signature: (message, history, session, model_role, show_trace)
        for produced in console.companion_chat(
            message, history or [], "sess", "test-role", True
        ):
            out_history, trace_text, status = produced
        return out_history, trace_text

    def test_lonely_flow_produces_companion_reply(self, harness):
        history, trace = self._run(harness, "I feel lonely.")
        assert history[-1]["role"] == "assistant"
        assert "I hear you" in history[-1]["content"]
        assert '"intent"' in trace  # developer trace rendered as JSON

    def test_crisis_flow_never_calls_model(self, harness):
        class FailRouter(ScriptedRouter):
            def chat(self, *a, **kw):
                raise AssertionError("model invoked on crisis input")

        fail_harness = CompanionHarness(
            FailRouter(), retrieval=None,
            memory_router=harness.memory_router, states=StateManager(),
            policy_engine=CompanionPolicyEngine(), context_builder=ContextBuilder(),
            validator=ResponseValidator(),
        )
        history, trace = self._run(fail_harness, "I want to kill myself")
        assert "emergency" in history[-1]["content"].lower()
        assert '"risk": "high"' in trace or "high" in trace

    def test_history_accumulates(self, harness):
        h1, _ = self._run(harness, "I feel lonely.")
        h2, _ = self._run(harness, "It's been like this for a while.", history=h1)
        assert len(h2) == 4  # 2 user + 2 assistant
        assert h2[0]["role"] == "user"

    def test_backend_error_surfaces_cleanly(self, harness):
        from apps.devui import console

        class ExplodingRouter(ScriptedRouter):
            def chat(self, *a, **kw):
                raise RuntimeError("LM Studio down")

        bad = CompanionHarness(
            ExplodingRouter(), retrieval=None,
            memory_router=harness.memory_router, states=StateManager(),
            policy_engine=CompanionPolicyEngine(), context_builder=ContextBuilder(),
            validator=ResponseValidator(),
        )
        console._HARNESSES["bad-role"] = bad
        out = None
        for produced in console.companion_chat("hi", [], "s", "bad-role", False):
            out = produced
        history = out[0]
        assert "[backend error]" in history[-1]["content"]

    def test_empty_message_noop(self, harness):
        from apps.devui import console

        console._HARNESSES["test-role"] = harness
        out = None
        for produced in console.companion_chat("  ", [], "s", "test-role", False):
            out = produced
        assert out[0] == []


class TestConsoleSearch:
    def test_raw_search_renders_citations(self):
        from apps.devui import console

        results = None
        for produced in console.raw_search("قل هو الله أحد", 4):
            results = produced
        assert "quran:112:1" in results

    def test_empty_query(self):
        from apps.devui import console

        out = None
        for produced in console.raw_search("", 4):
            out = produced
        assert out == ""


class TestAppBuild:
    def test_build_app(self):
        from apps.devui.console import build_app

        app = build_app()
        assert app is not None
