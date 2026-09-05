"""Golden companion tests (fixme_v2 §30, §38-43).

These verify ROUTING and POLICY, never exact generated wording (§7 'The test
should evaluate behavior, not exact text matching'). Uses a scripted model
so all acceptance tests are deterministic and model-free.
"""

from __future__ import annotations

import pytest

from agent.context.builder import ContextBuilder
from agent.core.harness import CompanionHarness
from agent.memory.router import MemoryRouter
from agent.policy.companion_policy import CompanionPolicyEngine
from agent.safety.router import safety_route
from agent.state.manager import StateManager
from agent.state.models import Mode, Risk
from agent.validators.companion_validator import ResponseValidator


class ScriptedRouter:
    """Perfect-citizen scripted model; records prompts."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.prompts: list[str] = []

    def chat(self, task_class, messages, tools=None, max_tokens=1200, **kw):
        from agent.core.model import ModelResponse

        self.prompts.append(messages[0].content)
        text = self.responses.pop(0) if self.responses else "Okay, I hear you."
        return ModelResponse(content=text, tool_calls=[], finish_reason="stop")


def _harness(router, memory=None, retrieval=None) -> CompanionHarness:
    return CompanionHarness(
        router, retrieval=retrieval,
        memory_router=memory,
        states=StateManager(),
        policy_engine=CompanionPolicyEngine(),
        context_builder=ContextBuilder(),
        validator=ResponseValidator(),
    )


@pytest.fixture()
def memory(tmp_path):
    from agent.companion.memory import CompanionMemory

    return MemoryRouter(CompanionMemory(db_path=tmp_path / "m.db"))


# --------------------------------------------------------------- §38 lonely
class TestLonelyGolden:
    def test_route_equivalent(self, memory):
        """§38: intent=emotional_support, emotion=loneliness, risk=low,
        mode=companion, rag=false, followup=true."""
        h = _harness(ScriptedRouter(["I'm sorry. Want to tell me more?"]), memory=memory)
        r = h.respond("s", "I feel lonely.")
        assert r.trace["intent"] == "loneliness" or r.state["intent"] == "loneliness"
        assert r.trace["risk"] == "low"
        assert r.mode is Mode.COMPANION
        assert r.policy["requires_rag"] is False
        assert r.policy["requires_followup"] is True

    def test_response_behavior(self, memory):
        """acknowledge + warm + concise + no preach + no fabrication signals
        + max one question (the §38 response contract)."""
        text = ("I'm sorry you're feeling that way. Feeling alone can be heavy. "
                "Would you like to tell me a little about it?")
        h = _harness(ScriptedRouter([text]), memory=memory)
        r = h.respond("s", "I feel lonely.")
        assert r.companion_validation["ok"], r.companion_validation
        assert "?" not in r.answer or r.answer.count("?") <= 1

    def test_no_quran_dump_on_lonely(self, memory):
        """§38: no immediate verse dump — evidence must NOT be retrieved."""
        h = _harness(ScriptedRouter(["I hear you."]), memory=memory)
        r = h.respond("s", "I feel lonely.")
        assert r.trace["rag_used"] is False


# --------------------------------------------------- §39 explicit islamic ask
class TestIslamicQuestionGolden:
    def test_islam_loneliness_routes_rag(self, memory):
        h = _harness(ScriptedRouter(["The Quran nearness verse is [quran:2:186]."]),
                     memory=memory)
        r = h.respond("s", "What does Islam say about loneliness?")
        assert r.trace["intent"] == "islamic_question"
        assert r.policy["requires_rag"] is True
        assert r.mode in (Mode.QA, Mode.COMPANION)

    def test_dua_request_routes_rag(self, memory):
        h = _harness(ScriptedRouter(["A dua..."]), memory=memory)
        r = h.respond("s", "Give me a dua for sadness.")
        assert r.policy["requires_rag"] is True


# --------------------------------------------------------------- §40 simple
class TestSimpleChatGolden:
    def test_good_morning_no_rag(self, memory):
        h = _harness(ScriptedRouter(["Good morning! How are you?"]), memory=memory)
        r = h.respond("s", "Good morning.")
        assert r.trace["intent"] == "normal_chat"
        assert r.policy["requires_rag"] is False
        assert r.trace["rag_used"] is False


# -------------------------------------------------------------- §41 memory
class TestMemoryGolden:
    def test_study_memory_retrieved_later(self, memory):
        h = _harness(ScriptedRouter([" noted. ", " Picking up Al-Kahf again."]),
                     memory=memory)
        h.respond("s", "I'm studying Surah Al-Kahf.")
        h.respond("s", "Let's continue our study.")
        # the study fact was routed into long-term memory on turn 1
        assert any("Al-Kahf" in f["fact"] for f in memory.memory.facts())
        assert h.states.machine("s").state.durable_notes

    def test_transient_not_saved(self, memory):
        h = _harness(ScriptedRouter(["I hear you."]), memory=memory)
        h.respond("s", "I feel lonely.")
        facts = memory.memory.facts()
        assert all("lonely" not in f["fact"].lower() for f in facts)


# ---------------------------------------------------------- §42 topic switch
class TestTopicSwitchGolden:
    def test_study_to_companion_switch(self, memory):
        h = _harness(ScriptedRouter([
            "This hadith means deeds are by intention.",
            "That sounds heavy. Tell me more?",
        ]), memory=memory)
        h.respond("s", "Explain the hadith about intentions.")
        r = h.respond("s", "Actually, I'm feeling lonely today.")
        assert r.mode is Mode.COMPANION
        assert "loneliness" in r.state["intent"] or r.state["emotion"] == "loneliness"

    def test_companion_to_study_return(self, memory):
        """§43: return to study — mode transitions naturally with RAG."""
        h = _harness(ScriptedRouter([
            "I hear you, that's heavy.",
            "The hadith says deeds depend on intention [hadith:sahih-bukhari:1].",
        ]), memory=memory)
        h.respond("s", "I'm feeling lonely.")
        r = h.respond("s", "Thanks. Now can we continue the hadith about intentions?")
        assert r.mode in (Mode.STUDY, Mode.QA)
        assert r.policy["requires_rag"] is True


# --------------------------------------------------------------- §19 safety
class TestSafetyGolden:
    @pytest.mark.parametrize("message", [
        "I want to kill myself",
        "saya nak bunuh diri",
    ])
    def test_high_risk_never_reaches_model(self, memory, message):
        class FailRouter:
            def chat(self, *a, **kw):
                raise AssertionError("model invoked on high-risk input")

        h = _harness(FailRouter(), memory=memory)
        r = h.respond("s", message)
        assert r.mode is Mode.CRISIS
        assert r.policy["safety_override"] is True
        assert "emergency" in r.answer.lower() or "kecemasan" in r.answer

    def test_elevated_risk_companion_calmer(self, memory):
        """moderate distress: model still allowed, gentler policy."""
        h = _harness(ScriptedRouter(["I hear you. That sounds really heavy."]), memory=memory)
        r = h.respond("s", "I feel worthless and hopeless about everything.")
        assert r.trace["risk"] == "elevated"
        assert r.policy["allow_islamic_reflection"] is False
        assert r.mode is Mode.COMPANION

    def test_safety_route_independence(self):
        """§19: safety decisions are a standalone module, not derived from
        companion policy."""
        assert safety_route("I want to die").risk is Risk.HIGH
        assert safety_route("I had a bad day").risk is Risk.LOW


# ----------------------------------------------------- §27 multi-turn scenario
class TestEmptyQAOutput:
    """Observed live failure (gemma-4-12b): QA question, evidence retrieved,
    model returned empty content -> harness answered with the companion
    'I hear you' fallback. Wrong answer semantics for QA mode."""

    def test_empty_qa_retry_then_notice(self):
        class EmptyRouter:
            def __init__(self):
                self.calls = 0

            def chat(self, task_class, messages, tools=None, max_tokens=1200, **kw):
                from agent.core.model import ModelResponse

                self.calls += 1
                # QA route must never return the listening fallback
                assert task_class == "complex_rag", (
                    f"QA question must route to complex_rag, got {task_class}"
                )
                return ModelResponse(content="", tool_calls=[], finish_reason="length")

        h = _harness(EmptyRouter(), memory=None)
        h.retrieval = None  # still QA-mode: islamic question -> rag policy
        r = h.respond("s-pillars", "What's the pillar of islam?")
        assert "I hear you" not in r.answer
        assert "could not verify" in r.answer.lower()
        assert r.trace["route"] in ("rag", "chat", "qa")

    def test_empty_companion_keeps_listening(self):
        """Companion mode empty output keeps the empathic fallback — that IS
        the right semantics there."""
        class EmptyCompanionRouter:
            def chat(self, task_class, messages, tools=None, max_tokens=1200, **kw):
                from agent.core.model import ModelResponse

                return ModelResponse(content="", tool_calls=[], finish_reason="length")

        h = _harness(EmptyCompanionRouter(), memory=None)
        h.retrieval = None
        r = h.respond("s-lonely", "I feel lonely.")
        assert "I hear you" in r.answer or "listening" in r.answer.lower()

    def test_qa_listening_fallback_fails_validation(self):
        """Evidence exists + question route: 'I hear you' must be a
        companion-validation failure, not a pass."""
        from agent.policy.companion_policy import ResponsePolicy
        from agent.validators.companion_validator import ResponseValidator

        policy = ResponsePolicy(route="rag")
        v = ResponseValidator().validate(
            "I hear you. If you want to tell me more, I'm listening.",
            policy, evidence_present=True,
        )
        assert not v.ok
        assert any("non-answer" in p for p in v.companion_problems)


class TestMultiTurnContinuity:
    def test_loneliness_thread(self, memory):
        """fixme_v2 §27 example: 4-turn emotional thread keeps state continuity."""
        responses = [
            "I'm sorry. Want to tell me more?",
            "That sounds isolating. What's that like for you?",
            "Carrying that for a while is exhausting.",
            "Even feeling that way, you reaching out says something.",
        ]
        h = _harness(ScriptedRouter(list(responses)), memory=memory)
        turns = [
            "I feel lonely.",
            "Yeah. I don't really have anyone to talk to.",
            "I've felt this way for a while.",
            "I don't know if Allah even hears me.",
        ]
        modes = [h.respond("s", t).mode for t in turns]
        assert all(m is Mode.COMPANION for m in modes)
        # state continuity: turn count advanced, emotion retained
        final = h.states.machine("s").state
        assert final.turn_count >= 8  # 4 user + 4 assistant turns
        assert final.emotion is not None

    def test_long_thread_stops_re_asking(self, memory):
        """§5 policy: late in an emotional thread, follow-up pressure drops."""
        h = _harness(ScriptedRouter(["ok"] * 5), memory=memory)
        for i in range(4):
            h.respond("s", f"I still feel lonely, day {i}")
        r = h.respond("s", "I still feel lonely, day 5")
        assert r.policy["requires_followup"] is False
