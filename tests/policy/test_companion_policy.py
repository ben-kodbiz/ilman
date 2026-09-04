from __future__ import annotations

from agent.policy.companion_policy import (
    CompanionPolicyEngine,
    PolicyValidator,
    ResponsePolicy,
    Tone,
    Verbosity,
)
from agent.state.models import ConversationState, Mode, Risk, UserGoal


def _state(**kw) -> ConversationState:
    return ConversationState(session_id="s", **kw)


class TestPolicyDecisions:
    """§16/§17/§18: routing decisions are machine-readable."""

    def test_lonely_companion_no_rag(self):
        engine = CompanionPolicyEngine()
        policy = engine.decide(
            _state(mode=Mode.COMPANION, intent="emotional_support",
                   emotion="loneliness", user_goal=UserGoal.BE_HEARD,
                   requires_followup=True),
        )
        assert policy.route == "companion"
        assert policy.requires_rag is False
        assert policy.max_followups == 1
        assert policy.preach is False
        assert policy.verbosity is Verbosity.SHORT

    def test_islam_loneliness_rag(self):
        engine = CompanionPolicyEngine()
        policy = engine.decide(
            _state(intent="islamic_question", emotion="loneliness"),
            explicit_islamic=True, turn_is_question=True,
        )
        assert policy.requires_rag is True
        assert policy.acknowledge_first is True  # empathy then evidence

    def test_quran_question_rag(self):
        engine = CompanionPolicyEngine()
        policy = engine.decide(_state(intent="quran_question"))
        assert policy.requires_rag is True and policy.mode is Mode.QA

    def test_dua_request(self):
        engine = CompanionPolicyEngine()
        policy = engine.decide(_state(intent="dua_request"), explicit_islamic=True)
        assert policy.mode is Mode.DUA and policy.requires_rag is True

    def test_normal_chat(self):
        engine = CompanionPolicyEngine()
        policy = engine.decide(_state(intent="normal_chat"))
        assert policy.route == "chat"
        assert policy.requires_rag is False
        assert policy.requires_followup is False
        assert policy.acknowledge_first is False  # no fake empathy for 'hi'

    def test_safety_override_unchallengable(self):
        """§19: companion policy can NEVER soften safety."""
        engine = CompanionPolicyEngine()
        policy = engine.decide(_state(risk=Risk.HIGH, intent="emotional_support",
                                      emotion="loneliness"))
        assert policy.safety_override is True
        assert policy.requires_rag is False
        assert policy.allow_islamic_reflection is False

    def test_elevated_risk_no_religious_push(self):
        engine = CompanionPolicyEngine()
        policy = engine.decide(_state(risk=Risk.ELEVATED, emotion="loneliness"))
        assert policy.allow_islamic_reflection is False
        assert policy.tone is Tone.CALM

    def test_long_thread_stops_asking(self):
        engine = CompanionPolicyEngine()
        policy = engine.decide(
            _state(intent="emotional_support", emotion="loneliness"),
            turn_count=4,
        )
        assert policy.requires_followup is False

    def test_guidance_hold_preference(self):
        engine = CompanionPolicyEngine()
        policy = engine.decide(
            _state(intent="emotional_support", emotion="loneliness"),
            memory_preferred="hold",
        )
        assert policy.allow_islamic_reflection is False


class TestPolicyValidator:
    def test_too_many_questions(self):
        pv = PolicyValidator()
        policy = ResponsePolicy(max_followups=1)
        assert pv.validate("Why? What? How?", policy)

    def test_preachy_opener_detected(self):
        pv = PolicyValidator()
        policy = ResponsePolicy(preach=False)
        assert any("preachy" in p for p in pv.validate("Allah says be patient.", policy))

    def test_word_budget(self):
        pv = PolicyValidator()
        policy = ResponsePolicy(word_budget=50)
        long_text = " ".join(["word"] * 100)
        assert any("verbosity" in p for p in pv.validate(long_text, policy))

    def test_safety_guilt_banned(self):
        pv = PolicyValidator()
        policy = ResponsePolicy(safety_override=True)
        assert any("guilt" in p for p in pv.validate("This is haram of you.", policy))

    def test_clean_response_passes(self):
        pv = PolicyValidator()
        policy = ResponsePolicy()
        assert pv.validate("I hear you. That sounds heavy.", policy) == []
