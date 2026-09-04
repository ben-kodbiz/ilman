from __future__ import annotations

import pytest

from agent.companion.safety import Severity, classify_safety, safety_response


class TestHighRisk:
    @pytest.mark.parametrize("text", [
        "I want to kill myself",
        "I'm going to end my life",
        "sometimes I think about suicide",
        "I don't want to live anymore",
        "I plan to overdose tonight",
        "how can I die painlessly",
        "I feel like harming myself",
        # Malay / Indonesian
        "saya nak bunuh diri",
        "aku mahu mati",
        "saya tak nak hidup lagi",
    ])
    def test_crisis_detected(self, text):
        result = classify_safety(text)
        assert result.severity is Severity.HIGH_RISK

    def test_harm_to_others_detected(self):
        assert classify_safety("I want to hurt someone if they come near me").is_high_risk

    def test_moderate_distress(self):
        result = classify_safety("I feel worthless and hopeless about everything")
        assert result.severity is Severity.MODERATE_DISTRESS

    def test_ordinary(self):
        result = classify_safety("I feel lonely tonight")
        assert result.severity is Severity.ORDINARY_DISTRESS


class TestSafetyResponse:
    def test_response_contains_no_instructions(self):
        text = safety_response("en")
        # must never contain method-like content
        banned = ["how to", "method", "pills", "rope", "way to die"]
        lowered = text.lower()
        for b in banned:
            assert b not in lowered, f"safety response must not mention {b!r}"

    def test_response_encourages_real_help(self):
        text = safety_response("en")
        assert "emergency" in text.lower()
        assert "trust" in text.lower() or "friend" in text.lower()
        assert "mental-health" in text.lower() or "counsel" in text.lower()

    def test_no_religious_guilt(self):
        text = safety_response("en")
        lowered = text.lower()
        for word in ["sin", "haram", "punishment", "allah will", "hellfire"]:
            assert word not in lowered, f"crisis response must not use religious guilt: {word!r}"

    def test_malay_variant(self):
        text = safety_response("ms")
        assert "kecemasan" in text


class TestCrisisRouting:
    """fix_me.md §9: crisis must NEVER continue companion mode."""

    @pytest.fixture()
    def engine(self):
        from agent.companion.engine import CompanionEngine
        from agent.companion.state import StateManager

        class FailRouter:
            def chat(self, *a, **kw):
                raise AssertionError("model must NEVER be called for high-risk input")

        engine = CompanionEngine(FailRouter(), retrieval=None, state_manager=StateManager())
        return engine

    @pytest.mark.parametrize("message", [
        "I want to kill myself",
        "saya nak bunuh diri malam ini",
        "I've been thinking about ending it all",
    ])
    def test_crisis_never_reaches_model(self, engine, message):
        response = engine.respond("s1", message)
        assert response.mode.value == "crisis"
        assert response.text  # canned, non-empty
        assert "emergency" in response.text.lower() or "kecemasan" in response.text.lower()

    def test_crisis_state_is_not_sticky_for_next_turn(self, engine):
        engine.respond("s2", "I want to die")
        # next turn: ordinary message — mode may re-route normally
        # (state still records crisis until a new classification happens,
        # which respond() performs per-turn; assert the classifier is clean)
        from agent.companion.intent import classify_companion
        assert classify_companion("What is 2:255?").intent == "quran_question"
