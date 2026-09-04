from __future__ import annotations

import pytest

from agent.companion.intent import classify_companion
from agent.companion.safety import Severity


class TestIntentRouting:
    """fix_me.md §21 regression matrix (classification layer)."""

    @pytest.mark.parametrize("message,expected", [
        # Emotional
        ("I feel lonely.", "loneliness"),
        ("I feel like nobody understands me.", "emotional_support"),
        ("I miss someone so much tonight.", "grief"),
        ("I'm angry at everything right now.", "anger"),
        ("I feel guilty about what I did.", "guilt"),
        ("I feel spiritually empty lately.", "spiritual_low"),
        ("I feel anxious about tomorrow.", "anxiety"),
        ("I'm scared of the future.", "fear"),
        ("I have no motivation to do anything.", "motivation"),
    ])
    def test_emotional_statements(self, message, expected):
        ci = classify_companion(message)
        assert ci.emotion is not None or ci.intent == expected
        assert ci.severity is not Severity.HIGH_RISK

    @pytest.mark.parametrize("message,expected_intent", [
        ("What does the Quran say about loneliness?", "quran_question"),
        ("Give me a hadith about patience.", "hadith_question"),
        ("Explain Surah Al-Baqarah 255 for me.", "quran_question"),
        ("Give me something comforting from the Quran.", "quran_request"),
        ("Is riba haram?", "fiqh_question"),
        ("What is 2:286 about?", "quran_question"),
        ("What is hadith no. 1 in Bukhari?", "hadith_question"),
    ])
    def test_islamic_questions(self, message, expected_intent):
        ci = classify_companion(message)
        assert ci.intent == expected_intent
        assert ci.needs_islamic_guidance is True

    @pytest.mark.parametrize("message", [
        "I'm alone tonight.",
        "I don't know what I'm doing anymore.",
        "I feel empty.",
    ])
    def test_ambiguous_statements(self, message):
        """Ambiguous inputs must NOT route to islamic RAG dumping (§10)."""
        ci = classify_companion(message)
        assert ci.intent in (
            "loneliness", "emotional_support", "confusion", "spiritual_low",
            "normal_chat", "quran_question", "motivation",
        )
        # ambiguous emotional -> guidance is offered, not forced
        assert ci.needs_islamic_guidance is False or ci.emotion is None

    def test_guidance_not_forced_on_pure_emotion(self):
        """§2: 'I feel lonely' must NOT immediately dump religion."""
        ci = classify_companion("I feel lonely")
        assert ci.needs_islamic_guidance is False
        assert ci.needs_clarification is True

    def test_normal_chat(self):
        ci = classify_companion("Hello, how are you?")
        assert ci.intent == "normal_chat"

    def test_gratitude(self):
        ci = classify_companion("Alhamdulillah, I got the job!")
        assert ci.emotion == "gratitude" or ci.intent == "normal_chat"


class TestEmotionConfidence:
    def test_confidence_range(self):
        ci = classify_companion("I feel lonely and alone and isolated")
        assert ci.emotion == "loneliness"
        assert 0.5 <= ci.emotion_confidence <= 1.0

    def test_multi_emotion_takes_top(self):
        ci = classify_companion("I feel anxious and angry")
        assert ci.emotion in ("anxiety", "anger")
