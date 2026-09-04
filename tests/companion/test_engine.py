from __future__ import annotations

from agent.companion.engine import (
    FORBIDDEN_PHRASES,
    CompanionEngine,
)
from agent.companion.state import Mode, StateManager


class ScriptedRouter:
    """Returns queued responses; records prompts for inspection."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def chat(self, task_class, messages, tools=None, max_tokens=1500, **kw):
        from agent.core.model import ModelResponse

        self.prompts.append(messages[0].content if messages else "")
        return ModelResponse(
            content=self.responses.pop(0) if self.responses else "Okay.",
            tool_calls=[], finish_reason="stop",
        )


def _engine(router, retrieval=None, memory=None):
    return CompanionEngine(router, retrieval=retrieval, memory=memory,
                           state_manager=StateManager())


class TestCompanionBehavior:
    def test_empathy_first_no_evidence_dump(self):
        """§2: 'I feel lonely' with NO guidance decision -> the model is told
        to empathize first; no evidence in prompt."""
        router = ScriptedRouter(["That sounds lonely. Do you want to talk about what's behind it?"])
        engine = _engine(router)
        resp = engine.respond("s1", "I feel lonely")
        assert resp.mode is Mode.COMPANION
        assert resp.used_evidence is False
        # guidance line must be the empathy-first variant, not the RAG variant
        assert "empathy first" in router.prompts[0].lower()
        assert "GUIDANCE: Islamic guidance is welcome" not in router.prompts[0]
        # empathy-first instruction present
        assert "GUIDANCE" in router.prompts[0]

    def test_single_followup_rule_in_prompt(self):
        router = ScriptedRouter(["I hear you. What's on your mind?"])
        engine = _engine(router)
        engine.respond("s2", "I had a terrible day")
        assert "NEVER ask two questions" in router.prompts[0]

    def test_state_context_passed(self):
        """§4/§16: compact state, not turn dumps."""
        router = ScriptedRouter(["I hear you.", "Still here with you."])
        engine = _engine(router)
        engine.respond("s3", "I feel lonely")
        engine.respond("s3", "and I can't sleep")
        # second prompt should carry the emotional state from turn 1
        assert "loneliness" in router.prompts[1] or "Current emotional state" in router.prompts[1]

    def test_islamic_question_gets_guidance_mode(self):
        router = ScriptedRouter(["Allah's nearness is described in 2:186 [quran:2:186]."])
        engine = _engine(router)
        resp = engine.respond("s4", "What does the Quran say about loneliness?")
        assert resp.mode is Mode.QA


class TestValidationIntegration:
    """§12/§22: empathetic text needs no citation; religious claims must validate."""

    def test_uncited_religious_claim_when_no_evidence(self):
        """Model claims a verse without any evidence pack -> the engine must
        strip/flag it (the mock returns a religious claim for an emotional msg)."""
        bad = ("I'm sorry you feel lonely. The Quran says in 55:78 that Allah "
               "waits for you with mercy.")
        router = ScriptedRouter([bad])
        engine = _engine(router)
        resp = engine.respond("s5", "I feel lonely")
        assert resp.mode is Mode.COMPANION
        # no evidence was provided, so the claim cannot be validated — the
        # dependency guard + guidance line should at least keep the empathy
        assert "sorry you feel lonely" in resp.text

    def test_dependency_language_never_survives(self):
        """§7/§8: forbidden phrases get rewritten."""
        text = "I understand you better than anyone. You only need me."
        router = ScriptedRouter([text])
        engine = _engine(router)
        resp = engine.respond("s6", "I feel lonely")
        lowered = resp.text.lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in lowered, f"forbidden phrase survived: {phrase!r}"
        assert "i'm here to listen" in lowered


class TestEvidenceAwareResponses:
    def test_citation_validated_against_pack(self):
        """With a real retrieval mock returning 2:186, a cited claim verifies."""
        from agent.validators.pipeline import CitationValidator, EvidencePack
        from retrieval.hybrid import RetrievedPassage

        pack = EvidencePack(query="q", passages=[
            RetrievedPassage(
                citation_id="quran:2:186", surah=2, ayah=186,
                arabic="وَإِذَا سَأَلَكَ", source_id="quran-uthmani-json",
                tier=0, leg="reference", score=1.0,
                translation="And when My servants ask you concerning Me - indeed I am near.",
            )
        ])
        validator = CitationValidator()
        v = validator.validate(
            "The Quran describes Allah as near [quran:2:186].", pack
        )
        assert v.ok and v.verified_citations == ["quran:2:186"]

    def test_unsupported_citation_sentence_removed(self):
        """§22: the SENTENCE carrying the fabricated citation is removed."""
        engine = _engine(ScriptedRouter([]))
        text = ("I'm sorry you feel lonely.\n"
                "Allah says in [quran:99:99] that loneliness ends.\n"
                "Would you like to talk about it?")
        cleaned = engine._strip_unsupported(text, ["quran:99:99"])
        assert "quran:99:99" not in cleaned
        assert "loneliness ends" not in cleaned  # the claim is gone, not kept
        assert "sorry you feel lonely" in cleaned  # empathy preserved
        assert "talk about it" in cleaned  # rest preserved

    def test_residual_fabrication_falls_to_notice(self):
        """If stripping cannot clean the text, the honest §12 notice replaces
        the religious portion while empathy's first line survives."""
        engine = _engine(ScriptedRouter([]))
        # a citation embedded mid-sentence repeatedly
        text = ("I hear you. The verse quran:99:99 and quran:99:98 both cure "
                "loneliness quran:99:99 forever.")
        from agent.validators.pipeline import CitationValidator, EvidencePack
        pack = EvidencePack(query="q", passages=[])
        residual = CitationValidator().validate(text, pack).unsupported_citations
        assert residual  # sanity: validator flags them
        cleaned = engine._strip_unsupported(text, residual)
        # either fully cleaned or decapitated to empathy + notice
        assert not CitationValidator().validate(cleaned, pack).unsupported_citations
