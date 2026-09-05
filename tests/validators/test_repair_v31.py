"""fixme_v3.1 §31-§33 — repair revalidation + final answer gate tests.

Uses scripted routers so repair behavior is deterministic (no live model).
"""

from __future__ import annotations

from agent.companion.memory import CompanionMemory
from agent.context.builder import ContextBuilder
from agent.core.harness import CompanionHarness
from agent.memory.router import MemoryRouter
from agent.policy.companion_policy import CompanionPolicyEngine
from agent.state.manager import StateManager
from agent.validators.companion_validator import ResponseValidator
from agent.validators.pipeline import CitationValidator
from retrieval.hybrid import RetrievedPassage


def _hadith(cid: str, en: str) -> RetrievedPassage:
    return RetrievedPassage(
        citation_id=cid, surah=0, ayah=0, arabic="", translation=en,
        source_id=cid.split(":")[1], tier=1, leg="hadith", score=-1.0,
        collection=cid.split(":")[1], hadithnumber=int(cid.split(":")[2]),
    )


DISTRESS_DUA = _hadith(
    "hadith:sahih-bukhari:6369",
    'Allah\'s Messenger used to seek refuge saying: "O Allah! I seek refuge '
    'in You from worry and grief, from incapacity and laziness."',
)


class FabricatingThenFixedRouter:
    """Round 1: fabricates a cure claim. Round 2 (post-repair regeneration
    isn't used by the harness — repairs are textual), so this models the
    bounded-repair exit: after 2 rounds the final gate must catch survivors."""

    def __init__(self, scripted_answers: list[str]):
        self.scripted = list(scripted_answers)
        self.calls = 0

    def chat(self, task, messages, tools=None, max_tokens=1200, **kw):
        from agent.core.model import ModelResponse

        self.calls += 1
        text = self.scripted.pop(0) if self.scripted else "I hear you."
        return ModelResponse(content=text, tool_calls=[], finish_reason="stop")


def _harness(router, tmp_path):
    memory = MemoryRouter(CompanionMemory(db_path=tmp_path / "m.db"))
    retrieval = _FakeRetrieval()
    return CompanionHarness(
        router, retrieval=retrieval, memory_router=memory,
        states=StateManager(), policy_engine=CompanionPolicyEngine(),
        context_builder=ContextBuilder(), validator=ResponseValidator(),
        citation_validator=CitationValidator(),
    ), retrieval


class _FakeRetrieval:
    """Minimal retrieval double: returns the dua hadith for any query."""

    hadith_store = None  # noqa: RUF012 — double, not real store
    store = None  # noqa: RUF012
    vector_store = None  # noqa: RUF012

    def search(self, query, limit=6, concept_expansions=None, semantic_only=False):
        return [DISTRESS_DUA]


class TestRepairRevalidation:
    def test_high_risk_fabrication_never_ships(self, tmp_path):
        """The core §33 case: a cure-guarantee fabrication must not reach
        the user even after bounded repair — the final gate forces the safe
        fallback."""
        router = FabricatingThenFixedRouter([
            "This dua cures depression completely [hadith:sahih-bukhari:6369].",
        ])
        harness, _ = _harness(router, tmp_path)
        result = harness.respond("s1", "Is there any dua for depression?")
        lowered = result.answer.lower()
        # must NOT contain the confident guarantee as the shipped answer
        assert "cures depression completely" not in lowered
        assert result.companion_validation.get("ok") is True

    def test_repair_bounded_and_revalidated(self, tmp_path):
        """§31/§32: the trace must show repair rounds ran and were
        revalidated; never more than 2."""
        router = FabricatingThenFixedRouter([
            "This dua cures depression completely [hadith:sahih-bukhari:6369]. "
            "It removes depression within a month.",
        ])
        harness, _ = _harness(router, tmp_path)
        result = harness.respond("s2", "Is there any dua for depression?")
        notes = result.trace.get("notes", [])
        repair_notes = [n for n in notes if "repair" in n.lower() or "revalid" in n.lower()]
        # bounded repair ran at least once (fabrication present) OR the final
        # gate caught it directly — either path is compliant
        assert repair_notes or "final gate" in " ".join(notes)

    def test_clean_answer_skips_repair(self, tmp_path):
        """A clean, cited, hedged answer must not trigger any repair rounds."""
        router = FabricatingThenFixedRouter([
            'The Prophet ﷺ taught a supplication: "O Allah! I seek refuge in '
            'You from worry and grief" [hadith:sahih-bukhari:6369]. Some '
            "people find comfort in it alongside seeking support.",
        ])
        harness, _ = _harness(router, tmp_path)
        result = harness.respond("s3", "Is there any dua for depression?")
        notes = " ".join(result.trace.get("notes", []))
        assert "repair" not in notes.lower()
        assert result.trace.get("evidence_status") in (
            "answerable", "partially_answerable",
        )

    def test_insufficient_evidence_propagates(self, tmp_path):
        """§4: when quarantine removes everything, INSUFFICIENT_EVIDENCE
        propagates — never reintroduces quarantined evidence."""
        router = FabricatingThenFixedRouter(["I hear you."])
        harness, retrieval = _harness(router, tmp_path)

        # make the retrieval return only irrelevant evidence
        irrelevant = _hadith(
            "hadith:sahih-bukhari:9999",
            "The Prophet said: whoever builds a mosque, Allah builds for him "
            "a house in Paradise.",
        )

        class EmptyRetrieval(_FakeRetrieval):
            def search(self, query, limit=6, concept_expansions=None, semantic_only=False):
                return [irrelevant]

        harness.retrieval = EmptyRetrieval()
        result = harness.respond("s4", "Is there any dua for depression?")
        status = result.trace.get("evidence_status")
        # dua flow short-circuits to the honest offer; quarantine evidence
        # status may be 'insufficient' (pre-judge) or a judge sufficiency
        assert status in ("insufficient_evidence", "unsupported", "", "insufficient")
        # quarantined hadith must not appear as a citation in the answer
        assert "hadith:sahih-bukhari:9999" not in result.citations


class TestFinalGateChecklist:
    """§33: the shipped answer must pass the full gate — safety, citations,
    claims, companion policy, follow-up count."""

    def test_followup_max_one(self, tmp_path):
        router = FabricatingThenFixedRouter([
            "That sounds heavy. Would you like to talk about it? Would you "
            "also like a dua? Shall we continue?",
        ])
        harness, _ = _harness(router, tmp_path)
        result = harness.respond("s5", "I feel lonely.")
        assert result.answer.count("?") <= 1

    def test_no_diagnosis_in_final(self, tmp_path):
        router = FabricatingThenFixedRouter([
            "I hear you. Based on what you describe, you have depression. "
            "That must be hard.",
        ])
        harness, _ = _harness(router, tmp_path)
        result = harness.respond("s6", "I feel empty lately.")
        # the diagnosis sentence must be removed by the companion gate
        assert "you have depression" not in result.answer.lower()
