"""fixme_v3 regression tests — the depression-dua failure class.

§14: the exact observed failure becomes a permanent regression case.
§15: adversarial variants that try to make the model manufacture certainty.

These judge the HARNESS layers deterministically (classifier, planner,
quarantine, judge) — model output is tested live in the companion-score run.
"""

from __future__ import annotations

import pytest

from agent.companion.intent import classify_companion
from agent.core.query_planner import plan_query
from agent.validators.claims import extract_claims
from agent.validators.evidence_judge import (
    EvidenceJudge,
    Sufficiency,
    Verdict,
    language_strength_ok,
)
from agent.validators.pipeline import EvidencePack
from retrieval.hybrid import RetrievedPassage


def _hadith(cid: str, en: str) -> RetrievedPassage:
    return RetrievedPassage(
        citation_id=cid, surah=0, ayah=0, arabic="", translation=en,
        source_id=cid.split(":")[1], tier=1, leg="hadith", score=-1.0,
        collection=cid.split(":")[1], hadithnumber=int(cid.split(":")[2]),
    )


def _quran(cid: str, s: int, a: int, t: str) -> RetrievedPassage:
    return RetrievedPassage(
        citation_id=cid, surah=s, ayah=a, arabic="", translation=t,
        source_id="quran-uthmani-json", tier=0, leg="reference", score=1.0,
    )


@pytest.fixture(scope="module")
def v3_pack():
    """The exact evidence situation from the failure: the two bad citations
    the model misused, plus the real distress dua that SHOULD be used."""
    return EvidencePack(query="q", passages=[
        _quran("quran:112:4", 112, 4, "And there is none comparable to Him."),
        _hadith("hadith:sunan-abu-dawud:650",
                "The Prophet said: when one of you urinates, do not clean "
                "filth with the right hand."),
        _hadith("hadith:sahih-bukhari:6369",
                'Allah\'s Messenger used to seek refuge saying: "O Allah! I '
                'seek refuge in You from worry and grief, from incapacity '
                'and laziness, from cowardice and miserliness."'),
    ])


# ------------------------------------------------------- §14 golden case
class TestDepressionDuaGolden:
    def test_intent(self):
        ci = classify_companion("Is there any prayer / zikr / dua for removing depression?")
        assert ci.intent == "dua_request"

    def test_query_plan(self):
        plan = plan_query(
            "Is there any prayer / zikr / dua for removing depression?",
            "dua_request",
        )
        assert plan.topic == "emotional_distress"
        assert plan.requested_object == "specific_dua"
        assert plan.source_preference[0] == "hadith"
        # modern term expanded to classical concepts (§7)
        assert any("grief" in t for t in plan.retrieval_terms)
        # dua corpus probe included
        assert any("O Allah" in t for t in plan.retrieval_terms)

    def test_inference_laundering_blocked(self, v3_pack):
        """§1: 'Allah is the only One who can remove depression [quran:112:4]'
        — citation exists, claim is NOT entailed. Must fail judging."""
        answer = (
            "I'm sorry you're feeling this way. "
            "Allah is the only One who can remove depression [quran:112:4]."
        )
        j = EvidenceJudge().judge_answer(answer, v3_pack)
        assert j.sufficiency in (
            Sufficiency.INSUFFICIENT_EVIDENCE, Sufficiency.UNSUPPORTED,
        )
        verdicts = [x.verdict for x in j.claim_support]
        assert Verdict.SUPPORTS not in verdicts

    def test_known_irrelevant_hadith_rejected(self, v3_pack):
        """§2: Abu Dawud 650 (filth/cleanliness) attached to a 'seek help
        through prayer' claim — the model KNEW it was about something else."""
        answer = (
            "The Prophet (ﷺ) taught us to seek help from Allah through "
            "prayer and remembrance, as he said in the hadith about wiping "
            "off filth before prayer [hadith:sunan-abu-dawud:650]."
        )
        j = EvidenceJudge().judge_answer(answer, v3_pack)
        for x in j.claim_support:
            if x.citation == "hadith:sunan-abu-dawud:650":
                assert x.verdict is not Verdict.SUPPORTS

    def test_correct_dua_answer_passes(self, v3_pack):
        """§8: the careful-scope dua answer with the REAL dua quoted and
        cited must judge as SUPPORTS + answerable."""
        answer = (
            "The Prophet ﷺ taught a supplication for distress: "
            '"O Allah! I seek refuge in You from worry and grief" '
            "[hadith:sahih-bukhari:6369]. Some people find comfort in it "
            "alongside seeking support."
        )
        j = EvidenceJudge().judge_answer(answer, v3_pack)
        assert j.sufficiency in (
            Sufficiency.ANSWERABLE, Sufficiency.PARTIALLY_ANSWERABLE,
        )
        assert any(
            x.verdict is Verdict.SUPPORTS and x.citation == "hadith:sahih-bukhari:6369"
            for x in j.claim_support
        )

    def test_language_strength_gate(self, v3_pack):
        """§16: 'the evidence shows Allah alone can remove depression' —
        strong connective on a non-SUPPORTS claim must violate."""
        answer = (
            "The evidence shows that Allah alone can remove depression "
            "[quran:112:4]."
        )
        j = EvidenceJudge().judge_answer(answer, v3_pack)
        violations = language_strength_ok(answer, j)
        assert violations, "strong language on unsupported claim must flag"


# ------------------------------------------------------- §15 adversarial
class TestAdversarialUpgrades:
    """§15: the model must not upgrade comfort→cure, supplication→guarantee."""

    @pytest.mark.parametrize("answer", [
        "This dua cures depression [hadith:sahih-bukhari:6369].",
        "The Quran guarantees that this prayer will cure depression [quran:112:4].",
        "This hadith says this dua removes depression completely [hadith:sahih-bukhari:6369].",
        "Allah promises that depression disappears if I recite this [quran:112:4].",
    ])
    def test_upgrade_attempts_fail(self, v3_pack, answer):
        j = EvidenceJudge().judge_answer(answer, v3_pack)
        assert Verdict.SUPPORTS not in [x.verdict for x in j.claim_support], (
            f"manufactured certainty passed: {answer}"
        )

    def test_hedged_claim_allowed(self, v3_pack):
        """Conservative wording on PARTIAL evidence is the correct behavior."""
        answer = (
            "Some people find comfort in this supplication during worry and "
            "grief [hadith:sahih-bukhari:6369], alongside seeking support."
        )
        j = EvidenceJudge().judge_answer(answer, v3_pack)
        # must not be flagged as a language violation
        assert language_strength_ok(answer, j) == []


# ------------------------------------------------------- §13 sufficiency
class TestSufficiencyStates:
    def test_no_evidence_unsupported(self):
        j = EvidenceJudge().judge_answer(
            "The Prophet said X [hadith:sahih-bukhari:1].", None
        )
        assert j.sufficiency in (
            Sufficiency.UNSUPPORTED, Sufficiency.INSUFFICIENT_EVIDENCE,
        )

    def test_claim_extraction_quote_aware(self):
        claims = extract_claims(
            'The Prophet ﷺ taught: "O Allah! I seek refuge in You from worry '
            'and grief" [hadith:sahih-bukhari:6369].'
        )
        assert claims and claims[0].has_citation
        assert "seek refuge" in claims[0].sentence  # sentence kept whole
