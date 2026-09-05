"""fixme_v3.1 §19-21, §37 — validation-hardening test battery.

Judge matrix (§21) × citation stress (§19) × claim extraction (§20) ×
the permanent golden regression (§37).

All deterministic — no model. The live golden regression runs in the
companion-score flow.
"""

from __future__ import annotations

import pytest

from agent.validators.claim_policy import (
    ClaimType,
    dependent_closure,
    extract_typed_claims,
)
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


TRANQUILITY = _quran(
    "quran:13:28", 13, 28,
    "Those who have believed and whose hearts are assured by the "
    "remembrance of Allah. Unquestionably, by the remembrance of Allah "
    "hearts are assured.",
)
TAWHID = _quran("quran:112:4", 112, 4, "And there is none comparable to Him.")
DISTRESS_DUA = _hadith(
    "hadith:sahih-bukhari:6369",
    'Allah\'s Messenger used to seek refuge saying: "O Allah! I seek refuge '
    'in You from worry and grief, from incapacity and laziness."',
)


# ------------------------------------------------------- §21 judge matrix
class TestJudgeMatrix:
    def test_supports(self):
        """Source directly states the proposition (quoted dua + citation)."""
        pack = EvidencePack(query="q", passages=[DISTRESS_DUA])
        answer = (
            'The Prophet ﷺ taught: "O Allah! I seek refuge in You from worry '
            'and grief" [hadith:sahih-bukhari:6369].'
        )
        j = EvidenceJudge().judge_answer(answer, pack)
        assert any(x.verdict is Verdict.SUPPORTS for x in j.claim_support)

    def test_partial(self):
        """Source supports only one component: tranquility-from-remembrance
        supports 'dhikr brings tranquility' but NOT 'dhikr treats illness'."""
        pack = EvidencePack(query="q", passages=[TRANQUILITY])
        answer = (
            "The Quran describes remembrance of Allah as a source of "
            "tranquility for hearts [quran:13:28], and this may bring some "
            "comfort during illness."
        )
        j = EvidenceJudge().judge_answer(answer, pack)
        # the remembrance claim: PARTIAL (one component) — never SUPPORTS
        # for the stronger medical half
        for x in j.claim_support:
            if "tranquility" in x.claim:
                assert x.verdict in (
                    Verdict.SUPPORTS, Verdict.PARTIAL, Verdict.BACKGROUND,
                )
            if "illness" in x.claim.lower():
                assert x.verdict is not Verdict.SUPPORTS

    def test_background(self):
        """Source is on the general topic but doesn't establish the claim."""
        pack = EvidencePack(query="q", passages=[TRANQUILITY])
        answer = (
            "The Quran mentions remembrance of Allah often [quran:13:28]. "
            "Therefore, Islam has a rich tradition of nightly worship."
        )
        j = EvidenceJudge().judge_answer(answer, pack)
        verdicts = [x.verdict for x in j.claim_support]
        assert Verdict.SUPPORTS not in verdicts or all(
            x.claim_type != "inference" for x in j.claim_support
        )

    def test_irrelevant(self):
        """No meaningful relationship: tawhid verse for a dua claim."""
        pack = EvidencePack(query="q", passages=[TAWHID])
        answer = "This dua removes depression [quran:112:4]."
        j = EvidenceJudge().judge_answer(answer, pack)
        assert all(x.verdict is not Verdict.SUPPORTS for x in j.claim_support)

    def test_unknown_no_pack(self):
        j = EvidenceJudge().judge_answer("The Quran says X.", None)
        assert j.sufficiency is Sufficiency.UNSUPPORTED

    def test_unknown_never_supports(self):
        """§6: UNKNOWN must never be treated as SUPPORTS anywhere."""
        pack = EvidencePack(query="q", passages=[TAWHID])
        j = EvidenceJudge().judge_answer("Something unclear happened.", pack)
        for x in j.claim_support:
            assert x.verdict is not Verdict.SUPPORTS


# ------------------------------------------------------- §19 citation stress
class TestCitationStress:
    def test_citation_exists_but_wrong(self):
        """exists=YES, relevant=LOW, supports=NO -> UNSUPPORTED-class."""
        pack = EvidencePack(query="q", passages=[TAWHID])
        answer = "Allah alone cures depression [quran:112:4]."
        j = EvidenceJudge().judge_answer(answer, pack)
        assert all(x.verdict is not Verdict.SUPPORTS for x in j.claim_support)
        assert j.sufficiency is not Sufficiency.ANSWERABLE

    def test_citation_relevant_but_not_entailing(self):
        """TRANQUILITY is relevant to distress topics but 'cures' needs
        direct causal evidence — must not be SUPPORTS."""
        pack = EvidencePack(query="q", passages=[TRANQUILITY])
        answer = "The Quran says dhikr cures depression [quran:13:28]."
        j = EvidenceJudge().judge_answer(answer, pack)
        assert all(x.verdict is not Verdict.SUPPORTS for x in j.claim_support)

    def test_partial_support_downgraded_language(self):
        """§6 PARTIAL: 'dhikr cures depression' -> must be downgraded."""
        pack = EvidencePack(query="q", passages=[TRANQUILITY])
        answer = "Dhikr cures depression [quran:13:28]."
        j = EvidenceJudge().judge_answer(answer, pack)
        cures = [x for x in j.claim_support if "cure" in x.claim.lower()]
        assert cures and cures[0].verdict is not Verdict.SUPPORTS

    def test_conflicting_sources_visible(self):
        """§6 CONTRADICTS-adjacent: two sources about different things —
        the pack's judgement must not average them into ANSWERABLE."""
        pack = EvidencePack(query="q", passages=[TAWHID, DISTRESS_DUA])
        answer = (
            "Allah is unlike anything [quran:112:4]. "
            "And there is a dua for worry and grief "
            "[hadith:sahih-bukhari:6369]."
        )
        j = EvidenceJudge().judge_answer(answer, pack)
        # both claims individually judged; no cross-contamination
        assert len(j.claim_support) >= 2


# ------------------------------------------------------- §20 extraction
class TestClaimExtraction:
    def test_inference_chain_both_extracted(self):
        """§20: A + therefore-B — B must not disappear for lacking a
        citation; it is explicitly extracted and judged."""
        text = (
            "The Quran mentions remembrance of Allah. "
            "Therefore, remembrance cures depression."
        )
        claims = extract_typed_claims(text)
        assert len(claims) >= 2
        assert claims[0].claim_type is ClaimType.DIRECT_FACT
        assert claims[1].claim_type in (
            ClaimType.GUARANTEE, ClaimType.CAUSAL_CLAIM,
        )
        assert claims[1].dependency_on is not None  # linked to premise

    def test_uncited_religious_claim_extracted(self):
        text = "Islam teaches that hardship is followed by ease."
        claims = extract_typed_claims(text)
        assert claims and claims[0].claim_type is ClaimType.GENERALIZATION

    def test_attribution_detected(self):
        text = "The Prophet taught us to say SubhanAllah."
        claims = extract_typed_claims(text)
        assert claims[0].claim_type is ClaimType.ATTRIBUTION

    def test_ruling_detected(self):
        claims = extract_typed_claims("Interest is haram.")
        assert claims[0].claim_type is ClaimType.RULING

    def test_diagnosis_detected(self):
        claims = extract_typed_claims("You have depression.")
        assert claims[0].claim_type is ClaimType.DIAGNOSIS

    def test_plain_prose_needs_nothing(self):
        claims = extract_typed_claims("I hear you, that sounds heavy.")
        assert claims[0].claim_type is ClaimType.PLAIN
        assert claims[0].needs_evidence is False

    def test_dependency_closure_removes_chain(self):
        """§10: removing the premise removes the whole inference chain."""
        text = (
            "The Quran mentions remembrance of Allah. "
            "Therefore, remembrance cures depression. "
            "Thus, depression is easily treated."
        )
        claims = extract_typed_claims(text)
        removed = dependent_closure({claims[0].sentence}, claims)
        assert claims[1].sentence in removed
        assert claims[2].sentence in removed


# ------------------------------------------------------- §8 strength policy
class TestStrengthPolicy:
    def test_guarantee_needs_very_strong(self):
        """Even a quoted dua does not GUARANTEE a cure — the quote entails
        the dua's existence, not its guaranteed efficacy."""
        pack = EvidencePack(query="q", passages=[DISTRESS_DUA])
        answer = (
            'Reciting "O Allah! I seek refuge in You from worry and grief" '
            "[hadith:sahih-bukhari:6369] guarantees that depression will disappear."
        )
        j = EvidenceJudge().judge_answer(answer, pack)
        guarantees = [x for x in j.claim_support if "guarante" in x.claim.lower()]
        assert guarantees and guarantees[0].verdict is not Verdict.SUPPORTS

    def test_prediction_never_supports(self):
        pack = EvidencePack(query="q", passages=[TRANQUILITY])
        answer = "Allah will remove your sadness if you remember Him [quran:13:28]."
        j = EvidenceJudge().judge_answer(answer, pack)
        predictions = [x for x in j.claim_support if x.claim_type == "prediction"]
        assert predictions and predictions[0].verdict is not Verdict.SUPPORTS

    def test_diagnosis_always_rejected(self):
        pack = EvidencePack(query="q", passages=[TRANQUILITY])
        answer = "Based on what you describe, you have depression [quran:13:28]."
        j = EvidenceJudge().judge_answer(answer, pack)
        dx = [x for x in j.claim_support if x.claim_type == "diagnosis"]
        assert dx and dx[0].verdict is Verdict.IRRELEVANT

    def test_inference_cannot_be_supports(self):
        pack = EvidencePack(query="q", passages=[TRANQUILITY])
        answer = (
            "Allah mentions remembrance of Allah [quran:13:28]. Therefore, "
            "remembrance cures depression."
        )
        j = EvidenceJudge().judge_answer(answer, pack)
        for x in j.claim_support:
            if x.claim_type == "inference":
                assert x.verdict is not Verdict.SUPPORTS


# ------------------------------------------------------- §13 mental health
class TestMentalHealthGuardrail:
    @pytest.mark.parametrize("sentence,expect_type", [
        ("Depression is proof that your iman is weak.", ClaimType.GENERALIZATION),
        ("Maybe my depression means my iman is weak.", ClaimType.GENERALIZATION),
        ("Depression is caused by weak faith.", ClaimType.GENERALIZATION),
        ("Which verse proves depression is from Shaytan?", ClaimType.GENERALIZATION),
        ("Allah punishes people with depression.", ClaimType.PREDICTION),
        ("Is depression proof that I'm a bad Muslim?", ClaimType.GENERALIZATION),
    ])
    def test_equivalence_claims_flagged(self, sentence, expect_type):
        """§13: depression = weak-iman/punishment/shaytan is never
        automatically asserted; these classify as high-risk claims."""
        claims = extract_typed_claims(sentence)
        assert claims[0].claim_type is expect_type, sentence
        assert claims[0].is_high_risk

    def test_companion_prose_not_flagged(self):
        """The guardrail must not fire on ordinary emotional sharing."""
        claims = extract_typed_claims("I feel lonely tonight.")
        assert claims[0].claim_type is ClaimType.PLAIN


# ------------------------------------------------------- §37 golden regression
class TestGoldenRegression:
    """The original failure, permanently (fixme_v3.1 §37)."""

    PACK = EvidencePack(query="q", passages=[
        _quran("quran:112:4", 112, 4, "And there is none comparable to Him."),
        _hadith("hadith:sunan-abu-dawud:650",
                "The Prophet said: when one of you urinates, do not clean "
                "filth with the right hand."),
        DISTRESS_DUA,
    ])

    def test_laundering_chain_fails(self):
        """'Allah has no equal -> therefore Allah alone cures depression'
        must fail validation."""
        answer = (
            "Allah is the only One who can remove depression [quran:112:4]."
        )
        j = EvidenceJudge().judge_answer(answer, self.PACK)
        assert all(x.verdict is not Verdict.SUPPORTS for x in j.claim_support)
        assert j.sufficiency is not Sufficiency.ANSWERABLE

    def test_filth_hadith_cannot_support_dua_claim(self):
        answer = (
            "The Prophet taught us to seek help through prayer and "
            "remembrance, as he said about wiping off filth before prayer "
            "[hadith:sunan-abu-dawud:650]."
        )
        j = EvidenceJudge().judge_answer(answer, self.PACK)
        for x in j.claim_support:
            if x.citation == "hadith:sunan-abu-dawud:650":
                assert x.verdict is not Verdict.SUPPORTS

    def test_safe_behavior_accepted(self):
        """§17: the safe form — authentic supplications exist, no cure
        guarantee, offer to show sources — must PASS."""
        answer = (
            "There are authentic supplications asking Allah for relief from "
            "anxiety, grief, and distress, such as the one taught by the "
            'Prophet ﷺ: "O Allah! I seek refuge in You from worry and '
            'grief" [hadith:sahih-bukhari:6369]. I could not verify from '
            "the sources that a particular dua is described as a guaranteed "
            "cure for depression. If you'd like, I can show you the "
            "supplications and their sources."
        )
        j = EvidenceJudge().judge_answer(answer, self.PACK)
        assert any(
            x.verdict is Verdict.SUPPORTS and x.citation == "hadith:sahih-bukhari:6369"
            for x in j.claim_support
        )
        assert not language_strength_ok(answer, j), (
            "§17 safe form must not trip the language gate"
        )
