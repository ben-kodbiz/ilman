"""Evidence Judge: claim → evidence entailment (fixme_v3 §4-5, §9-13).

The architectural principle (§10):

    VALID_CITATION != SUPPORTED_CLAIM
    SOURCE_EXISTS   != SOURCE_RELEVANT
    SOURCE_RELEVANT != SOURCE_ENTAILS_CLAIM

Retrieval tells Ilman what MIGHT be relevant; the judge decides what Ilman is
ALLOWED to claim. Multi-signal scoring (never a single embedding score):

    lexical relevance  — content-stem overlap between claim and passage
    semantic relevance — cosine similarity between claim and passage (vector)
    claim type fit      — does the passage's TYPE match what the claim asserts?
                         (a dua claim needs dua text; a tawhid verse cannot
                         entail a depression-cure claim)
    entailment shape    — quoted spans entail verbatim; paraphrase needs
                         overlap; strong language needs STRONG support

Verdicts (§11): SUPPORTS / PARTIAL / BACKGROUND / IRRELEVANT / CONTRADICTS /
UNKNOWN. Only SUPPORTS may carry authoritative claims; PARTIAL forces
conservative language; IRRELEVANT is quarantined (§12).

Evidence sufficiency (§9/§13): ANSWERABLE / PARTIALLY_ANSWERABLE /
INSUFFICIENT_EVIDENCE / UNSUPPORTED.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from agent.validators.claims import STRONG_CONNECTIVE_RE, Claim

_SUPPORT_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "with", "for", "as", "by", "at", "from", "not", "no",
    "but", "if", "then", "than", "so", "what", "which", "who", "when",
    "where", "why", "how", "all", "any", "some", "there", "here", "his",
    "her", "its", "their", "our", "your", "my", "me", "us", "them", "also",
    "one", "only", "every",
}


def _content_stems(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z']{4,}", (text or "").lower())
    return {w[:5] for w in words if w not in _SUPPORT_STOP}


class Verdict(StrEnum):
    SUPPORTS = "supports"
    PARTIAL = "partial"
    BACKGROUND = "background"
    IRRELEVANT = "irrelevant"
    CONTRADICTS = "contradicts"
    UNKNOWN = "unknown"


class Sufficiency(StrEnum):
    ANSWERABLE = "answerable"
    PARTIALLY_ANSWERABLE = "partially_answerable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED = "unsupported"


@dataclass
class ClaimJudgement:
    claim: str
    verdict: Verdict
    citation: str | None
    support_score: float = 0.0
    reason: str = ""
    claim_type: str = ""           # fixme_v3.1 §7
    requirement: str = ""          # §8 evidence requirement
    citation_exists: bool = False  # §5.1
    citation_relevant: bool = False  # §5.2 (topic-level relation)
    dependency_on: str | None = None  # §9 inference premise
    is_high_risk: bool = False       # §28 critical claim

    def to_dict(self) -> dict:
        return {
            "claim": self.claim[:120], "verdict": self.verdict.value,
            "citation": self.citation, "support": round(self.support_score, 2),
            "reason": self.reason, "claim_type": self.claim_type,
            "requirement": self.requirement,
            "citation_exists": self.citation_exists,
            "citation_relevant": self.citation_relevant,
            "is_high_risk": self.is_high_risk,
        }


@dataclass
class EvidenceJudgement:
    sufficiency: Sufficiency
    evidence_sufficiency: float  # 0..1 aggregate (§9)
    claim_support: list[ClaimJudgement] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(
            j.verdict in (Verdict.IRRELEVANT, Verdict.UNKNOWN)
            for j in self.claim_support
        )

    def to_dict(self) -> dict:
        return {
            "sufficiency": self.sufficiency.value,
            "evidence_sufficiency": round(self.evidence_sufficiency, 2),
            "claim_support": [j.to_dict() for j in self.claim_support],
        }


# claim-type requirements: certain claims can only be supported by certain
# passage shapes (fixme_v3 §2/§7 — a tawhid verse cannot entail a dua claim)
_DUA_TEXT_RE = re.compile(
    r"\bo allah\b|\ballahum(a|ma)\b|i seek refuge|i seek protection",
    re.IGNORECASE,
)
_MEDICAL_GUARANTEE_RE = re.compile(
    r"\b(cure|cures|remove[sd]?|removing|treat[sd]?|treatment|heal[sd]?|"
    r"guarantee[sd]?|guarantees|disappear[sd]?)\b",
    re.IGNORECASE,
)
_DISTRESS_WORDS = {
    "anxie", "grief", "sorro", "worry", "sadne", "distre", "hardsh",
    "depress", "heavine", "despon", "melanc",
}


def _normalize_citation(marker: str) -> str:
    """[quran:2:255] / quran:2:255 / [sahih-bukhari:1] -> canonical id."""
    inner = marker.strip("[]()").lower()
    if inner.startswith(("quran:", "hadith:", "tafsir:", "tafsir-en:", "webfatwa:")):
        return inner
    parts = inner.split(":")
    if len(parts) == 2 and parts[1].isdigit():
        return f"hadith:{parts[0]}:{parts[1]}"
    return inner


class EvidenceJudge:
    def __init__(self, embed=None):
        """embed: optional callable(text) -> vector, for semantic signal.
        Falls back to lexical-only when absent (tests, offline)."""
        self.embed = embed

    # ------------------------------------------------------------- judging
    def judge_answer(self, answer: str, pack, topic: str | None = None,
                     requested_object: str | None = None) -> EvidenceJudgement:
        """Judge every claim in the answer against the pack (§4 pipeline).

        fixme_v3.1 §7/§11/§12: claims are TYPED (claim_policy), each judged
        independently; sufficiency is derived from claim-level results, not
        an average — one unsupported high-risk claim forces repair (§28) and
        caps the aggregate at PARTIALLY_ANSWERABLE at best.
        """
        from agent.validators.claim_policy import (
            ClaimType,
            TypedClaim,
            extract_typed_claims,
        )

        typed: list[TypedClaim] = extract_typed_claims(answer)
        claims = [c for c in typed if c.needs_evidence or c.has_citation]
        if not pack or not pack.passages:
            judgements = [
                ClaimJudgement(
                    claim=c.sentence, verdict=Verdict.UNKNOWN, citation=None,
                    reason="no evidence pack", claim_type=c.claim_type.value,
                    requirement=c.requirement.value, dependency_on=c.dependency_on,
                    is_high_risk=c.is_high_risk,
                )
                for c in claims
            ]
            suff = (
                Sufficiency.UNSUPPORTED if claims
                else Sufficiency.INSUFFICIENT_EVIDENCE
            )
            return EvidenceJudgement(suff, 0.0, judgements)

        by_citation = {p.citation_id: p for p in pack.passages}
        judgements: list[ClaimJudgement] = []

        for claim in claims:
            # §7 claim types that are NEVER allowed as assertions
            if claim.claim_type is ClaimType.DIAGNOSIS:
                judgements.append(ClaimJudgement(
                    claim=claim.sentence, verdict=Verdict.IRRELEVANT,
                    citation=None, reason="diagnosis is never allowed",
                    claim_type=claim.claim_type.value,
                    requirement=claim.requirement.value,
                    is_high_risk=True,
                ))
                continue

            verdict = Verdict.UNKNOWN
            best_score = 0.0
            reason = ""
            citation = None
            cited_exists = False
            if claim.citations:
                for marker in claim.citations:
                    cid = _normalize_citation(marker)
                    passage = by_citation.get(cid)
                    if passage is None:
                        judgements.append(ClaimJudgement(
                            claim=claim.sentence, verdict=Verdict.IRRELEVANT,
                            citation=cid, reason="citation not in evidence pack",
                            claim_type=claim.claim_type.value,
                            requirement=claim.requirement.value,
                            is_high_risk=claim.is_high_risk,
                        ))
                        continue
                    cited_exists = True
                    v, s, r = self._judge_one(claim, passage, pack)
                    citation = citation or cid
                    if s > best_score:
                        verdict, best_score, reason = v, s, r
                if citation is None and not any(
                    j.verdict is Verdict.IRRELEVANT
                    for j in judgements[-len(claim.citations):]
                ):
                    continue
            else:
                # §9: an inference-boundary claim is judged on its OWN; the
                # fact that its premise was supported never carries over.
                for passage in pack.passages:
                    v, s, r = self._judge_one(claim, passage, pack)
                    if s > best_score:
                        verdict, best_score, reason = v, s, r

            # §5.2 citation relevance (topic-level): does the claim's topic
            # share content with the cited/best passage at all?
            citation_relevant = best_score >= 0.25

            # §8 claim-strength enforcement
            verdict, reason = self._enforce_strength_policy(
                claim, verdict, best_score, reason
            )
            judgements.append(ClaimJudgement(
                claim=claim.sentence, verdict=verdict, citation=citation,
                support_score=best_score, reason=reason,
                claim_type=claim.claim_type.value,
                requirement=claim.requirement.value,
                citation_exists=cited_exists,
                citation_relevant=citation_relevant,
                dependency_on=claim.dependency_on,
                is_high_risk=claim.is_high_risk,
            ))

        # §11/§12 sufficiency from CLAIM-LEVEL results, not averages
        if not claims:
            return EvidenceJudgement(Sufficiency.INSUFFICIENT_EVIDENCE, 0.0, [])
        supported = [j for j in judgements if j.verdict is Verdict.SUPPORTS]
        partial = [j for j in judgements if j.verdict is Verdict.PARTIAL]
        bad = [
            j for j in judgements
            if j.verdict in (Verdict.IRRELEVANT, Verdict.UNKNOWN)
        ]
        high_risk_bad = [j for j in bad if j.is_high_risk]
        if bad or high_risk_bad:
            # any unsupported claim — high-risk ones especially — forces
            # at most PARTIALLY_ANSWERABLE (repair path decides removal)
            suff = (
                Sufficiency.PARTIALLY_ANSWERABLE
                if (supported or partial) and not high_risk_bad
                else Sufficiency.INSUFFICIENT_EVIDENCE
            )
        elif partial and supported:
            suff = Sufficiency.PARTIALLY_ANSWERABLE
        elif supported:
            suff = Sufficiency.ANSWERABLE
        elif partial:
            suff = Sufficiency.PARTIALLY_ANSWERABLE
        else:
            suff = Sufficiency.INSUFFICIENT_EVIDENCE
        avg = (
            sum(j.support_score for j in judgements) / len(judgements)
            if judgements else 0.0
        )
        return EvidenceJudgement(suff, avg, judgements)

    def _enforce_strength_policy(self, claim, verdict: Verdict,
                                 score: float, reason: str) -> tuple[Verdict, str]:
        """fixme_v3.1 §8: claim type -> minimum evidence bar. Nothing can be
        SUPPORTS below its type's bar; GUARANTEE/RULING need the strongest
        support, INFERENCE must be labeled as inference (never asserted)."""
        from agent.validators.claim_policy import (
            ClaimType,
        )

        if verdict is not Verdict.SUPPORTS:
            return verdict, reason
        ctype: ClaimType = claim.claim_type
        if ctype is ClaimType.GUARANTEE:
            # §8: guarantee needs VERY_STRONG (near-verbatim) support
            if score < 0.8:
                return Verdict.PARTIAL, (
                    f"guarantee claim needs very strong support (score {score:.2f})"
                )
        elif ctype in (ClaimType.RULING, ClaimType.ATTRIBUTION):
            if score < 0.7:
                return Verdict.PARTIAL, (
                    f"{ctype.value} needs an exact authoritative source "
                    f"(score {score:.2f})"
                )
        elif ctype is ClaimType.CAUSAL_CLAIM:
            if score < 0.75:
                return Verdict.PARTIAL, (
                    f"causal claim needs direct evidence (score {score:.2f})"
                )
        elif ctype is ClaimType.PREDICTION:
            return Verdict.PARTIAL, (
                "predictions may never be religious certainty"
            )
        elif ctype is ClaimType.INFERENCE:
            # §8: inference must be LABELLED as inference; an asserted
            # 'therefore X' that the sources don't state cannot be SUPPORTS
            return Verdict.PARTIAL, (
                "inference: allowed only as explicitly labeled reflection"
            )
        return verdict, reason

    # --------------------------------------------------------- one claim×passage
    def _judge_one(self, claim: Claim, passage, pack) -> tuple[Verdict, float, str]:
        text = (passage.translation or passage.arabic or "")
        claim_text = claim.sentence

        # 1) verbatim quote entailment: strongest possible support
        re.sub(r"[^a-z\s]", "", claim_text.lower())
        norm_passage = re.sub(r"[^a-z\s]", "", text.lower())
        quoted = re.findall(r"[\"'“”\*]{1,2}([^\"'“”\*]{15,})[\"'“”\*]{1,2}", claim_text)
        quote_match = any(
            re.sub(r"[^a-z\s]", "", q.lower()).strip() in norm_passage
            for q in quoted if len(q.strip()) > 12
        )

        # 2) content-stem overlap (lexical relevance)
        claim_stems = _content_stems(claim_text)
        passage_stems = _content_stems(text)
        overlap = len(claim_stems & passage_stems)
        overlap_ratio = overlap / max(len(claim_stems), 1)

        # 3) semantic relevance (optional vector signal)
        sem = 0.0
        if self.embed is not None and len(claim_text) > 20:
            try:
                import numpy as np

                cv = np.array(self.embed(claim_text), dtype=np.float32)
                pv = np.array(self.embed(text[:2000]), dtype=np.float32)
                denom = (np.linalg.norm(cv) * np.linalg.norm(pv)) or 1.0
                sem = float(cv @ pv / denom)
            except Exception:
                sem = 0.0

        # 4) claim-type fit (§2/§7): type-mismatched passages cannot SUPPORT
        type_fit = self._type_fit(claim_text, text)

        # aggregate score
        # §22 (v3.1): semantic similarity is ONE signal, never entailment
        # alone — but a paraphrase with zero lexical overlap to its source
        # ("no equal" vs "none comparable") IS legitimately semantic. For
        # SHORT claims (paraphrase-scale) the cosine above 0.55 is strong
        # evidence of restatement; long claims still need lexical corroboration.
        # §11 authority axis (enhance_v1): entailment AND authority are
        # separate checks. A tafsir chunk quoting a verse cannot SUPPORT a
        # claim asserted as "The Quran says/commands" — wrong source class
        # caps the verdict at PARTIAL with an AUTHORITY_FAIL reason.
        from agent.evidence.authority import AuthorityResult, check_authority

        claim_type_name = getattr(claim, "claim_type", None)
        claim_type_name = claim_type_name.value if hasattr(claim_type_name, "value") else str(claim_type_name or "")
        # authority lookup keys on the CITATION (always prefixed, e.g.
        # hadith:sahih-bukhari:6369) — source_id is the bare collection
        # name for hadiths and cannot be prefix-matched
        citation = getattr(passage, "citation_id", "") or getattr(passage, "source_id", "")
        authority = check_authority(claim_type_name, citation)

        sem_component = max(sem - 0.4, 0.0) * 0.5
        if len(claim_text.split()) <= 14 and sem >= 0.55:
            sem_component = max(sem_component, (sem - 0.45) * 1.6)
            # whole-passage paraphrase (§22-safe): a SHORT claim restating a
            # SHORT passage needs BOTH near-similarity AND the passage's own
            # key stems present in the claim — 'no equal'/'none comparable'
            # share the stem family 'none/equal-compa'... which fails, so
            # ALSO accept the semantic route ONLY when the passage is fully
            # covered: every key stem of the PASSAGE appears in the CLAIM.
            # 'depression weak faith' vs 'hearts remembrance' share nothing,
            # so topical false-matches cannot ride this path.
            if len(text.split()) <= 16 and sem >= 0.60:
                passage_key = [t for t in passage_stems if len(t) >= 5]
                if passage_key:
                    covered = sum(1 for t in passage_key if t in claim_stems) / len(passage_key)
                    if covered >= 0.5:  # claim covers >= half the passage's stems
                        if authority is AuthorityResult.AUTHORITY_FAIL:
                            return Verdict.PARTIAL, 0.45, (
                                "AUTHORITY_FAIL: source class not qualified "
                                "for this claim type"
                            )
                        prelim = (
                            (1.0 if quote_match else 0.0) * 0.45
                            + min(overlap_ratio * 1.6, 1.0) * 0.3 + sem_component
                        )
                        return Verdict.SUPPORTS, max(prelim, 0.72), (
                            "short-passage paraphrase (semantic + stem coverage)"
                        )
        score = (
            (1.0 if quote_match else 0.0) * 0.45
            + min(overlap_ratio * 1.6, 1.0) * 0.3
            + sem_component
        )
        if not type_fit:
            score *= 0.35  # type mismatch caps support hard

        # 3b) enumerated-content support: a passage that LISTS items
        # (e.g. the pillars hadith listing all five) supports a claim about
        # ONE of its list items — the item's key terms are all present in
        # the passage even though stem-overlap ratio is low. Key-term
        # coverage is the signal, not ratio.
        claim_key_terms = [t for t in claim_stems if len(t) >= 5]
        key_coverage = (
            sum(1 for t in claim_key_terms if t in passage_stems)
            / max(len(claim_key_terms), 1)
        )
        if claim_key_terms and key_coverage >= 0.6:
            # enumerated-content support: a passage LISTS items and the claim
            # restates those items — key coverage of 2/3 or more is strong
            # evidence the claim derives from this passage (0.667*0.3+0.4=0.6;
            # 0.8 coverage -> 0.64; attributions need the strong-language bar
            # cleared too). A 0.8+ claim-side key coverage on a claim whose
            # citation points here is effectively a paraphrased enumeration.
            score = max(score, 0.4 + 0.3 * key_coverage)
            if key_coverage >= 0.66 and overlap_ratio >= 0.5:
                score = max(score, 0.72)

        # strong language demands strong evidence (§16 v3 / §8 v3.1):
        # high-risk claim types use the STRONG_CONNECTIVE vocabulary as the
        # proxy for assertive phrasing
        uses_strong_language = getattr(claim, "uses_strong_language", None)
        if uses_strong_language is None:
            from agent.validators.claims import STRONG_CONNECTIVE_RE

            uses_strong_language = bool(STRONG_CONNECTIVE_RE.search(claim_text))
        if uses_strong_language and score < 0.7:
            return Verdict.PARTIAL if score >= 0.45 else Verdict.IRRELEVANT, score, (
                "strong claim language with weak support"
            )

        if quote_match and type_fit:
            if authority is AuthorityResult.AUTHORITY_FAIL:
                return Verdict.PARTIAL, min(score, 0.5), (
                    "AUTHORITY_FAIL: source class not qualified for this claim type"
                )
            return Verdict.SUPPORTS, max(score, 0.85), "quoted passage text"
        if score >= 0.7 and type_fit:
            if authority is AuthorityResult.AUTHORITY_FAIL:
                return Verdict.PARTIAL, min(score, 0.5), (
                    "AUTHORITY_FAIL: source class not qualified for this claim type"
                )
            return Verdict.SUPPORTS, score, "strong content overlap"
        if score >= 0.4:
            return Verdict.PARTIAL, score, "partial relevance"
        if overlap == 0 and sem < 0.5:
            return Verdict.IRRELEVANT, score, "no content overlap with claim"
        return Verdict.BACKGROUND, score, "topical but not claim-entailing"

    @staticmethod
    def _type_fit(claim_text: str, passage_text: str) -> bool:
        """A dua/guarantee claim needs dua text or distress words in the
        passage; a tawhid-only passage can never support a dua claim."""
        claim_lower = claim_text.lower()
        passage_lower = (passage_text or "").lower()
        wants_dua = (
            _DUA_TEXT_RE.search(claim_lower) is not None
            or "dua" in claim_lower or "supplication" in claim_lower
        )
        claims_distress_relief = bool(
            _MEDICAL_GUARANTEE_RE.search(claim_lower)
            or any(w in claim_lower for w in ("depress", "anxie", "grief", "worry", "distre"))
        )
        if not (wants_dua or claims_distress_relief):
            return True  # no special type requirement
        passage_is_dua = _DUA_TEXT_RE.search(passage_lower) is not None
        passage_about_distress = any(
            w in passage_lower for w in _DISTRESS_WORDS
        )
        # depression-cure/guarantee claims need distress-context dua texts
        if claims_distress_relief and _MEDICAL_GUARANTEE_RE.search(claim_lower):
            return passage_is_dua and (
                passage_about_distress or "seek refuge" in passage_lower
            )
        if wants_dua:
            return passage_is_dua
        return passage_about_distress


# --------------------------------------------------------------- §24 gate
# v3.1 §24: verdict -> allowed phrasing. Strong assertive language is gated
# by verdict; each verdict tier has its allowed sentence openers.
VERDICIDAL_OPENERS = {
    # SUPPORTS -> direct attribution allowed
    "supports": [],
    # PARTIAL -> qualified language only
    "partial": ["may indicate", "may suggest", "can provide context",
                "some people find", "one understanding is", "this may reflect"],
    # BACKGROUND -> context framing only
    "background": ["provides context", "for background"],
    # UNKNOWN -> uncertainty only
    "unknown": ["I could not verify"],
}

QUALIFIER_WORDS = re.compile(
    r"\b(may|might|can\s+be|often|some|many|one\s+way|part\s+of|alongside|"
    r"not\s+a\s+guarantee|helps\s+many|find\s+comfort|may\s+indicate|"
    r"may\s+suggest|some\s+people|one\s+understanding|provides\s+context|"
    r"could\s+not\s+verify)\b",
    re.IGNORECASE,
)

# assertive language forms that need the highest verdict to be allowed (§24)
ASSERTIVE_FORMS = re.compile(
    r"\b(the\s+quran\s+(says|proves|states)|islam\s+teaches|"
    r"allah\s+(will|alone|says|promises)|the\s+prophet\s+(taught|said)|"
    r"this\s+(cures|guarantees|proves)|this\s+is\s+definitely)\b",
    re.IGNORECASE,
)


def language_strength_ok(answer: str, judgement: EvidenceJudgement) -> list[str]:
    """v3.1 §24: evidence strength -> allowed language strength.

    Strong assertive religious phrasing is only allowed on SUPPORTS claims;
    PARTIAL/BACKGROUND/UNKNOWN claims must carry qualifier hedges. Returns
    the list of violations."""
    violations: list[str] = []
    for j in judgement.claim_support:
        if j.verdict is Verdict.SUPPORTS:
            continue
        strong = STRONG_CONNECTIVE_RE.search(j.claim) or ASSERTIVE_FORMS.search(j.claim)
        if strong:
            if not QUALIFIER_WORDS.search(j.claim):
                violations.append(
                    f"strong language on {j.verdict.value} claim: {j.claim[:90]}"
                )
    return violations
