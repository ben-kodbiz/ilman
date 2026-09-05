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

from agent.validators.claims import STRONG_CONNECTIVE_RE, Claim, extract_claims

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

    def to_dict(self) -> dict:
        return {
            "claim": self.claim[:120], "verdict": self.verdict.value,
            "citation": self.citation, "support": round(self.support_score, 2),
            "reason": self.reason,
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
    if inner.startswith(("quran:", "hadith:", "tafsir:", "tafsir-en:")):
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
        """Judge every claim in the answer against the pack (§4 pipeline)."""
        claims = extract_claims(answer)
        if not pack or not pack.passages:
            judgements = [
                ClaimJudgement(
                    claim=c.sentence, verdict=Verdict.UNKNOWN, citation=None,
                    reason="no evidence pack",
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
        support_scores: list[float] = []

        for claim in claims:
            verdict = Verdict.UNKNOWN
            best_score = 0.0
            reason = ""
            citation = None
            if claim.citations:
                # judge against the cited passage(s) specifically
                for marker in claim.citations:
                    cid = _normalize_citation(marker)
                    passage = by_citation.get(cid)
                    if passage is None:
                        judgements.append(ClaimJudgement(
                            claim=claim.sentence, verdict=Verdict.IRRELEVANT,
                            citation=cid, reason="citation not in evidence pack",
                        ))
                        continue
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
                # uncited religious claim: judge against the WHOLE pack;
                # if nothing entails it, it is unsupported
                for passage in pack.passages:
                    v, s, r = self._judge_one(claim, passage, pack)
                    if s > best_score:
                        verdict, best_score, reason = v, s, r
                citation = None
            judgements.append(ClaimJudgement(
                claim=claim.sentence, verdict=verdict, citation=citation,
                support_score=best_score, reason=reason,
            ))
            support_scores.append(best_score)

        # aggregate sufficiency (§9/§13)
        if not claims:
            return EvidenceJudgement(Sufficiency.INSUFFICIENT_EVIDENCE, 0.0, [])
        avg = sum(support_scores) / len(support_scores)
        if avg >= 0.75 and all(s >= 0.5 for s in support_scores):
            suff = Sufficiency.ANSWERABLE
        elif avg >= 0.4:
            suff = Sufficiency.PARTIALLY_ANSWERABLE
        else:
            suff = Sufficiency.INSUFFICIENT_EVIDENCE
        return EvidenceJudgement(suff, avg, judgements)

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
        score = (
            (1.0 if quote_match else 0.0) * 0.45
            + min(overlap_ratio * 1.6, 1.0) * 0.3
            + max(sem - 0.4, 0.0) * 0.5  # cosine above ~0.4 starts to matter
        )
        if not type_fit:
            score *= 0.35  # type mismatch caps support hard

        # strong language demands strong evidence (§16)
        if claim.uses_strong_language and score < 0.7:
            return Verdict.PARTIAL if score >= 0.45 else Verdict.IRRELEVANT, score, (
                "strong claim language with weak support"
            )

        if quote_match and type_fit:
            return Verdict.SUPPORTS, max(score, 0.85), "quoted passage text"
        if score >= 0.7 and type_fit:
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


# --------------------------------------------------------------- §16 gate
QUALIFIER_WORDS = re.compile(
    r"\b(may|might|can\s+be|often|some|many|one\s+way|part\s+of|alongside|"
    r"not\s+a\s+guarantee|helps\s+many|find\s+comfort)\b",
    re.IGNORECASE,
)


def language_strength_ok(answer: str, judgement: EvidenceJudgement) -> list[str]:
    """§16: evidence strength -> allowed language strength. Returns the list
    of violations (strong language on non-SUPPORTS claims)."""
    violations: list[str] = []
    for j in judgement.claim_support:
        if j.verdict is Verdict.SUPPORTS:
            continue
        if STRONG_CONNECTIVE_RE.search(j.claim):
            # strong connective — allowed only with a qualifying hedge
            # nearby in the same claim sentence
            if not QUALIFIER_WORDS.search(j.claim):
                violations.append(f"strong language on {j.verdict.value} claim: {j.claim[:90]}")
    return violations
