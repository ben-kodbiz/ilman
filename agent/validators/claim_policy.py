"""Claim types, strength policy, and inference-dependency handling
(fixme_v3.1 §7-§10).

Claim taxonomy:

    DIRECT_FACT    "The Quran says X."
    PARAPHRASE     restatement of a source
    ATTRIBUTION    "The Prophet taught X."
    INFERENCE      "which means... / we can conclude..."
    CAUSAL_CLAIM   "X causes/cures Y."
    GENERALIZATION "Islam teaches / Muslims should..."
    GUARANTEE      "X will cure / guarantees"
    RULING         "X is haram/halal/obligatory"
    DIAGNOSIS      "you have depression"
    PREDICTION     "Allah will..."
    INTERPRETATION tafsir-flavored reading

Strength policy (§8): each type has an evidence requirement; the judge
enforces it. DIAGNOSIS and PREDICTION are never allowed as religious
certainty.

Inference boundaries (§9): connectives (therefore/thus/which means...)
split A→B chains; B is extracted as its own claim with a dependency on A
and must be independently judged — validating A does not validate B.

Dependency graph (§10): if a claim is removed, its dependents are removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum, auto

from agent.validators.claims import _sentences


class ClaimType(StrEnum):
    DIRECT_FACT = auto()
    PARAPHRASE = auto()
    ATTRIBUTION = auto()
    INFERENCE = auto()
    CAUSAL_CLAIM = auto()
    GENERALIZATION = auto()
    GUARANTEE = auto()
    RULING = auto()
    DIAGNOSIS = auto()
    PREDICTION = auto()
    INTERPRETATION = auto()
    PLAIN = auto()  # non-religious prose; needs no evidence


class EvidenceRequirement(StrEnum):
    """What a claim type needs before it may appear (§8)."""

    DIRECT_SOURCE = auto()
    STRONG_SEMANTIC = auto()
    EXACT_SOURCE = auto()
    LABEL_AS_INFERENCE = auto()
    VERY_STRONG = auto()
    AUTHORITATIVE_SOURCE = auto()
    NEVER_ALLOWED = auto()
    NOT_RELIGIOUS_CERTAINTY = auto()
    NONE = auto()


# §8 strength policy table
CLAIM_STRENGTH_POLICY: dict[ClaimType, EvidenceRequirement] = {
    ClaimType.DIRECT_FACT: EvidenceRequirement.DIRECT_SOURCE,
    ClaimType.PARAPHRASE: EvidenceRequirement.STRONG_SEMANTIC,
    ClaimType.ATTRIBUTION: EvidenceRequirement.EXACT_SOURCE,
    ClaimType.INFERENCE: EvidenceRequirement.LABEL_AS_INFERENCE,
    ClaimType.CAUSAL_CLAIM: EvidenceRequirement.DIRECT_SOURCE,
    ClaimType.GENERALIZATION: EvidenceRequirement.DIRECT_SOURCE,
    ClaimType.GUARANTEE: EvidenceRequirement.VERY_STRONG,
    ClaimType.RULING: EvidenceRequirement.AUTHORITATIVE_SOURCE,
    ClaimType.DIAGNOSIS: EvidenceRequirement.NEVER_ALLOWED,
    ClaimType.PREDICTION: EvidenceRequirement.NOT_RELIGIOUS_CERTAINTY,
    ClaimType.INTERPRETATION: EvidenceRequirement.DIRECT_SOURCE,
    ClaimType.PLAIN: EvidenceRequirement.NONE,
}


# ---------------------------------------------------------------- detection
_GUARANTEE_RE = re.compile(
    r"\b(guarantees?|guaranteed|will\s+(certainly\s+)?cure|cures?\s|heals?\s|"
    r"removes?\s+\w+\s+completely|will\s+definitely|disappears?\s+if|"
    r"prescribed\s+to\s+cure|replacing\s+therapy)\b"
    r"|\bspecifically\s+prescribed\s+for\b",
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r"\b(causes?|cures?|treats?|heals?|removes?|prevents?|leads?\s+to|"
    r"results?\s+in|because\s+of)\b",
    re.IGNORECASE,
)
_INFERENCE_RE = re.compile(
    r"\b(therefore|thus|hence|so\s+it\s+follows|which\s+means|this\s+proves|"
    r"this\s+shows|we\s+can\s+conclude|it\s+follows\s+that)\b",
    re.IGNORECASE,
)
_ATTRIBUTION_RE = re.compile(
    r"\b(the\s+)?(prophet|messenger|rasul)\b[^.\n]{0,60}"
    r"\b(said|says|taught|taught\s+us|told|advised|commanded)\b"
    r"|\b(allah)\b[^.\n]{0,40}\b(said|says|states|promises)\b",
    re.IGNORECASE,
)
_DIRECT_FACT_RE = re.compile(
    r"\b(the\s+)?(quran|qur'?an)\b[^.\n]{0,60}"
    r"\b(says|said|states|tells\s+us|mentions|describes|teaches)\b",
    re.IGNORECASE,
)
_GENERALIZATION_RE = re.compile(
    r"\b(islam\s+(teaches|says|considers|requires|encourages)|"
    r"muslims\s+(should|must|are\s+required)|in\s+islam,|"
    # v3.1 §13 mental-health guardrail: any sentence linking a modern
    # psychological term to a spiritual causation/judgment is a religious
    # generalization about mental health — highest-scrutiny claim class
    r"(depress\w+|lonely|loneliness|feeling\s+(sad|alone|anxious)|"
    r"sadness|anxiety)[^.\n]{0,60}"
    r"(weak\s+iman|weak\s+faith|lack\s+of\s+faith|punish\w*|"
    r"shaytan|satan|sign\s+of|proof\s+that|caused\s+by|"
    r"iman\s+is\s+weak|faith\s+is\s+weak)|"
    r"(weak\s+iman|weak\s+faith|lack\s+of\s+faith|"
    r"iman\s+is\s+weak|faith\s+is\s+weak)[^.\n]{0,40}depress\w*)\b",
    re.IGNORECASE,
)
_RULING_RE = re.compile(
    r"\b(is|are|it\s+is)\s+(absolutely\s+|completely\s+|categorically\s+)?"
    r"(haram|halal|forbidden|obligatory|fard|wajib|sunnah\s+to|recommended)\b",
    re.IGNORECASE,
)
_DIAGNOSIS_RE = re.compile(
    r"\byou\s+(have|are\s+suffering\s+from|are\s+clinically|seem\s+to\s+have)\b"
    r"[^.\n]{0,40}\b(depress\w*|anxiety|bipolar|disorder|mental\s+illness)\b",
    re.IGNORECASE,
)
_PREDICTION_RE = re.compile(
    r"\ballah\s+(will|would|shall|punishes|punished|hates|rewards|"
    r"promises)\b[^.\n]{0,60}"
    r"|\b(allah|god)\s+is\s+punishing\b"
    r"|\bpromises?\s+to\s+(remove|take\s+away|lift)"
    r"|\ballah\s+punishing\b|\bpunishing\s+me\b",
    re.IGNORECASE,
)
_INTERPRETATION_RE = re.compile(
    r"\b(this\s+verse\s+means|this\s+means|the\s+meaning\s+is|"
    r"scholars\s+interpret|this\s+signifies)\b",
    re.IGNORECASE,
)
_RELIGIOUS_ANY_RE = re.compile(
    r"\b(quran|qur'?an|allah|prophet|islam|muslim|hadith|sunnah|"
    r"dua|ayah|surah|haram|halal)\b",
    re.IGNORECASE,
)


_FIRST_PERSON_FEELING_RE = re.compile(
    r"^\s*(i\s+(feel|felt)|i'?m\s+feeling)\b", re.IGNORECASE
)


def classify_claim_type(sentence: str) -> ClaimType:
    """Priority: most-specific first. Diagnosis/guarantee outrank causal;
    causal outranks inference; attribution/direct-fact outrank generic.
    GENERALIZATION heads (Islam teaches/says...) keep the generalization
    type even when their object is a cure claim — 'Islam says X cures Y'
    is an unattributed generalization, not a direct guarantee."""
    is_generalization_head = bool(_GENERALIZATION_RE.search(sentence))
    if _DIAGNOSIS_RE.search(sentence):
        return ClaimType.DIAGNOSIS
    if _GUARANTEE_RE.search(sentence) and not is_generalization_head:
        return ClaimType.GUARANTEE
    if is_generalization_head:
        return ClaimType.GENERALIZATION
    if _RULING_RE.search(sentence):
        return ClaimType.RULING
    if _FIRST_PERSON_FEELING_RE.search(sentence):
        # "I feel like Allah hates me" — the user's felt experience, not an
        # asserted prediction about Allah. Companion prose (§25 warmth).
        return ClaimType.PLAIN
    if _PREDICTION_RE.search(sentence):
        return ClaimType.PREDICTION
    if _CAUSAL_RE.search(sentence):
        return ClaimType.CAUSAL_CLAIM
    if _INFERENCE_RE.search(sentence):
        return ClaimType.INFERENCE
    if _ATTRIBUTION_RE.search(sentence):
        return ClaimType.ATTRIBUTION
    if _DIRECT_FACT_RE.search(sentence):
        return ClaimType.DIRECT_FACT
    if _GENERALIZATION_RE.search(sentence):
        return ClaimType.GENERALIZATION
    if _INTERPRETATION_RE.search(sentence):
        return ClaimType.INTERPRETATION
    if _RELIGIOUS_ANY_RE.search(sentence):
        return ClaimType.PARAPHRASE
    return ClaimType.PLAIN


# ------------------------------------------------- inference boundaries (§9)
@dataclass
class TypedClaim:
    sentence: str
    claim_type: ClaimType
    requirement: EvidenceRequirement
    has_citation: bool = False
    citations: list[str] = field(default_factory=list)
    dependency_on: str | None = None  # sentence this claim was inferred FROM
    is_inference_boundary: bool = False  # introduced by therefore/thus/...

    @property
    def needs_evidence(self) -> bool:
        return self.claim_type is not ClaimType.PLAIN

    @property
    def is_high_risk(self) -> bool:
        """Claims whose unsupported escape is unacceptable (§28 critical)."""
        return self.claim_type in (
            ClaimType.GUARANTEE, ClaimType.RULING, ClaimType.DIAGNOSIS,
            ClaimType.CAUSAL_CLAIM, ClaimType.PREDICTION,
            ClaimType.ATTRIBUTION, ClaimType.GENERALIZATION,
        )


def extract_typed_claims(text: str) -> list[TypedClaim]:
    """Sentence-level typed claim extraction with inference-boundary linking.

    §9: 'A. Therefore B.' -> A is judged alone; B carries dependency_on=A
    and must be INDEPENDENTLY supported. §20: uncited B must still be
    extracted and judged, never silently skipped."""
    from agent.validators.claims import CITATION_MARKER_RE

    typed: list[TypedClaim] = []
    prev_sentence: str | None = None
    for sentence in _sentences(text):
        if len(sentence) < 12:
            continue
        citations = [m.group(0) for m in CITATION_MARKER_RE.finditer(sentence)]
        ctype = classify_claim_type(sentence)
        boundary = bool(_INFERENCE_RE.search(sentence))
        typed.append(TypedClaim(
            sentence=sentence,
            claim_type=ctype,
            requirement=CLAIM_STRENGTH_POLICY[ctype],
            has_citation=bool(citations),
            citations=citations,
            dependency_on=prev_sentence if boundary else None,
            is_inference_boundary=boundary,
        ))
        prev_sentence = sentence
    return typed


# ------------------------------------------------- dependency removal (§10)
def dependent_closure(removed_sentences: set[str],
                       claims: list[TypedClaim]) -> set[str]:
    """Transitive closure: if a removed claim was the premise of an
    inference, the inference (and anything inferred from IT) goes too."""
    removed = set(removed_sentences)
    changed = True
    while changed:
        changed = False
        for c in claims:
            if c.sentence in removed or c.dependency_on is None:
                continue
            if c.dependency_on in removed:
                removed.add(c.sentence)
                changed = True
    return removed
