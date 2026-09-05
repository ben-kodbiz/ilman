"""Response validator (fixme_v2 §25).

Adds companion-level validation ON TOP of the existing religious validation
(source/citation/unsupported-claim stay in agent/validators/pipeline.py and
are never weakened):

  - policy compliance (question count, word budget, preachy opener)
  - safety compliance (no religious guilt in safety mode, no harmful content)
  - companion tone (dependency language, human-pretense, diagnosis)
  - follow-up validation (max one question; question should be gentle)

A response can be factually correct and still FAIL companion policy (§25).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.policy.companion_policy import PolicyValidator, ResponsePolicy

DEPENDENCY_RE = re.compile(
    r"\b(i am all you need|you only need me|no one else will understand|"
    r"i understand you better than anyone|i'?m always here instead|"
    r"instead of (your|other) (friends|family|people)|talk to me instead of)\b",
    re.IGNORECASE,
)
HUMAN_PRETENSE_RE = re.compile(
    r"\b(as a (muslim )?(man|woman|person)|when i was (young|a child)|i cried\b|"
    r"my (heart|wife|husband|children)|i (grew up|was born))\b",
    re.IGNORECASE,
)
DIAGNOSIS_RE = re.compile(
    r"\b(you (have|are suffering from) (depression|anxiety disorder|bipolar)|"
    r"clinical(ally)? depressed|your diagnosis|this is a symptom of)\b",
    re.IGNORECASE,
)
# §22: religious CLAIMS in prose that carry no citation marker. Empathy needs
# no citation; "The Prophet said ..." / "The Quran says ..." does. When an
# evidence pack exists but the claim cites nothing, it is unverified content.
RELIGIOUS_CLAIM_RE = re.compile(
    # "The Prophet said/told/taught/reminded..." / "Allah says..."
    r"\b(the\s+)?(prophet|quran|qur'?an|allah|rasul)\b[^.\n]{0,60}"
    r"\b(said|says|stated|states|tells|told|teaches|taught|describes|mentions|"
    r"reminds|reminded|promises|promised|commands|commanded|warns|warned|"
    r"guides|guided|advises|advised|encourages|encouraged)\b"
    # "according to the Quran/prophet/sunnah"
    r"|\baccording\s+to\s+(the\s+)?(quran|prophet|sunnah|hadith)\b"
    # rulings
    r"|\b(?:it\s+is|this\s+is)\s+(haram|halal|obligatory|fard|sunnah)\b"
    # "Islam teaches/says/reminds..."
    r"|\bislam\s+(?:also\s+)?(teaches|teach|says|states|tells|reminds|"
    r"emphasizes|encourages|promises|commands)\b"
    # "Allah is/does/will..." attribute claims
    r"|\ballah\s+(is|does not|will not|never|does)\b[^.\n]{0,80}"
    r"|\bin\s+islam,?\s+[^.\n]{0,80}",
    re.IGNORECASE,
)
CITATION_MARKER_RE = re.compile(
    r"\[(?:quran|hadith|tafsir|tafsir-en):[^\]]+\]|\b(?:quran|hadith):\S+",
    re.IGNORECASE,
)
HARMFUL_RE = re.compile(
    r"\b(how to (die|kill|overdose)|painless way|which bridge|rope|pills to take)\b",
    re.IGNORECASE,
)


@dataclass
class CompanionValidation:
    ok: bool
    policy_problems: list[str] = field(default_factory=list)
    companion_problems: list[str] = field(default_factory=list)
    uncited_religious_claims: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "policy_problems": self.policy_problems,
            "companion_problems": self.companion_problems,
            "uncited_religious_claims": self.uncited_religious_claims,
        }


class ResponseValidator:
    def __init__(self) -> None:
        self.policy_validator = PolicyValidator()

    def validate(self, text: str, policy: ResponsePolicy,
                 evidence_present: bool = False) -> CompanionValidation:
        policy_problems = self.policy_validator.validate(text, policy)
        companion_problems: list[str] = []
        uncited_claims: list[str] = []

        # RAG-route non-answers: the user asked a question, evidence exists,
        # and the 'answer' is generic listening empathy -> that is a wrong
        # answer for the mode, regardless of claim cleanliness.
        if evidence_present and policy.route == "rag":
            lowered = text.lower().strip()
            if lowered.startswith(("i hear you", "i'm listening", "i am listening", "if you want to tell me")):
                companion_problems.append(
                    "QA non-answer: listening fallback on a question route"
                )

        if DEPENDENCY_RE.search(text):
            companion_problems.append("dependency-forming language")
        if HUMAN_PRETENSE_RE.search(text):
            companion_problems.append("pretends to be human")
        if DIAGNOSIS_RE.search(text):
            companion_problems.append("diagnosis-like language")
        if HARMFUL_RE.search(text):
            companion_problems.append("potentially harmful content")
        # §22: religious claims in prose must carry a citation marker when
        # evidence exists; with NO evidence they are outright fabrication.
        for m in RELIGIOUS_CLAIM_RE.finditer(text):
            window = text[max(0, m.start() - 120): m.end() + 160]
            if not CITATION_MARKER_RE.search(window):
                snippet = m.group(0).strip()[:60]
                problem = (
                    "uncited religious claim (no evidence pack)"
                    if not evidence_present
                    else "uncited religious claim (evidence exists but not cited)"
                )
                uncited_claims.append(f"{problem}: {snippet!r}")
        if policy.safety_override:
            lowered = text.lower()
            for guilt in ("haram", "sin ", "punish", "hellfire"):
                if guilt in lowered:
                    companion_problems.append(f"religious guilt in safety mode: {guilt!r}")
        ok = not policy_problems and not companion_problems and not uncited_claims
        return CompanionValidation(ok, policy_problems, companion_problems, uncited_claims)
