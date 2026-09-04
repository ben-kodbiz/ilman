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
GENTLE_QUESTION_RE = re.compile(
    r"\?(\s|$)"
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

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "policy_problems": self.policy_problems,
            "companion_problems": self.companion_problems,
        }


class ResponseValidator:
    def __init__(self) -> None:
        self.policy_validator = PolicyValidator()

    def validate(self, text: str, policy: ResponsePolicy) -> CompanionValidation:
        policy_problems = self.policy_validator.validate(text, policy)
        companion_problems: list[str] = []

        if DEPENDENCY_RE.search(text):
            companion_problems.append("dependency-forming language")
        if HUMAN_PRETENSE_RE.search(text):
            companion_problems.append("pretends to be human")
        if DIAGNOSIS_RE.search(text):
            companion_problems.append("diagnosis-like language")
        if HARMFUL_RE.search(text):
            companion_problems.append("potentially harmful content")
        if policy.safety_override:
            lowered = text.lower()
            for guilt in ("haram", "sin ", "punish", "hellfire"):
                if guilt in lowered:
                    companion_problems.append(f"religious guilt in safety mode: {guilt!r}")
        ok = not policy_problems and not companion_problems
        return CompanionValidation(ok, policy_problems, companion_problems)
