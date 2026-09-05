"""Safety / crisis classifier (fix_me.md §9).

Separate from normal emotional support. Deterministic pattern matching first
(EN + Malay/Indonesian); a model pass can be added later behind the same
interface. NEVER lets the companion continue normal mode on a high-risk
signal: routing returns a canned, compassionate safety response with real
contact suggestions — no instructions, no romanticizing, no religious guilt.

The classifier is intentionally high-recall: ambiguous crisis phrasing routes
to safety, where the canned response is safe for both true and false
positives. A false positive costs a slightly cautious reply; a false negative
can cost a life.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    ORDINARY_DISTRESS = "ordinary_distress"
    MODERATE_DISTRESS = "moderate_distress"
    HIGH_RISK = "high_risk"
    UNKNOWN = "unknown"


# Direct self-harm / suicide ideation signals (EN + MS/ID)
HIGH_RISK_PATTERNS = [
    r"\bkill\s+myself\b",
    r"\bkilling\s+myself\b",
    r"\bsuicid\w*",
    r"\bend\s+(my|it\s+all|my\s+life)\b",
    r"\bending\s+it\s+all\b",
    r"\btake\s+my\s+(own\s+)?life\b",
    r"\bwant\s+to\s+die\b",
    r"\bwish\s+(i\s+)?w(as)?ere?\s+dead\b",
    r"\bbetter\s+off\s+dead\b",
    r"\bno\s+reason\s+to\s+(live|go\s+on)\b",
    r"\bhurt\s+(myself|others)\b",
    r"\bharm(ing)?\s+(myself|someone|others)\b",
    r"\bcut(ting)?\s+my\s?(self|wrists?|arms?)\b",
    r"\boverdose\b",
    r"\bhow\s+(do|can)\s+i\s+(die|kill)\b",
    r"\bnot\s+want\s+to\s+(live|be\s+here)\b",
    r"\bdon'?t\s+want\s+to\s+(live|be\s+here)\b",
    r"\bdoesn'?t\s+matter\s+if\s+i\s+(live|die)\b",
    # Malay / Indonesian
    r"\bnak\s+bunuh\s+diri\b",
    r"\bbunuh\s+diri\b",
    r"\bmahu\s+mati\b",
    r"\bnak\s+mati\b",
    r"\btak\s+nak\s+hidup\b",
    r"\nmau\s+mati\b",
    r"\btinggalkan\s+dunia\b",
]

# Harm-to-others signals
HARM_OTHERS_PATTERNS = [
    r"\bkill\s+(someone|him|her|them|people|everybody)\b",
    r"\bhurt\s+(someone|him|her|them)\b.*\b(if|when)\b",
    r"\bmake\s+(them|him|her)\s+pay\b.*\bhurt\b",
]

# Moderate distress: heavy hopelessness/self-hate WITHOUT direct intent
MODERATE_PATTERNS = [
    r"\bhate\s+myself\b",
    r"\bworthless\b",
    r"\bhopeless\b",
    r"\bno\s+point\s+in\s+(living|anything)\b",
    r"\bcan'?t\s+(take|do)\s+this\s+anymore\b",
    r"\bgiv(e|ing)\s+up\b",
    r"\bwhat'?s\s+the\s+point\b",
    r"\bworthless\s+person\b",
    # MS/ID
    r"\btak\s+ada\s+guna\b",
    r"\sudah\s+give\s+up\b",
]

_HIGH_RE = re.compile("|".join(HIGH_RISK_PATTERNS) + "|" + "|".join(HARM_OTHERS_PATTERNS),
                      re.IGNORECASE)
_MODERATE_RE = re.compile("|".join(MODERATE_PATTERNS), re.IGNORECASE)


@dataclass
class SafetyResult:
    severity: Severity
    matched: str = ""

    @property
    def is_high_risk(self) -> bool:
        return self.severity is Severity.HIGH_RISK


def classify_safety(text: str) -> SafetyResult:
    m = _HIGH_RE.search(text)
    if m:
        return SafetyResult(Severity.HIGH_RISK, m.group(0))
    m = _MODERATE_RE.search(text)
    if m:
        return SafetyResult(Severity.MODERATE_DISTRESS, m.group(0))
    return SafetyResult(Severity.ORDINARY_DISTRESS)


# The canned high-risk response. Model-free on purpose (fix_me.md §9: no
# romanticizing, no religious guilt, no normal companion mode). Locale-aware
# contact lines; never fabricated hotlines — generic services only.
SAFETY_RESPONSE_EN = (
    "I'm really glad you told me this, and I want to be honest with you: what "
    "you're describing sounds heavier than something I should handle alone, "
    "and you shouldn't have to carry it alone either.\n\n"
    "Please reach out to a person who can actually be there with you:\n"
    "- Someone you trust who is nearby — a friend, family member, or neighbour. "
    "Even one message tonight can matter.\n"
    "- If you feel in immediate danger of hurting yourself, please contact your "
    "local emergency number right away.\n"
    "- If you can, talk to a doctor, counselor, or a local crisis / mental-health "
    "support line — they are trained for exactly this.\n\n"
    "If you want, you can keep talking to me here too — I'm not going anywhere. "
    "But please let one real person in tonight. You matter more than you feel "
    "right now."
)

SAFETY_RESPONSE_MS = (
    "Terima kasih kerana beritahu saya ini. Saya nak jujur: apa yang awak "
    "ceritakan terasa lebih berat daripada sesuatu yang patut awak tanggung "
    "sorang-sorang.\n\n"
    "Sila hubungi seseorang yang boleh benar-benar bersama awak:\n"
    "- Seseorang yang awak percayai dan berdekatan — kawan, keluarga, atau jiran. "
    "Satu mesej pun boleh membantu.\n"
    "- Jika awak rasa dalam bahaya hendak mencederakan diri, sila hubungi "
    "nombor kecemasan tempatan sekarang.\n"
    "- Jika boleh, bercakap dengan doktor, kaunselor, atau talian sokongan "
    "kesihatan mental di tempat awak — mereka terlatih untuk situasi ini.\n\n"
    "Kalau awak nak, kita boleh terus berbual di sini juga. Tapi tolong "
    "benarkan satu orang lain tahu malam ini. Awak lebih berharga daripada "
    "yang awak rasa sekarang."
)


def safety_response(lang: str = "en") -> str:
    return SAFETY_RESPONSE_MS if lang.startswith(("ms", "id")) else SAFETY_RESPONSE_EN
