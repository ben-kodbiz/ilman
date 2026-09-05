"""Companion intent + emotion classification (fix_me.md §3, §17).

Deterministic-first: regex/keyword classifiers for the full companion intent
set, emotion labels, severity (via safety classifier), and conversation-mode
selection. Lightweight by design — no model call in the hot path (§3: "Do not
make the 4B model perform every classification task").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.companion.safety import Severity, classify_safety
from agent.core.intent import IntentResult
from agent.core.intent import classify as classify_core

COMPANION_INTENTS = {
    "normal_chat", "islamic_question", "quran_question", "hadith_question",
    "fiqh_question", "emotional_support", "loneliness", "grief", "anxiety",
    "anger", "guilt", "fear", "confusion", "motivation", "spiritual_low",
    "gratitude", "relationship_problem", "life_problem", "crisis_signal",
    "quran_request", "dua_request", "reflection_request",
}

# emotion key -> (patterns EN, patterns MS/ID)
EMOTION_PATTERNS: dict[str, list[str]] = {
    "loneliness": [
        r"\b(lonely|aloneness)\b", r"\bno\s+one\b.*\b(cares?|understands?)\b",
        r"\bnobody\b.*\b(checks?|calls?|asks?|cares?)\b",
        r"\bfeel\s+(alone|isolated|left\s+out)\b", r"\bsepi\b", r"\bsendiri\b.*\b(rasa|saja)\b",
    ],
    "grief": [
        r"\b(grief|grieving|mourn\w*)\b", r"\bi\s+miss\s+(him|her|them|my\b|someone)",
        r"\bmiss\s+(him|her|them|someone)\b.*\b(so\s+much|tonight|every\s+day)\b",
        r"\bpassed\s+away\b", r"\bdied\b", r"\bdeath\b.*\b(family|friend|father|mother)\b",
        r"\bkematian\b", r"\bsedih\b.*\b(kematian|hilang)\b",
    ],
    "sadness": [
        r"\bfeel\s+sad\b", r"\bfeeling\s+sad\b", r"\bi'?m\s+so\s+down\b",
        r"\bsad\b.*\b(as\s+a\s+muslim|lately|inside)\b",
        r"\bterrible\s+day\b", r"\bsedih\b",
    ],
    "anxiety": [
        r"\b(anxious|anxiety|panic\w*|worried|nervous|on\s+edge)\b",
        r"\bcan'?t\s+stop\s+worrying\b", r"\brasa\s+cemas\b", r"\bgelisah\b",
        r"\bdepress(ed|ion|ive)?\b", r"\bkemurungan\b", r"\bduka\s+nestapa\b",
    ],
    "anger": [
        r"\b(angry|furious|mad\s+at|rage)\b", r"\bso\s+angry\b", r"\bmarah\b", r"\bpanas\s+baran\b",
    ],
    "guilt": [
        r"\bfeel\s+guilty\b", r"\bguilt\b", r"\bmy\s+fault\b", r"\bi\s+(sinned|failed)\b",
        r"\brasa\s+bersalah\b", r"\bdosa\b.*\bsaya\b",
    ],
    "fear": [
        r"\b(scared|afraid|terrified|fear)\b", r"\btakut\b",
    ],
    "confusion": [
        r"\b(confused|lost|don'?t\s+know\s+what\s+to\s+do)\b",
        r"\bdon'?t\s+understand\b.*\b(myself|life)\b", r"\bconfused\b", r"\bkeliru\b",
    ],
    "motivation": [
        r"\b(motivat\w+|inspir\w+|lazy|procrastinat\w+|can'?t\s+start)\b",
        r"\bno\s+(drive|energy|motivation)\b", r"\bmalas\b",
    ],
    "spiritual_low": [
        r"\b(spiritually\s+empty|empty\s+inside|far\s+from\s+(allah|god|islam))\b",
        r"\biman\b.*\b(low|lemah|turun)\b", r"\bfeel\s+disconnected\s+from\s+allah\b",
        r"\bno\s+khushu\b", r"\bsolat\b.*\b(rasa\s+kosong|tidak\s+khusyuk)\b",
        r"\b(prayers?|salat|solat|worship|ibadah)\b.*\b(empty|meaningless|mechanical|distant)\b",
        r"\bfeel(s|ing)?\s+empty\b.*\b(prayer|worship|lately)\b",
    ],
    "gratitude": [
        r"\b(grateful|thankful|alhamdulillah|blessed)\b", r"\bsyukur\b",
    ],
}

FIRST_PERSON_RE = re.compile(
    r"\b(i|i'm|im|i\s+feel|i\s+am|saya|aku)\b", re.IGNORECASE
)
ISLAMIC_QUESTION_RE = re.compile(
    r"\b(islam|quran|qur'?an|hadith|sunnah|allah|prophet|muslim|muslims|fiqh|"
    r"halal|haram|salat|solat|prayer|fasting|ramadan|zakah|tafsir|ayah|surah|"
    r"verse|dua)\b",
    re.IGNORECASE,
)
QUESTION_MARK_RE = re.compile(r"\?")
QURAN_REQUEST_RE = re.compile(
    r"(give|show|share|tell)\s+me\b.*\b(quran|verse|ayah)", re.IGNORECASE
)
HADITH_QUESTION_RE = re.compile(r"\bhadith\b", re.IGNORECASE)
DUA_REQUEST_RE = re.compile(r"\bdua\b", re.IGNORECASE)
REFLECTION_RE = re.compile(r"\b(reflect|contemplate|ponder|tadabbur)\b", re.IGNORECASE)
RELATIONSHIP_RE = re.compile(
    r"\b(my\s+(wife|husband|friend|mother|father|parents|family|marriage))\b", re.IGNORECASE
)


@dataclass
class CompanionIntent:
    intent: str
    emotion: str | None = None
    emotion_confidence: float = 0.0
    severity: Severity = Severity.ORDINARY_DISTRESS
    is_question: bool = False
    first_person: bool = False
    islamic_requested: bool = False
    needs_islamic_guidance: bool = False
    needs_clarification: bool = False
    core: IntentResult | None = None
    matched_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "emotion": self.emotion,
            "emotion_confidence": round(self.emotion_confidence, 2),
            "severity": self.severity.value,
            "needs_islamic_guidance": self.needs_islamic_guidance,
            "needs_clarification": self.needs_clarification,
        }


def _detect_emotions(text: str) -> list[tuple[str, float]]:
    hits: list[tuple[str, float]] = []
    for emotion, patterns in EMOTION_PATTERNS.items():
        score = 0
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                score += 1
        if score:
            # simple confidence: distinct pattern hits, capped
            hits.append((emotion, min(0.6 + 0.15 * (score - 1), 0.95)))
    return sorted(hits, key=lambda x: -x[1])


def classify_companion(text: str) -> CompanionIntent:
    """Full companion classification. Deterministic; fast; no model."""
    safety = classify_safety(text)
    if safety.is_high_risk:
        return CompanionIntent(
            intent="crisis_signal",
            emotion="crisis",
            emotion_confidence=1.0,
            severity=Severity.HIGH_RISK,
            first_person=bool(FIRST_PERSON_RE.search(text)),
        )
    core = classify_core(text)
    emotions = _detect_emotions(text)
    top_emotion, confidence = (emotions[0] if emotions else (None, 0.0))

    is_question = bool(QUESTION_MARK_RE.search(text) or core.is_question)
    first_person = bool(FIRST_PERSON_RE.search(text))
    islamic = bool(ISLAMIC_QUESTION_RE.search(text))
    quran_request = bool(QURAN_REQUEST_RE.search(text))
    # past-tense thanks/gratitude about earlier content is NOT a new request:
    # "That verse helped, thank you." -> gratitude, not quran_question
    gratitude_re = re.compile(
        r"\b(thank you|thanks|that helped|helped me|alhamdulillah|syukur)\b", re.IGNORECASE
    )
    is_gratitude_reply = bool(gratitude_re.search(text)) and not QUESTION_MARK_RE.search(text)
    dua_request = bool(DUA_REQUEST_RE.search(text))
    reflection = bool(REFLECTION_RE.search(text))
    relationship = bool(RELATIONSHIP_RE.search(text))
    hadith_request = bool(re.search(
        r"(give|show|share|tell)\s+me\b.*\bhadith\b", text, re.IGNORECASE
    ))

    # intent priority ladder
    if is_gratitude_reply and not quran_request and not hadith_request:
        # acknowledging help received: companion continuation, no retrieval
        intent = "gratitude" if top_emotion == "gratitude" else "normal_chat"
    elif quran_request:
        intent = "quran_request"
    elif dua_request:
        # dua mentions route to dua regardless of question form ("Is there
        # any dua for removing depression?" is still a dua request)
        intent = "dua_request"
    elif reflection and not is_question:
        intent = "reflection_request"
    elif hadith_request:
        intent = "hadith_question"
    elif core.hadith_refs or (HADITH_QUESTION_RE.search(text) and is_question):
        intent = "hadith_question"
    elif core.quran_refs or core.intent in ("quran_lookup", "quran_search"):
        intent = "quran_question"
    elif islamic and is_question and emotions:
        # "Is it wrong to feel sad as a Muslim?" — islamic framing of a
        # personal feeling: RAG with empathy (policy engine handles the blend)
        intent = "islamic_question"
    elif emotions and first_person and not (islamic and is_question):
        intent = top_emotion if top_emotion in {
            "loneliness", "grief", "sadness", "anxiety", "anger", "guilt",
            "fear", "confusion", "spiritual_low",
        } else "emotional_support"
    elif emotions:
        intent = "emotional_support"
    elif relationship and first_person:
        intent = "relationship_problem"
    elif first_person and not is_question and len(text.split()) <= 12:
        intent = "emotional_support"  # short unexplained personal statement
    elif islamic and is_question:
        intent = "fiqh_question" if re.search(
            r"\b(fiqh|halal|haram|ruling|permissible)\b", text, re.IGNORECASE
        ) else "islamic_question"
    elif islamic:
        intent = "islamic_question"
    else:
        intent = "normal_chat"  # generic small talk / plain questions

    # guidance decision (§10): explicit islamic ask -> yes; emotional first
    # statement without islamic words -> offer, don't dump
    needs_guidance = intent in (
        "quran_request", "hadith_question", "quran_question",
        "islamic_question", "fiqh_question", "dua_request", "reflection_request",
    ) or bool(
        emotions and (islamic or is_question)
        and intent not in ("gratitude", "normal_chat")
    )

    needs_clarification = bool(
        emotions and not is_question and len(text.split()) < 15
    ) or intent in ("emotional_support", "loneliness", "grief") and not is_question

    return CompanionIntent(
        intent=intent,
        emotion=top_emotion,
        emotion_confidence=confidence,
        severity=safety.severity,
        is_question=is_question,
        first_person=first_person,
        islamic_requested=islamic,
        needs_islamic_guidance=needs_guidance,
        needs_clarification=needs_clarification,
        core=core,
        matched_patterns=[e[0] for e in emotions],
    )
