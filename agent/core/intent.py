"""Intent classification + entity extraction (agentodo.md §10, §12).

Deterministic rules before any model call: Qur'an references, hadith
references, obvious intents. Small-model fallback for ambiguous text can be
added later behind the same interface (§10: "small model or deterministic
rules before calling the main model").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent.tools.quran_refs import (
    KNOWN_AYAH_ALIASES,
    SURAH_NAMES,
    extract_references,
)

# Surah-name references in queries: "Al-Ikhlas", "Al-Baqarah", "an-Nas"...
SURAH_NAME_ALIASES = {alias: number for alias, number in SURAH_NAMES.items()}
SURAH_NAME_RE = re.compile(
    r"\b(?:surah\s+|chapter\s+)?([a-z][a-z'’\- ]{2,25})\b", re.IGNORECASE
)

INTENTS = {
    "quran_lookup": "deterministic Qur'an reference present",
    "quran_search": "Arabic or 'verse/ayah' phrasing without a reference",
    "hadith_lookup": "collection + number present",
    "hadith_search": "hadith/prophet/narrated phrasing",
    "study_note": "save/note/remember instruction",
    "history": "history/recent/bookmark phrasing",
    "question": "general question needing evidence retrieval",
}

# Emotional statements ("I am lonely") share no vocabulary with the corpus's
# comfort verses, so lexical AND plain-vector retrieval both miss them. Expand
# detected emotional registers with explicit concepts the corpus does use.
EMOTIONAL_PATTERNS: dict[str, list[str]] = {
    "lonely": ["loneliness companionship", "Allah is near responds to dua",
               "feeling abandoned comfort from Allah", "hearts find rest in remembrance"],
    "alone": ["loneliness companionship", "Allah is near responds to dua",
              "do not grieve Allah does not forsake"],
    "sad": ["sadness grief hearts find rest", "do not despair of Allah's mercy"],
    "depressed": ["sadness grief hearts find rest", "do not despair of Allah's mercy",
                  "Allah is near responds to dua"],
    "anxious": ["anxiety fear hearts find rest in remembrance", "Allah is near responds to dua"],
    "afraid": ["fear of Allah trust in Allah", "do not fear Allah protects the believers"],
    "worried": ["anxiety fear hearts find rest in remembrance", "trust in Allah put your reliance"],
    "grieving": ["grief loss patience and prayer", "do not despair of Allah's mercy",
                 "to Allah we belong and to Him we return"],
    "tired": ["weariness patience and prayer", "Allah does not burden a soul beyond capacity"],
    "hopeless": ["do not despair of Allah's mercy", "Allah is near responds to dua",
                 "Allah does not burden a soul beyond capacity"],
}
EMOTIONAL_RE = re.compile(
    r"\b(i am|i feel|feeling|i'm)\s+(so\s+|very\s+|really\s+|completely\s+|all\s+)?"
    + r"(" + "|".join(EMOTIONAL_PATTERNS) + r")\b"
    + r"|\b(lonely|alone|sad|depressed|anxious|afraid|worried|grieving|hopeless)\b",
    re.IGNORECASE,
)

QURAN_REF_RE = re.compile(
    r"\b(?:quran|qur'an|surah|ayah|verse|chapter)\s*[:\-]?\s*\d{1,3}(?:[:.\s]\s*\d{1,3})?\b"
    r"|\b\d{1,3}\s*:\s*\d{1,3}\b",
    re.IGNORECASE,
)
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
HADITH_COLLECTION_RE = re.compile(
    r"\b(bukhari|muslim|abu\s*dawud|tirmidhi|nasai|nasa'i|ibn\s*majah)\b", re.IGNORECASE
)
HADITH_NUMBER_RE = re.compile(r"\b(?:no\.?|number|#|hadith)\s*(\d{1,5})\b", re.IGNORECASE)
HADITH_PHRASE_RE = re.compile(r"\b(hadith|prophet|narrated|messenger|sunnah)\b", re.IGNORECASE)
NOTE_RE = re.compile(r"\b(save|note|remember|bookmark)\b.*\b(note|this|ayah|verse|hadith)?\b", re.IGNORECASE)
HISTORY_RE = re.compile(r"\b(history|recent|bookmark|previously|earlier)\b", re.IGNORECASE)
QUESTION_RE = re.compile(r"^[^?]*\?|^(what|who|why|how|when|where|which|explain|tell|did|is|are|does)\b", re.IGNORECASE)

# Map text collection names -> registry source ids
COLLECTION_ALIASES = {
    "bukhari": "sahih-bukhari",
    "sahih bukhari": "sahih-bukhari",
    "muslim": "sahih-muslim",
    "sahih muslim": "sahih-muslim",
    "abu dawud": "sunan-abu-dawud",
    "abudawud": "sunan-abu-dawud",
    "tirmidhi": "jami-at-tirmidhi",
    "nasai": "sunan-an-nasai",
    "nasa'i": "sunan-an-nasai",
    "ibn majah": "sunan-ibn-majah",
}


@dataclass
class IntentResult:
    intent: str
    quran_refs: list[dict] = field(default_factory=list)
    hadith_refs: list[dict] = field(default_factory=list)  # {"collection": id, "number": n}
    is_question: bool = False
    arabic: bool = False
    emotional: bool = False
    concept_expansions: list[str] = field(default_factory=list)

    @property
    def routed_task_class(self) -> str:
        """Map intent -> model routing task class (§3)."""
        if self.intent in ("quran_lookup", "hadith_lookup"):
            return "simple_rag"  # deterministic tools answer; light synthesis
        if self.intent in ("quran_search", "hadith_search", "question"):
            return "complex_rag"
        return "simple_chat"


def _alias_refs(text: str) -> list[dict]:
    """Well-known ayah aliases ('Ayat al-Kursi', 'Ayatul kursi') and surah
    names ('Al-Ikhlas', 'Al-Baqarah 255') resolve deterministically (§14)."""
    refs: list[dict] = []
    seen: set[tuple[int, int]] = set()
    lowered = " " + re.sub(r"\s+", " ", text.lower()) + " "

    def _add(surah: int, ayah: int) -> None:
        if (surah, ayah) not in seen:
            refs.append({"surah": surah, "ayah": ayah})
            seen.add((surah, ayah))

    for alias, (s, a) in KNOWN_AYAH_ALIASES.items():
        if alias in lowered:
            _add(s, a)
    # named-surah reference: "Al-Baqarah 255" / "in Al-Ikhlas" / "Surah Al-Baqarah"
    for alias, number in SURAH_NAME_ALIASES.items():
        pattern = re.compile(
            rf"(?:surah\s+|chapter\s+)?\b{re.escape(alias)}\b\s*(\d{{1,3}})?", re.IGNORECASE
        )
        m = pattern.search(text)
        if m:
            ayah = int(m.group(1)) if m.group(1) else 1
            if 1 <= ayah <= 286:
                _add(number, ayah)
    return refs


def _hadith_refs(text: str) -> list[dict]:
    """Collection + explicit number -> hadith lookup refs (deterministic).

    Number may come before or after the collection mention ('hadith no. 1 in
    Bukhari' / 'Bukhari 123').
    """
    refs = []
    seen: set[tuple[str, int]] = set()

    def _add(source_id: str, number: int) -> None:
        if number > 0 and (source_id, number) not in seen:
            refs.append({"collection": source_id, "number": number})
            seen.add((source_id, number))

    for m in HADITH_COLLECTION_RE.finditer(text):
        name = m.group(1).lower().replace("'", "")
        source_id = COLLECTION_ALIASES.get(name)
        if not source_id:
            continue
        after = text[m.end(): m.end() + 30]
        before = text[max(0, m.start() - 30): m.start()]
        nm_after = HADITH_NUMBER_RE.search(after) or re.search(r"#\s*(\d{1,5})\b", after)
        nm_before = re.search(
            r"\b(?:no\.?|number|#|hadith)\s*(\d{1,5})\b", before, re.IGNORECASE
        )
        if nm_after:
            _add(source_id, int(nm_after.group(1)))
        elif nm_before:
            _add(source_id, int(nm_before.group(1)))
    return refs


def classify(text: str) -> IntentResult:
    result = IntentResult(intent="question")
    result.arabic = bool(ARABIC_RE.search(text))
    result.is_question = bool(QUESTION_RE.search(text))
    result.quran_refs = extract_references(text)
    for ref in _alias_refs(text):
        if not any(r["surah"] == ref["surah"] and r["ayah"] == ref["ayah"] for r in result.quran_refs):
            result.quran_refs.append(ref)
    # emotional register detection (concept expansion for retrieval)
    if EMOTIONAL_RE.search(text):
        result.emotional = True
        lowered = text.lower()
        for emotion, expansions in EMOTIONAL_PATTERNS.items():
            if re.search(rf"\b{emotion}\b", lowered):
                result.concept_expansions.extend(expansions)
        result.concept_expansions = list(dict.fromkeys(result.concept_expansions))[:4]
    # explicit "surah 12" style refs
    surah_only = re.findall(
        r"\b(?:surah|chapter)\s+(\d{1,3})\b", text, re.IGNORECASE
    )
    for s in surah_only:
        if 1 <= int(s) <= 114 and not any(r["surah"] == int(s) for r in result.quran_refs):
            result.quran_refs.append({"surah": int(s), "ayah": 1})
    result.hadith_refs = _hadith_refs(text)
    # priority: note > history > lookups > searches > question
    if NOTE_RE.search(text) and result.is_question is False:
        result.intent = "study_note"
    elif HISTORY_RE.search(text) and not result.is_question:
        result.intent = "history"
    elif result.quran_refs:
        result.intent = "quran_lookup"
    elif result.hadith_refs:
        result.intent = "hadith_lookup"
    elif result.hadith_refs or HADITH_PHRASE_RE.search(text):
        result.intent = "hadith_search"
    elif result.arabic or re.search(r"\b(verse|ayah|quran|qur'an)\b", text, re.IGNORECASE):
        result.intent = "quran_search"
    elif result.is_question:
        result.intent = "question"
    else:
        result.intent = "question"
    return result
