"""Query planner (fixme_v3 §6-7).

Before retrieval, normalize the question into a structured information
need — modern terms expand to classical candidate concepts, the requested
object decides what shapes of evidence count, and source preference orders
the legs. Deterministic; no model call.

    {"intent": "dua_request", "topic": "emotional_distress",
     "requested_object": "specific_dua", "condition": "depression",
     "source_preference": ["hadith", "quran"]}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# modern psychological term -> classical concept expansions (§7: never a
# 1:1 mapping; these are RETRIEVAL candidates, not claimed synonyms)
MODERN_TO_CLASSICAL: dict[str, list[str]] = {
    "depression": [
        "grief and anxiety", "worry and sorrow", "hearts find rest",
        "distress hardship", "heaviness of heart", "despair of Allah's mercy",
    ],
    "anxiety": ["anxiety and grief", "worry fear of the future", "hearts find rest in remembrance"],
    "stress": ["hardship patience", "burden Allah does not overload the soul", "reliance on Allah"],
    "trauma": ["calamity patience and prayer", "to Allah we belong and to Him we return"],
    "loneliness": ["Allah is near responds to dua", "companionship believers brotherhood"],
    "addiction": ["repentance returning to Allah", "overcoming desires self-control"],
}

# well-known concept queries -> canonical source anchors (deterministic
# retrieval by citation ID — no ranking lottery; these are among the most
# famous narrations and their IDs are stable in the dataset)
CONCEPT_ANCHORS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"pillar[s]?\s+of\s+islam|arkan\s+al\s+islam|five\s+pillar", re.I),
     ["hadith:sahih-muslim:112", "hadith:sahih-muslim:113", "hadith:sahih-muslim:114",
      "hadith:sahih-muslim:115", "hadith:sahih-muslim:116"]),
    (re.compile(r"pillar[s]?\s+of\s+(i?man|faith|belief)", re.I),
     ["hadith:sahih-muslim:112", "hadith:sahih-muslim:113"]),
    (re.compile(r"how\s+to\s+pray|step[s]?\s+of\s+(the\s+)?prayer|"
                r"method\s+of\s+salat|how\s+to\s+perform\s+salat", re.I),
     ["hadith:sahih-bukhari:631", "hadith:sahih-bukhari:6251"]),
    (re.compile(r"etiquette\s+of\s+eating|how\s+to\s+eat", re.I),
     ["hadith:sahih-muslim:2022", "hadith:sahih-bukhari:5376"]),
    # opening/meta queries -> the first surah, deterministically (the
    # fatihah canary: 'how does the Quran begin' has no lexical handle)
    (re.compile(
        r"how\s+does\s+(the\s+)?(quran|qur'?an)\s+(begin|start|open)|"
        r"(first|opening)\s+(surah|chapter)|what\s+is\s+the\s+first\s+surah|"
        r"opening\s+of\s+(the\s+)?(the\s+)?(quran|qur'?an)", re.I),
     ["quran:1:1", "quran:1:2", "quran:1:3"]),
    # ayat al-kursi meta
    (re.compile(r"throne\s+verse|ayat\s*al.?kursi|ayatul\s+kursi", re.I),
     ["quran:2:255"]),
    # beginner / learning-path guidance: the hadith of Jibril (the complete
    # framework of the religion — Islam/Iman/Ihsan) + the five pillars;
    # a beginner needs the map, not random lexical matches
    (re.compile(
        r"\b(learn|learning|study|begin|beginner|start|new)\b.*\b(islam|muslim|deen|faith|religion)\b"
        r"|\b(islam|deen|religion)\b.*\b(learn|study|begin|beginner|new\s+to)\b"
        r"|\b(how\s+do\s+i\s+start|where\s+do\s+i\s+start)\b",
        re.I),
     ["hadith:sunan-ibn-majah:63", "hadith:sahih-muslim:112",
      "hadith:sahih-muslim:113", "hadith:sahih-muslim:114",
      "hadith:sahih-muslim:115", "hadith:sahih-muslim:116"]),
]
CONCEPT_QUERIES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"pillar[s]?\s+of\s+islam|arkan\s+al\s+islam|five\s+pillar", re.I),
     ["Islam is built upon five pillars", "testimony of faith prayer zakat fasting pilgrimage",
      "shahadah salah zakat sawm hajj"]),
    (re.compile(r"pillar[s]?\s+of\s+(i?man|faith|belief)", re.I),
     ["pillars of faith belief in Allah His angels His books His messengers"]),
    (re.compile(r"pillar[s]?\s+of\s+ihsan", re.I),
     ["excellence worship Allah as if you see Him"]),
    (re.compile(r"articles\s+of\s+faith|six\s+belief", re.I),
     ["belief in Allah His angels His books His messengers the Last Day"]),
    (re.compile(r"conditions\s+of\s+(the\s+)?shahadah|how\s+to\s+become\s+a\s+muslim", re.I),
     ["testimony there is no god but Allah Muhammad is the Messenger"]),
    # learning-path guidance: retrieval expansions aimed at study-guidance
    # material (the hadith of Jibril, degrees of the deen, seeking knowledge)
    (re.compile(
        r"\b(learn|learning|study|begin|beginner|start|new)\b.*\b(islam|muslim|deen|faith|religion)\b"
        r"|\b(islam|deen|religion)\b.*\b(learn|study|begin|beginner|new\s+to)\b"
        r"|\b(how\s+do\s+i\s+start|where\s+do\s+i\s+start)\b",
        re.I),
     ["what is Islam iman ihsan the hadith of Jibril",
      "Islam is built upon five pillars",
      "degrees of Islam seeking knowledge path"]),
]

TOPIC_PATTERNS = [
    (re.compile(r"depress|anxiet|anxious|stress|worry|grief|sad|distress|despair|sorrow", re.I),
     "emotional_distress"),
    (re.compile(r"lonel|alone|isolat", re.I), "emotional_distress"),
    (re.compile(r"patience|sabr|persever", re.I), "patience"),
    (re.compile(r"mercy|forgiveness|repent", re.I), "mercy_repentance"),
    (re.compile(r"provision|rizq|sustain", re.I), "provision"),
    (re.compile(r"protection|refuge|evil eye|sihr|black magic", re.I), "protection"),
]

OBJECT_PATTERNS = [
    (re.compile(r"\bdua\b|supplication|prayer\s+for|zikr|dhikr|remembrance\s+formula", re.I),
     "specific_dua"),
    (re.compile(r"verse|ayah|surah|quran\s+says|chapter", re.I), "verse"),
    (re.compile(r"hadith|narration|prophet\s+said|sunnah", re.I), "hadith"),
    (re.compile(r"explain|meaning|tafsir|interpret", re.I), "explanation"),
    (re.compile(r"story|example|seerah|biography", re.I), "narrative"),
]


@dataclass
class QueryPlan:
    raw: str
    intent: str
    topic: str | None = None
    requested_object: str | None = None
    condition: str | None = None
    source_preference: list[str] = field(default_factory=list)
    retrieval_terms: list[str] = field(default_factory=list)
    anchor_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "intent": self.intent, "topic": self.topic,
            "anchor_citations": self.anchor_citations,
            "requested_object": self.requested_object,
            "condition": self.condition,
            "source_preference": self.source_preference,
            "retrieval_terms": self.retrieval_terms,
        }


def plan_query(message: str, intent: str) -> QueryPlan:
    plan = QueryPlan(raw=message, intent=intent)

    for pattern, topic in TOPIC_PATTERNS:
        m = pattern.search(message)
        if m:
            plan.topic = topic
            plan.condition = m.group(0).lower()
            break

    for pattern, obj in OBJECT_PATTERNS:
        if pattern.search(message):
            plan.requested_object = obj
            break

    # deterministic source anchors for well-known concepts (§6: no ranking
    # lottery — the canonical narration goes straight into the pack)
    for pattern, anchors in CONCEPT_ANCHORS:
        if pattern.search(message):
            plan.anchor_citations = anchors
            break

    # source preference by requested object (§14 source_priority)
    if plan.requested_object == "specific_dua":
        plan.source_preference = ["hadith", "quran"]
    elif plan.requested_object == "hadith":
        plan.source_preference = ["hadith"]
    elif plan.requested_object == "verse":
        plan.source_preference = ["quran", "tafsir"]
    elif plan.requested_object == "explanation":
        plan.source_preference = ["tafsir", "quran", "hadith"]
    else:
        plan.source_preference = ["quran", "hadith", "tafsir"]

    # retrieval terms: raw + concept-query expansions + classical expansions
    plan.retrieval_terms = [message]
    lowered = message.lower()
    for pattern, expansions in CONCEPT_QUERIES:
        if pattern.search(message):
            plan.retrieval_terms.extend(expansions)
    for modern, classical in MODERN_TO_CLASSICAL.items():
        if re.search(rf"\b{modern}\b", lowered):
            plan.retrieval_terms.extend(classical)
    # fixme_v3.1 §14-15: dua requests retrieve a CANDIDATE concept set,
    # never one hard-coded dua as the answer. The planner proposes; the
    # evidence layer (quarantine + judge) decides applicability.
    if plan.requested_object == "specific_dua":
        plan.retrieval_terms.insert(
            1, "O Allah I seek refuge in You from anxiety and grief"
        )
        # additional distinct supplication contexts so no single dua is
        # structurally privileged; the judge filters what applies
        extra_candidates = [
            "supplication for relief from hardship and distress",
            "Prophet taught dua morning and evening remembrance",
        ]
        for term in extra_candidates:
            if term not in plan.retrieval_terms:
                plan.retrieval_terms.append(term)
    plan.retrieval_terms = plan.retrieval_terms[:9]
    return plan
