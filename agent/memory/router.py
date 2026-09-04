"""Memory router (fixme_v2 §10-13).

Categories (§10): PROFILE / PREFERENCE / FACT / STUDY / CONVERSATION /
OPEN_THREAD / SUMMARY. Lifecycle (§11): importance -> stability -> privacy
-> deduplication -> save/reject. Retrieval (§12): relevant only, ranked;
three separate layers kept distinct (§13): recent context (state manager),
session state (state engine), long-term memory (this module).

Built on the v1 CompanionMemory storage (SQLite) with the §5C transient-
emotion gate — that invariant is preserved, not weakened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.companion.memory import CompanionMemory, FactRejected

CATEGORIES = ("profile", "preference", "fact", "study", "conversation",
              "open_thread", "summary")

# §11 privacy: content classes that must never auto-save
_PRIVACY_BLOCK_RE = re.compile(
    r"\b(suicid\w*|kill\w*\s+myself|self[-\s]?harm|overdose|abused?|"
    r"addicted|debt details|password|ic number|bank account)\b",
    re.IGNORECASE,
)
# §11 stability: transient markers mean the content is NOT stable
_TRANSIENT_RE = re.compile(
    r"\b(today|tonight|right now|currently|at the moment|temporarily|this morning)\b"
    r"|\b(i feel|i felt|feeling|i think)\b",
    re.IGNORECASE,
)
# §11 importance: what makes a candidate worth remembering
_IMPORTANT_RE = re.compile(
    r"\b(prefer|always|never|i am a|i'm a|i study|i work|my name|call me|"
    r"remember|keep in mind|i live|learning|memorizing)\b",
    re.IGNORECASE,
)

_EXTRACTORS: list[tuple[str, re.Pattern, str]] = [
    ("preference", re.compile(
        r"\bi\s+(?:really\s+)?prefer\s+(.{3,80})", re.IGNORECASE), "User prefers {0}"),
    ("preference", re.compile(
        r"\bplease\s+(?:always\s+)?(?:answer|explain|reply)\s+in\s+(.{3,40})", re.IGNORECASE),
        "User prefers responses in {0}"),
    ("profile", re.compile(r"\bmy name is\s+([A-Za-z' -]{2,40})", re.IGNORECASE),
     "User's name is {0}"),
    ("profile", re.compile(r"\bcall me\s+([A-Za-z' -]{2,30})", re.IGNORECASE),
     "User likes to be called {0}"),
    ("profile", re.compile(r"\bi(?:'m| am)\s+(?:a|an)\s+([a-z ]{3,40}?)[.,!?]", re.IGNORECASE),
     "User is a {0}"),
    ("study", re.compile(
        r"\bi(?:'m| am)\s+(?:currently\s+)?(?:studying|learning|memorizing)\s+"
        r"([^.!?\n]{3,80})", re.IGNORECASE), "User is studying {0}"),
    ("fact", re.compile(r"\bi live in\s+([A-Za-z' -]{2,40})", re.IGNORECASE),
     "User lives in {0}"),
    ("fact", re.compile(r"\bi work as\s+(.{3,60})", re.IGNORECASE),
     "User works as {0}"),
]


def _normalize(text: str) -> str:
    """Dedup key: lowercase, strip punctuation/whitespace variance."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


@dataclass
class MemoryCandidate:
    text: str
    category: str
    source: str = "inferred"  # inferred | explicit
    explicit: bool = False


class LifecycleVerdict:
    SAVE = "save"
    REJECT = "reject"


def evaluate_candidate(candidate: MemoryCandidate) -> tuple[str, str]:
    """§11 lifecycle: importance -> stability -> privacy -> dedup happens at
    the store. Returns (verdict, reason). Extractor-found candidates target
    explicit self-statements (preferences, names, study topics) — the pattern
    match itself is the importance signal."""
    text = candidate.text.strip()
    if not text:
        return LifecycleVerdict.REJECT, "empty"
    if _PRIVACY_BLOCK_RE.search(text):
        return LifecycleVerdict.REJECT, "privacy-sensitive content"
    if not candidate.explicit and _TRANSIENT_RE.search(text):
        return LifecycleVerdict.REJECT, "transient content (state, not memory)"
    return LifecycleVerdict.SAVE, "passed importance/stability/privacy"


class MemoryRouter:
    """Category-aware front door to the CompanionMemory store (§12)."""

    def __init__(self, memory: CompanionMemory):
        self.memory = memory

    # -- extraction (§12 extractor) ---------------------------------------
    def extract_candidates(self, message: str) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for category, pattern, template in _EXTRACTORS:
            for m in pattern.finditer(message):
                raw = " ".join(m.group(1).split())[:80]
                if raw:
                    candidates.append(
                        MemoryCandidate(text=template.format(raw), category=category)
                    )
        # explicit "remember that ..." instruction
        m = re.match(r"\bremember\b(?:\s+that)?\s+(.{3,200})", message, re.IGNORECASE)
        if m:
            candidates.append(
                MemoryCandidate(
                    text=" ".join(m.group(1).split())[:200],
                    category="fact", source="explicit", explicit=True,
                )
            )
        return candidates

    # -- lifecycle (§11) ----------------------------------------------------
    def route_incoming(self, message: str) -> dict:
        """Extract candidates from a user message, run the lifecycle, save
        what passes. §11 dedup runs before save. Messages that look like they
        carry rememberable content but produce no structured candidate still
        get an explicit REJECT decision (never silently swallowed)."""
        saved, rejected = [], []
        for cand in self.extract_candidates(message):
            verdict, reason = evaluate_candidate(cand)
            if verdict is LifecycleVerdict.SAVE:
                # §11 deduplication: skip facts already stored (normalized)
                normalized = _normalize(cand.text)
                existing = {
                    _normalize(f["fact"]) for f in self.memory.facts(limit=200)
                }
                if normalized in existing:
                    rejected.append({"fact": cand.text, "reason": "duplicate"})
                    continue
                try:
                    fid = self.memory.save_fact(
                        cand.text, category=cand.category, explicit=cand.explicit
                    )
                    saved.append({"id": fid, "fact": cand.text, "category": cand.category})
                except FactRejected as e:
                    rejected.append({"fact": cand.text, "reason": str(e)})
            else:
                rejected.append({"fact": cand.text, "reason": reason})
        # explicit transient self-report with no structured candidate -> explicit reject
        if not saved and not rejected and _TRANSIENT_RE.search(message):
            rejected.append({
                "fact": message.strip()[:120],
                "reason": "transient content (state, not memory)",
            })
        return {"saved": saved, "rejected": rejected}

    # -- retrieval (§12 ranking) -------------------------------------------
    def relevant(self, message: str, limit: int = 3) -> list[dict]:
        if not self.memory.memory_enabled:
            return []
        hits = self.memory.relevant_facts(message, limit=limit)
        # category enrichment for the context pack
        return [
            {"fact": h["fact"], "category": h.get("category", "fact")} for h in hits
        ]

    # -- controls (§12 forget/delete) ---------------------------------------
    def forget(self, fact_id: int) -> bool:
        return self.memory.forget_fact(fact_id)

    def clear_all(self) -> dict:
        return self.memory.clear_all()
