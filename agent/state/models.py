"""Structured conversation state (fixme_v2 §2-4).

Machine-readable enums over free-form values; explicit mode/intent/emotion/
risk/user_goal/requires_rag/requires_followup fields; a state machine with
a ROUTE phase and special routes (SAFETY/RAG/MEMORY/COMPANION).

§9: emotion and intent are SEPARATE — emotion=loneliness with
intent=emotional_support, or emotion=anger with intent=islamic_question.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class Mode(StrEnum):
    QA = "qa"
    STUDY = "study"
    COMPANION = "companion"
    REFLECTION = "reflection"
    DUA = "dua"
    CRISIS = "crisis"


class Phase(StrEnum):
    IDLE = "idle"
    UNDERSTAND = "understand"
    ROUTE = "route"
    RESPOND = "respond"
    FOLLOW_UP = "follow_up"
    CONTINUE = "continue"


class Route(StrEnum):
    """Special routes out of the ROUTE phase (fixme_v2 §4)."""

    SAFETY = "safety"
    RAG = "rag"
    MEMORY = "memory"
    COMPANION = "companion"
    CHAT = "chat"


class Risk(StrEnum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"


class UserGoal(StrEnum):
    BE_HEARD = "be_heard"
    ANSWER = "answer"
    STUDY = "study"
    REFLECT = "reflect"
    DUA = "dua"
    UNSPECIFIED = "unspecified"


@dataclass
class ConversationState:
    """fixme_v2 §2: the structured state object. No free-form fields where a
    structured enum/value suffices."""

    session_id: str
    mode: Mode = Mode.QA
    phase: Phase = Phase.IDLE
    intent: str = "normal_chat"
    emotion: str | None = None          # §9: separate from intent
    risk: Risk = Risk.LOW
    user_goal: UserGoal = UserGoal.UNSPECIFIED
    requires_rag: bool = False
    requires_followup: bool = False
    turn_count: int = 0
    # continuity bookkeeping (kept structured; compact summaries, not dumps)
    open_threads: list[str] = field(default_factory=list)
    durable_notes: list[str] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)

    MAX_OPEN_THREADS = 4
    MAX_NOTES = 8

    def advance(self, target: Phase) -> None:
        """Allowed forward transitions only (§4 order)."""
        order = [
            Phase.IDLE, Phase.UNDERSTAND, Phase.ROUTE,
            Phase.RESPOND, Phase.FOLLOW_UP, Phase.CONTINUE,
        ]
        if order.index(target) >= order.index(self.phase):
            self.phase = target

    def note_thread(self, thread: str) -> None:
        thread = thread.strip()
        if thread and thread not in self.open_threads:
            self.open_threads.append(thread)
            self.open_threads = self.open_threads[-self.MAX_OPEN_THREADS:]

    def close_threads_matching(self, needle: str) -> None:
        """Close threads whose text shares a word-root with the needle
        ('lonely' closes 'loneliness discussion' via shared stem)."""
        stem = needle.lower()[:5]
        self.open_threads = [
            t for t in self.open_threads
            if stem not in t.lower()[: len(t) + 5] and stem not in " ".join(
                w[:5] for w in t.lower().split()
            )
        ]

    def note(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self.durable_notes:
            self.durable_notes.append(fact)
            self.durable_notes = self.durable_notes[-self.MAX_NOTES:]

    def to_dict(self) -> dict:
        def _val(x):
            return x.value if hasattr(x, "value") else x

        return {
            "session_id": self.session_id,
            "mode": _val(self.mode),
            "phase": _val(self.phase),
            "intent": self.intent,
            "emotion": self.emotion,
            "risk": _val(self.risk),
            "user_goal": _val(self.user_goal),
            "requires_rag": self.requires_rag,
            "requires_followup": self.requires_followup,
            "turn_count": self.turn_count,
            "open_threads": self.open_threads,
        }
