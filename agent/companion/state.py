"""Companion conversation state (fix_me.md §4, §13, §14, §16).

Short-lived, per-session state — explicitly NOT persistent memory (§4:
"The state is temporary and should not automatically become permanent
memory"). Holds: current emotion, conversation mode, phase in the companion
state machine, religious-guidance preference (learned from user behavior),
and a compact rolling context (§16 compression: durable facts + current
state + open thread, never raw turn dumps).

Mode transitions are visible internally (§13) and exposed to the UI.
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
    RESPOND = "respond"
    FOLLOW_UP = "follow_up"
    CONTINUE = "continue"


MODE_ANNOUNCEMENTS = {
    Mode.COMPANION: "companion",
    Mode.QA: "qa",
    Mode.STUDY: "study",
    Mode.REFLECTION: "reflection",
    Mode.DUA: "dua",
    Mode.CRISIS: "crisis",
}


@dataclass
class ConversationState:
    session_id: str
    current_emotion: str | None = None
    emotion_confidence: float = 0.0
    mode: Mode = Mode.QA
    phase: Phase = Phase.IDLE
    engagement_level: str = "open"  # open | guarded | withdrawn
    religious_guidance_preference: str = "unknown"  # unknown | welcome | hold
    # §16 compact context: replace raw turns with structured summary
    conversation_summary: str = ""
    durable_facts: list[str] = field(default_factory=list)
    open_threads: list[str] = field(default_factory=list)
    turns_in_mode: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    _turns: list[tuple[str, str]] = field(default_factory=list)  # (role, text) — capped

    MAX_TURNS = 8  # raw turns kept for prompt building; summary covers older

    def advance_phase(self) -> None:
        order = [Phase.IDLE, Phase.UNDERSTAND, Phase.RESPOND, Phase.FOLLOW_UP, Phase.CONTINUE]
        idx = order.index(self.phase)
        self.phase = order[min(idx + 1, len(order) - 1)]

    def set_emotion(self, emotion: str, confidence: float) -> None:
        if emotion and emotion != self.current_emotion:
            if self.open_threads or self.conversation_summary:
                self._roll_summary(emotion)
        self.current_emotion = emotion
        self.emotion_confidence = confidence

    def _roll_summary(self, new_emotion: str) -> None:
        """Very small deterministic compression (§16): keep durable facts +
        open threads; a real summarizer model can replace the topic line
        later behind this same method."""
        if self.current_emotion and self.turns_in_mode > 0 and not self.conversation_summary:
            self.conversation_summary = f"User has been talking about feeling {self.current_emotion}."

    def note_fact(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self.durable_facts:
            self.durable_facts.append(fact)
            self.durable_facts = self.durable_facts[-10:]

    def note_open_thread(self, thread: str) -> None:
        thread = thread.strip()
        if thread and thread not in self.open_threads:
            self.open_threads.append(thread)
            self.open_threads = self.open_threads[-5:]

    def close_thread(self, thread: str | None = None) -> None:
        if thread:
            self.open_threads = [t for t in self.open_threads if thread not in t]
        else:
            self.open_threads = []

    def mark_guidance(self, liked: bool) -> None:
        """Learn guidance preference from explicit user feedback or behavior
        (accepting an offered verse = welcome; deflecting = hold)."""
        self.religious_guidance_preference = "welcome" if liked else "hold"

    def add_turn(self, role: str, text: str) -> None:
        self._turns.append((role, text))
        if len(self._turns) > self.MAX_TURNS:
            self._turns = self._turns[-self.MAX_TURNS:]
        self.turns_in_mode += 1
        self.last_active = time.time()

    @property
    def recent_turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    def to_prompt_context(self) -> str:
        """Compact LLM context (§6/§16): structured state, never turn dumps
        beyond the last few turns."""
        parts: list[str] = []
        if self.conversation_summary:
            parts.append(f"Conversation so far: {self.conversation_summary}")
        if self.current_emotion:
            parts.append(
                f"Current emotional state: {self.current_emotion} "
                f"(confidence {self.emotion_confidence:.2f})"
            )
        if self.durable_facts:
            parts.append("Things shared: " + "; ".join(self.durable_facts[-4:]))
        if self.open_threads:
            parts.append("Not yet resolved: " + "; ".join(self.open_threads[-2:]))
        parts.append(f"Guidance preference: {self.religious_guidance_preference}")
        parts.append(f"Engagement: {self.engagement_level}")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "phase": self.phase.value,
            "current_emotion": self.current_emotion,
            "emotion_confidence": round(self.emotion_confidence, 2),
            "engagement_level": self.engagement_level,
            "religious_guidance_preference": self.religious_guidance_preference,
            "conversation_summary": self.conversation_summary,
            "open_threads": self.open_threads,
            "turns_in_mode": self.turns_in_mode,
        }


class StateManager:
    """Session registry with TTL: state expires after inactivity (§25: never
    store emotional state longer than the product needs it)."""

    TTL_SECONDS = 2 * 60 * 60  # 2h of inactivity -> discard

    def __init__(self, ttl_seconds: int = TTL_SECONDS):
        self.ttl = ttl_seconds
        self._sessions: dict[str, ConversationState] = {}

    def get(self, session_id: str, create: bool = True) -> ConversationState | None:
        state = self._sessions.get(session_id)
        if state and (time.time() - state.last_active) > self.ttl:
            del self._sessions[session_id]  # expired: never becomes memory
            state = None
        if state is None and create:
            state = ConversationState(session_id=session_id)
            self._sessions[session_id] = state
        return state

    def drop(self, session_id: str) -> None:
        """'Clear conversation' control (§25)."""
        self._sessions.pop(session_id, None)

    def active_count(self) -> int:
        return len(self._sessions)
