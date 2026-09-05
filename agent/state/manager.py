"""State machine + manager (fixme_v2 §4, §13).

RECENT CONTEXT (last N turns) / SESSION STATE / LONG-TERM MEMORY are kept
separate (§13). The manager owns sessions with TTL; recent turns live here
per-session, long-term memory lives in the memory router (never mixed).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent.state.models import ConversationState, Phase, Route


@dataclass
class Turn:
    role: str
    text: str
    ts: float = field(default_factory=time.time)


class StateMachine:
    """fixme_v2 §4: IDLE -> UNDERSTAND -> ROUTE -> RESPOND -> FOLLOW_UP ->
    CONTINUE, with special routes out of ROUTE (SAFETY/RAG/MEMORY/COMPANION)."""

    def __init__(self, state: ConversationState, recent_window: int = 6):
        self.state = state
        self.recent: list[Turn] = []
        self.recent_window = recent_window
        self.last_route: Route | None = None

    def understand(self) -> None:
        self.state.advance(Phase.UNDERSTAND)

    def route(self, route: Route) -> Route:
        self.state.advance(Phase.ROUTE)
        self.last_route = route
        return route

    def respond(self) -> None:
        self.state.advance(Phase.RESPOND)

    def follow_up(self) -> None:
        self.state.advance(Phase.FOLLOW_UP)

    def continue_(self) -> None:
        self.state.advance(Phase.CONTINUE)

    # §13 recent context: bounded, separate from long-term memory
    def add_turn(self, role: str, text: str) -> None:
        self.recent.append(Turn(role, text))
        if len(self.recent) > self.recent_window:
            self.recent = self.recent[-self.recent_window :]
        self.state.turn_count += 1
        self.state.last_active = time.time()

    def recent_context(self, n: int | None = None) -> list[Turn]:
        if n is None or n >= len(self.recent):
            return list(self.recent)
        return self.recent[-n:]

    def recent_as_text(self, n: int = 4) -> str:
        lines = [f"{t.role}: {t.text}" for t in self.recent_context(n)]
        return "\n".join(lines[-n:])


class StateManager:
    """Session registry with TTL — state is never silently persisted."""

    TTL_SECONDS = 2 * 60 * 60

    def __init__(self, ttl_seconds: int = TTL_SECONDS):
        self.ttl = ttl_seconds
        self._machines: dict[str, StateMachine] = {}

    def machine(self, session_id: str, create: bool = True) -> StateMachine | None:
        sm = self._machines.get(session_id)
        if sm and (time.time() - sm.state.last_active) > self.ttl:
            del self._machines[session_id]
            sm = None
        if sm is None and create:
            sm = StateMachine(ConversationState(session_id=session_id))
            self._machines[session_id] = sm
        return sm

    def drop(self, session_id: str) -> None:
        self._machines.pop(session_id, None)

    def active_count(self) -> int:
        return len(self._machines)
