from __future__ import annotations

from agent.state.manager import StateMachine, StateManager
from agent.state.models import ConversationState, Mode, Phase, Route


class TestStateMachine:
    def test_phase_order(self):
        sm = StateMachine(ConversationState(session_id="s"))
        sm.understand()
        assert sm.state.phase is Phase.UNDERSTAND
        sm.route(Route.COMPANION)
        assert sm.state.phase is Phase.ROUTE
        assert sm.last_route is Route.COMPANION
        sm.respond()
        sm.follow_up()
        sm.continue_()
        assert sm.state.phase is Phase.CONTINUE

    def test_no_backward_transition(self):
        sm = StateMachine(ConversationState(session_id="s"))
        sm.understand()
        sm.state.advance(Phase.IDLE)  # backward — must be ignored
        assert sm.state.phase is Phase.UNDERSTAND

    def test_recent_context_window(self):
        sm = StateMachine(ConversationState(session_id="s"), recent_window=3)
        for i in range(10):
            sm.add_turn("user", f"m{i}")
        assert len(sm.recent) == 3
        assert sm.recent[-1].text == "m9"

    def test_turn_count(self):
        sm = StateMachine(ConversationState(session_id="s"))
        sm.add_turn("user", "a")
        sm.add_turn("assistant", "b")
        assert sm.state.turn_count == 2


class TestStateManager:
    def test_ttl_expiry(self):
        sm = StateManager(ttl_seconds=0)
        sm.machine("x")
        assert sm.machine("x", create=False) is None

    def test_drop(self):
        sm = StateManager()
        sm.machine("x")
        sm.drop("x")
        assert sm.active_count() == 0

    def test_state_is_structured(self):
        """§2: enums/values, no free-form where a structure suffices."""
        sm = StateManager()
        st = sm.machine("s").state
        st.mode = Mode.COMPANION
        st.emotion = "loneliness"
        st.risk = "low"
        d = st.to_dict()
        assert d["mode"] == "companion" and isinstance(d["risk"], str)


class TestTopicContinuity:
    def test_open_threads_capped(self):
        sm = StateMachine(ConversationState(session_id="s"))
        for i in range(10):
            sm.state.note_thread(f"thread {i}")
        assert len(sm.state.open_threads) == ConversationState.MAX_OPEN_THREADS

    def test_close_threads_matching(self):
        st = ConversationState(session_id="s")
        st.note_thread("loneliness discussion")
        st.note_thread("studying Al-Kahf")
        st.close_threads_matching("lonely")
        assert [t for t in st.open_threads if "loneliness" in t] == []
        assert any("Al-Kahf" in t for t in st.open_threads)
