from __future__ import annotations

import pytest

from agent.companion.memory import CompanionMemory, FactRejected
from agent.companion.state import Mode, Phase, StateManager


@pytest.fixture()
def memory(tmp_path):
    return CompanionMemory(db_path=tmp_path / "mem.db")


class TestFactPolicy:
    """fix_me.md §5C: only useful, stable, explicitly-shared facts."""

    def test_transient_emotion_rejected(self, memory):
        with pytest.raises(FactRejected):
            memory.save_fact("User feels lonely today")

    def test_feeling_statement_rejected(self, memory):
        with pytest.raises(FactRejected):
            memory.save_fact("I feel sad")

    def test_stable_fact_saved(self, memory):
        fid = memory.save_fact("User's name is Adam")
        assert fid > 0
        assert any("Adam" in f["fact"] for f in memory.facts())

    def test_explicit_save_allows_emotional_context(self, memory):
        # user EXPLICITLY asked to remember — §5C allows it
        fid = memory.save_fact("User is grieving his father's death",
                              explicit=True, category="context")
        assert fid > 0

    def test_empty_rejected(self, memory):
        with pytest.raises(FactRejected):
            memory.save_fact("   ")

    def test_memory_disabled_rejects(self, tmp_path):
        mem = CompanionMemory(db_path=tmp_path / "m2.db", memory_enabled=False)
        with pytest.raises(FactRejected):
            mem.save_fact("User lives in Kuala Lumpur")


class TestRelevantRetrieval:
    """fix_me.md §6: only memories relevant to the current message."""

    def test_relevant_only(self, memory):
        memory.save_fact("User is learning Surah Al-Baqarah")
        memory.save_fact("User works as a teacher")
        hits = memory.relevant_facts("Can you help me study Surah Al-Baqarah tonight?")
        assert hits and any("Al-Baqarah" in f["fact"] for f in hits)
        assert all("teacher" not in f["fact"] for f in hits)

    def test_no_overlap_no_hits(self, memory):
        memory.save_fact("User works as a teacher")
        assert memory.relevant_facts("what time is it") == []

    def test_never_more_than_limit(self, memory):
        for i in range(10):
            memory.save_fact(f"User likes topic number {i} about patience")
        assert len(memory.relevant_facts("tell me about topic patience", limit=3)) <= 3


class TestControls:
    """fix_me.md §25: view / forget / clear / disable."""

    def test_view(self, memory):
        memory.save_fact("User's name is Adam")
        view = memory.memory_view()
        assert "facts" in view and any("Adam" in f["fact"] for f in view["facts"])
        assert view["memory_enabled"] is True

    def test_forget(self, memory):
        fid = memory.save_fact("User's name is Adam")
        assert memory.forget_fact(fid)
        assert not any("Adam" in f["fact"] for f in memory.facts())

    def test_clear_all(self, memory):
        memory.save_fact("User's name is Adam")
        memory.save_note("a study note")
        cleared = memory.clear_all()
        assert cleared["facts"] >= 1 and cleared["notes"] >= 1
        assert memory.facts() == []

    def test_disable(self, memory):
        memory.set_memory_enabled(False)
        view = memory.memory_view()
        assert view["memory_enabled"] is False


class TestState:
    """fix_me.md §4/§13/§14: modes, phases, TTL, compression."""

    def test_mode_visible(self):
        sm = StateManager()
        state = sm.get("s")
        state.mode = Mode.COMPANION
        assert state.to_dict()["mode"] == "companion"

    def test_phase_advances(self):
        sm = StateManager()
        state = sm.get("s")
        assert state.phase is Phase.IDLE
        state.advance_phase()
        assert state.phase is Phase.UNDERSTAND
        state.advance_phase()
        assert state.phase is Phase.RESPOND

    def test_ttl_expiry(self):
        sm = StateManager(ttl_seconds=0)
        sm.get("gone")
        assert sm.get("gone", create=False) is None  # expired instantly

    def test_compact_context_not_turn_dump(self):
        sm = StateManager()
        state = sm.get("s")
        for i in range(30):
            state.add_turn("user", f"long message number {i} " + "x" * 100)
        ctx = state.to_prompt_context()
        assert len(ctx) < 2000  # compact, not the whole conversation

    def test_guidance_preference_learned(self):
        sm = StateManager()
        state = sm.get("s")
        state.mark_guidance(liked=True)
        assert state.religious_guidance_preference == "welcome"
        state.mark_guidance(liked=False)
        assert state.religious_guidance_preference == "hold"

    def test_drop_session(self):
        sm = StateManager()
        sm.get("s")
        sm.drop("s")
        assert sm.get("s", create=False) is None or sm.get("s", create=False).turns_in_mode == 0
