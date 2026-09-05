from __future__ import annotations

import pytest

from agent.context.builder import ContextBudget, ContextBuilder, context_to_prompt
from agent.policy.companion_policy import ResponsePolicy
from agent.state.manager import StateMachine
from agent.state.models import ConversationState, Mode


@pytest.fixture()
def machine():
    return StateMachine(ConversationState(session_id="s"))


class TestContextBuilder:
    def test_pack_structure(self, machine):
        machine.state.mode = Mode.COMPANION
        machine.state.intent = "emotional_support"
        machine.state.emotion = "loneliness"
        policy = ResponsePolicy(mode=Mode.COMPANION)
        pack = ContextBuilder().build(machine, policy,
                                      memory_hits=[{"fact": "User studies Al-Kahf", "category": "study"}],
                                      evidence=[{"citation_id": "quran:2:186"}])
        d = pack.to_dict()
        assert d["mode"] == "companion"
        assert d["emotion"] == "loneliness"
        assert len(d["relevant_memory"]) == 1
        assert d["policy"]["max_followups"] == 1

    def test_budget_limits_recent_turns(self, machine):
        for i in range(12):
            machine.add_turn("user", f"turn {i}")
        policy = ResponsePolicy()
        pack = ContextBuilder(ContextBudget(recent_turns=3)).build(machine, policy)
        assert len(pack.recent_context) == 3
        assert pack.recent_context[-1]["text"] == "turn 11"

    def test_budget_limits_memory(self, machine):
        policy = ResponsePolicy()
        pack = ContextBuilder(ContextBudget(memory_items=2)).build(
            machine, policy,
            memory_hits=[{"fact": f"fact {i}"} for i in range(10)],
        )
        assert len(pack.relevant_memory) == 2

    def test_budget_limits_evidence(self, machine):
        policy = ResponsePolicy(evidence_limit=4)
        pack = ContextBuilder(ContextBudget(evidence_items=3)).build(
            machine, policy,
            evidence=[{"citation_id": f"c{i}"} for i in range(10)],
        )
        assert len(pack.evidence) == 3

    def test_no_memory_injected_by_default(self, machine):
        """§12: never the whole profile into every prompt."""
        policy = ResponsePolicy()
        pack = ContextBuilder().build(machine, policy)
        assert pack.relevant_memory == []


class TestPromptRendering:
    def test_companion_instructions(self, machine):
        machine.state.mode = Mode.COMPANION
        machine.state.emotion = "loneliness"
        policy = ResponsePolicy(mode=Mode.COMPANION, acknowledge_first=True,
                                 max_followups=1, word_budget=90)
        pack = ContextBuilder().build(machine, policy)
        prompt = context_to_prompt(pack)
        assert "Acknowledge" in prompt
        assert "at most 1 gentle follow-up" in prompt
        assert "under 90 words" in prompt

    def test_no_question_instruction(self, machine):
        policy = ResponsePolicy(max_followups=0)
        pack = ContextBuilder().build(machine, policy)
        assert "Do not end with a question" in context_to_prompt(pack)

    def test_religion_hold_instruction(self, machine):
        policy = ResponsePolicy(allow_islamic_reflection=False)
        pack = ContextBuilder().build(machine, policy)
        assert "Do NOT bring up religious content" in context_to_prompt(pack)

    def test_offer_not_dump_instruction(self, machine):
        policy = ResponsePolicy(allow_islamic_reflection=True, requires_rag=False)
        pack = ContextBuilder().build(machine, policy)
        assert "BRIEFLY mention" in context_to_prompt(pack)
