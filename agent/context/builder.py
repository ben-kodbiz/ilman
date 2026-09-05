"""Context builder (fixme_v2 §14-15).

Builds a controlled, budgeted ContextPack: only what the policy decided
reaches the model. Three context layers stay separate (§13):
recent turns / session state / long-term memory. Explicit budgets (§15):
recent context limited, memory top-relevant only, evidence only what was
retrieved, instructions compact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.policy.companion_policy import ResponsePolicy
from agent.state.manager import StateMachine


@dataclass
class ContextPack:
    """fixme_v2 §14: the exact object handed to prompt construction."""

    mode: str
    intent: str
    emotion: str | None
    risk: str
    recent_context: list[dict] = field(default_factory=list)
    relevant_memory: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    policy: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def estimated_tokens(self) -> int:
        def _tok(items: list[dict]) -> int:
            return sum(len(str(v).split()) for item in items for v in item.values()) * 2

        return _tok(self.recent_context) + _tok(self.relevant_memory) + _tok(self.evidence)


class ContextBudget:
    """fixme_v2 §15: explicit limits; a big context window is not an excuse."""

    def __init__(self, recent_turns: int = 4, memory_items: int = 3,
                 evidence_items: int = 4, instruction_words: int = 350):
        self.recent_turns = recent_turns
        self.memory_items = memory_items
        self.evidence_items = evidence_items
        self.instruction_words = instruction_words


class ContextBuilder:
    def __init__(self, budget: ContextBudget | None = None):
        self.budget = budget or ContextBudget()

    def build(
        self,
        machine: StateMachine,
        policy: ResponsePolicy,
        memory_hits: list[dict] | None = None,
        evidence: list[dict] | None = None,
    ) -> ContextPack:
        state = machine.state
        recent = [
            {"role": t.role, "text": t.text}
            for t in machine.recent_context(self.budget.recent_turns)
        ][-self.budget.recent_turns:]
        memory = (memory_hits or [])[: self.budget.memory_items]
        evid = (evidence or [])[: policy.evidence_limit][: self.budget.evidence_items]
        return ContextPack(
            mode=state.mode.value,
            intent=state.intent,
            emotion=state.emotion,
            risk=state.risk.value,
            recent_context=recent,
            relevant_memory=memory,
            evidence=evid,
            policy=policy.to_dict(),
        )


def context_to_prompt(pack: ContextPack) -> str:
    """Render the ContextPack into the compact model-facing instruction block
    (§15 'instructions: compact'). Evidence is rendered separately by the
    harness so citation formats stay intact."""
    lines: list[str] = []
    p = pack.policy
    lines.append(
        f"MODE: {pack.mode} | intent: {pack.intent} | "
        f"emotion: {pack.emotion or 'none'} | risk: {pack.risk}"
    )
    if p.get("acknowledge_first") and pack.emotion:
        lines.append(
            "Acknowledge the person's feeling first, before anything else."
        )
    if not p.get("preach") and not p.get("allow_islamic_reflection"):
        lines.append("Do NOT bring up religious content in this reply.")
    elif p.get("allow_islamic_reflection") and not p.get("requires_rag"):
        lines.append(
            "You may BRIEFLY mention that Islamic guidance exists on this "
            "(one sentence), but do not quote or cite it unless asked."
        )
    if p.get("max_followups", 0) <= 0:
        lines.append("Do not end with a question.")
    else:
        lines.append(
            f"Ask at most {p.get('max_followups')} gentle follow-up question "
            "— only if it genuinely helps."
        )
    lines.append(
        f"Target length: under {p.get('word_budget', 90)} words. Tone: {p.get('tone', 'warm')}."
    )
    if pack.recent_context:
        lines.append("RECENT CONVERSATION:")
        for t in pack.recent_context:
            lines.append(f"{t['role']}: {t['text'][:240]}")
    if pack.relevant_memory:
        lines.append("WHAT YOU KNOW ABOUT THIS PERSON (shared earlier):")
        for m in pack.relevant_memory:
            lines.append(f"- {m.get('fact', str(m))[:200]}")
    return "\n".join(lines)
