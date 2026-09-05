"""Companion policy engine (fixme_v2 §5-6, §16-18).

The policy engine decides — machine-readably — what the response should be
BEFORE the model is invoked:

  retrieve? use memory? ask a question? how many? Islamic guidance?
  evidence volume? tone? verbosity? safety mode?

The LLM generates language; these decisions belong to the harness (§1).
Safety policy ALWAYS overrides companion policy (§19).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agent.state.models import ConversationState, Mode, Risk, UserGoal


class Tone(StrEnum):
    WARM = "warm"
    NEUTRAL = "neutral"
    CALM = "calm"


class Verbosity(StrEnum):
    SHORT = "short"      # companion default (§7)
    MEDIUM = "medium"    # qa answers
    DETAILED = "detailed"  # study / explicit requests for depth


@dataclass
class ResponsePolicy:
    """fixme_v2 §6: the structured, machine-readable policy object."""

    mode: Mode = Mode.COMPANION
    route: str = "companion"          # safety | rag | memory | companion | chat
    tone: Tone = Tone.WARM
    verbosity: Verbosity = Verbosity.SHORT
    requires_rag: bool = False
    requires_memory: bool = True
    requires_followup: bool = True
    max_followups: int = 1
    allow_islamic_reflection: bool = True  # offering is allowed; dumping is not
    preach: bool = False
    evidence_limit: int = 4
    acknowledge_first: bool = True
    safety_override: bool = False
    word_budget: int = 90
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value, "route": self.route,
            "tone": self.tone.value, "verbosity": self.verbosity.value,
            "requires_rag": self.requires_rag,
            "requires_memory": self.requires_memory,
            "requires_followup": self.requires_followup,
            "max_followups": self.max_followups,
            "allow_islamic_reflection": self.allow_islamic_reflection,
            "preach": self.preach,
            "evidence_limit": self.evidence_limit,
            "acknowledge_first": self.acknowledge_first,
            "safety_override": self.safety_override,
            "word_budget": self.word_budget,
        }


# intent classes the policy engine reasons over (produced by the classifier)
RAG_INTENTS = {
    "quran_question", "quran_request", "hadith_question", "islamic_question",
    "fiqh_question", "dua_request",
}
EMOTIONAL_INTENTS = {
    "emotional_support", "loneliness", "grief", "anxiety", "anger",
    "guilt", "fear", "confusion", "spiritual_low", "motivation",
    "gratitude", "relationship_problem", "life_problem",
}
STUDY_INTENTS = {"study_note", "history"}


class CompanionPolicyEngine:
    """fixme_v2 §5: routing + response policy. Pure functions of the state;
    no model involved."""

    def decide(self, state: ConversationState, *, explicit_islamic: bool = False,
               turn_is_question: bool = False, memory_preferred: str = "unknown",
               turn_count: int = 0) -> ResponsePolicy:
        # §19: safety overrides everything — companion policy cannot soften it
        if state.risk is Risk.HIGH:
            return ResponsePolicy(
                mode=Mode.CRISIS, route="safety",
                tone=Tone.CALM, verbosity=Verbosity.SHORT,
                requires_rag=False, requires_memory=False,
                requires_followup=False, max_followups=0,
                allow_islamic_reflection=False, preach=False,
                acknowledge_first=True, safety_override=True, word_budget=140,
                notes=["safety override: crisis canned response"],
            )

        policy = ResponsePolicy(mode=state.mode)

        intent = state.intent
        emotion = state.emotion

        if state.risk is Risk.ELEVATED:
            # still companionship, but gentler, no follow-up pressure
            policy.mode = Mode.COMPANION
            policy.route = "companion"
            policy.tone = Tone.CALM
            policy.verbosity = Verbosity.SHORT
            policy.requires_rag = False
            policy.requires_followup = True
            policy.max_followups = 1
            policy.allow_islamic_reflection = False  # no preaching at elevated risk
            policy.word_budget = 80
            policy.notes.append("elevated risk: calmer tone, no religious push")
            return policy

        # §16 RAG routing decision
        wants_rag = intent in RAG_INTENTS or explicit_islamic
        if emotion and wants_rag:
            # "What does Islam say about loneliness?" — empathize, THEN RAG
            policy.route = "rag"
            policy.requires_rag = True
            policy.mode = Mode.QA if turn_is_question else Mode.COMPANION
            policy.acknowledge_first = True
            policy.verbosity = Verbosity.MEDIUM
            policy.word_budget = 160
            policy.evidence_limit = 4
            policy.requires_followup = False
            policy.notes.append("emotional islamic question: empathy then evidence")
        elif wants_rag:
            policy.route = "rag"
            policy.requires_rag = True
            policy.mode = Mode.QA
            policy.verbosity = Verbosity.MEDIUM
            policy.word_budget = 200
            policy.requires_followup = False
            if intent == "dua_request":
                policy.mode = Mode.DUA
                policy.tone = Tone.CALM
        elif intent in STUDY_INTENTS or state.user_goal is UserGoal.STUDY:
            policy.route = "memory"
            policy.mode = Mode.STUDY
            policy.requires_memory = True
            policy.verbosity = Verbosity.MEDIUM
        elif emotion or intent in EMOTIONAL_INTENTS:
            # §7/§17: pure emotion — companion first; guidance only offered
            policy.route = "companion"
            policy.mode = Mode.COMPANION
            policy.requires_rag = False
            policy.requires_followup = state.requires_followup
            policy.max_followups = 1
            policy.allow_islamic_reflection = True
            policy.preach = False
            policy.verbosity = Verbosity.SHORT
            policy.word_budget = 90
            # late in an emotional thread, prefer continuation over re-asking
            if turn_count >= 3:
                policy.requires_followup = False
                policy.notes.append("long emotional thread: listen, don't re-ask")
        else:
            policy.route = "chat"
            policy.mode = Mode.QA
            policy.tone = Tone.NEUTRAL
            policy.requires_rag = False
            policy.requires_followup = False
            policy.verbosity = Verbosity.SHORT
            policy.word_budget = 60
            policy.acknowledge_first = False

        # guidance preference learned from user behavior (§17)
        if memory_preferred == "hold":
            policy.allow_islamic_reflection = False
            policy.preach = False
        return policy


class PolicyValidator:
    """fixme_v2 §25: policy compliance checks — a response can be factually
    fine yet fail companion policy."""

    def validate(self, response_text: str, policy: ResponsePolicy) -> list[str]:
        import re

        problems: list[str] = []
        questions = len(re.findall(r"\?", response_text))
        if questions > policy.max_followups:
            problems.append(
                f"too many questions: {questions} > max_followups {policy.max_followups}"
            )
        words = len(response_text.split())
        if words > policy.word_budget * 1.5:
            problems.append(f"verbosity: {words} words exceeds budget {policy.word_budget}")
        if policy.preach is False and policy.mode.value == "companion":
            first = response_text.strip().split("\n")[0].lower()
            for opener in ("allah says", "the quran says", "quran says", "the prophet said"):
                if first.startswith(opener):
                    problems.append(f"preachy opener: {opener!r}")
                    break
        if policy.safety_override:
            lowered = response_text.lower()
            for banned in ("haram", "sinful", "allah will punish", "hellfire"):
                if banned in lowered:
                    problems.append(f"religious guilt in safety mode: {banned!r}")
        return problems
