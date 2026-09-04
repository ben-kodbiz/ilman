"""Companion Harness (fixme_v2 §1, §47).

The full pipeline around the EXISTING knowledge architecture (which is not
rewritten — §0 'incremental'):

  UNDERSTAND (classifier) -> STATE ENGINE -> SAFETY GATE -> POLICY ENGINE
    -> MEMORY ROUTER -> RAG ROUTER -> CONTEXT BUILDER -> local model
    -> VALIDATION (religious + companion) -> USER

The harness owns state, policy, memory, context, safety and validation
decisions. The model only generates language through the task-class routing
that already exists (§23/§24 — no coupling to Ling/Gemma).
"""

from __future__ import annotations

import re
import time as _t
from dataclasses import dataclass, field
from typing import Any

from agent.companion.intent import classify_companion
from agent.context.builder import ContextBuilder, context_to_prompt
from agent.core.model import ChatMessage
from agent.core.observability import DebugTrace
from agent.memory.router import MemoryRouter
from agent.policy.companion_policy import CompanionPolicyEngine, ResponsePolicy
from agent.safety.router import canned_safety_response, safety_route
from agent.state.manager import StateManager
from agent.state.models import Mode, Route, UserGoal
from agent.validators.companion_validator import ResponseValidator
from agent.validators.pipeline import CitationValidator, EvidencePack

COMPANION_SYSTEM_PROMPT = (
    "You are Ilman, a warm, calm and humble Islamic companion. You are an AI — "
    "never pretend to be human, never simulate a personal relationship, never "
    "encourage reliance on you over real people. You are not a therapist, "
    "doctor or mufti: no diagnosis, no rulings.\n\n"
    "Follow the CONTEXT instructions exactly: they decide tone, length, "
    "whether to acknowledge feelings first, whether Islamic content may "
    "appear, and how many questions you may ask.\n\n"
    "If an <evidence> block is present, any religious statement MUST quote "
    "from it and cite as [quran:surah:ayah] / [hadith:collection:number]; "
    "never invent Qur'an, hadith, gradings or scholars. If no evidence block "
    "is present, make NO religious claims at all — empathy and practical "
    "warmth only."
)

# emotion -> user-goal mapping (§2 user_goal field)
_EMOTION_GOAL = {
    "loneliness": UserGoal.BE_HEARD, "grief": UserGoal.BE_HEARD,
    "anxiety": UserGoal.BE_HEARD, "anger": UserGoal.BE_HEARD,
    "guilt": UserGoal.BE_HEARD, "fear": UserGoal.BE_HEARD,
    "confusion": UserGoal.ANSWER, "spiritual_low": UserGoal.REFLECT,
    "gratitude": UserGoal.REFLECT, "motivation": UserGoal.ANSWER,
}


@dataclass
class HarnessResult:
    answer: str
    mode: Mode
    policy: dict
    citations: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    companion_validation: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "mode": self.mode.value,
            "policy": self.policy,
            "citations": self.citations,
            "unsupported_citations": self.unsupported_citations,
            "companion_validation": self.companion_validation,
            "state": self.state,
            # dev trace: internal only, not for the public client (§33)
            "debug_trace": self.trace,
        }


class CompanionHarness:
    def __init__(
        self,
        router,                    # ModelRouter (§23 adapter, any backend)
        retrieval=None,            # RetrievalOrchestrator or None (tests)
        memory_router: MemoryRouter | None = None,
        states: StateManager | None = None,
        policy_engine: CompanionPolicyEngine | None = None,
        context_builder: ContextBuilder | None = None,
        validator: ResponseValidator | None = None,
        citation_validator: CitationValidator | None = None,
        model_label: str = "",
    ):
        self.router = router
        self.retrieval = retrieval
        self.memory_router = memory_router
        self.states = states or StateManager()
        self.policy_engine = policy_engine or CompanionPolicyEngine()
        self.context_builder = context_builder or ContextBuilder()
        self.validator = validator or ResponseValidator()
        self.citation_validator = citation_validator or CitationValidator()
        self.model_label = model_label

    # ----------------------------------------------------------------- turn
    def respond(self, session_id: str, message: str,
                max_tokens: int = 1200) -> HarnessResult:
        trace = DebugTrace(model=self.model_label)
        machine = self.states.machine(session_id)
        if machine is None:
            machine = self.states.machine(session_id, create=True)
        state = machine.state

        # 1. SAFETY GATE — independent, before everything (§19)
        safety = safety_route(message)
        state.risk = safety.risk
        trace.risk = safety.risk.value

        if not safety.model_allowed:
            state.mode = Mode.CRISIS
            machine.add_turn("user", message)
            text = canned_safety_response(self._lang(message))
            machine.add_turn("assistant", text)
            trace.mode = state.mode.value
            trace.route = "safety"
            crisis_policy = ResponsePolicy(
                mode=Mode.CRISIS, route="safety", safety_override=True,
                max_followups=0, word_budget=140,
            )
            companion_v = self.validator.validate(text, crisis_policy)
            trace.mark_validation(
                companion_v.ok,
                "; ".join(companion_v.policy_problems + companion_v.companion_problems),
            )
            trace.latency_s = _t.time() - trace.started_at
            result = self._result(text, state, crisis_policy, trace, [], [])
            result.companion_validation = companion_v.to_dict()
            return result

        # 2. UNDERSTAND (§1) — deterministic classifier, no model.
        # Emotion continuity (§27): a turn without a fresh emotion signal
        # retains the thread's emotion — emotional threads don't reset.
        machine.understand()
        ci = classify_companion(message)
        state.intent = ci.intent
        if ci.emotion:
            state.emotion = ci.emotion
        elif state.mode is Mode.COMPANION and ci.intent in (
            "emotional_support", "loneliness", "grief", "anxiety", "anger",
            "guilt", "fear", "confusion", "spiritual_low", "normal_chat",
        ):
            pass  # retain previous emotion within a companion thread
        else:
            state.emotion = None
        state.user_goal = _EMOTION_GOAL.get(state.emotion or "", UserGoal.UNSPECIFIED)
        if ci.intent in ("quran_question", "hadith_question", "islamic_question",
                         "fiqh_question", "quran_request", "dua_request"):
            state.user_goal = UserGoal.ANSWER
        state.requires_rag = ci.needs_islamic_guidance
        state.requires_followup = ci.needs_clarification and state.risk.value == "low"
        trace.intent, trace.emotion = ci.intent, ci.emotion

        # topic-switch continuity (§42): memory of threads, closed by classifier
        if ci.intent in ("hadith_question", "quran_question", "islamic_question"):
            state.close_threads_matching("lonely")
            state.close_threads_matching("feeling")
        if ci.emotion:
            state.note_thread(f"{ci.emotion} discussion")

        # 3. POLICY (§5-6)
        policy = self.policy_engine.decide(
            state,
            explicit_islamic=ci.islamic_requested and ci.needs_islamic_guidance,
            turn_is_question=ci.is_question,
            memory_preferred=(self._memory_pref(state)),
            turn_count=state.turn_count,
        )
        state.mode = policy.mode
        trace.mode = state.mode.value
        trace.route = policy.route
        trace.policy = policy.to_dict()
        route = Route(policy.route)
        machine.route(route)

        # 4. MEMORY ROUTER (§10-12): extract + lifecycle + relevant retrieval
        memory_hits: list[dict] = []
        if self.memory_router is not None and policy.requires_memory:
            incoming = self.memory_router.route_incoming(message)
            trace.memory_saved = len(incoming["saved"])
            for saved in incoming["saved"]:
                state.note(saved["fact"])
            memory_hits = self.memory_router.relevant(message, limit=3)
            trace.memory_hits = len(memory_hits)

        # 5. RAG ROUTER (§16-17): policy.decide set requires_rag
        pack: EvidencePack | None = None
        if policy.requires_rag and self.retrieval is not None:
            passages = self.retrieval.search(
                message, limit=policy.evidence_limit,
                concept_expansions=(ci.core.concept_expansions or None) if ci.core else None,
                semantic_only=bool(ci.emotion),
            )
            pack = EvidencePack(query=message, passages=passages)
            trace.rag_used = bool(passages)

        # 6. CONTEXT BUILDER (§14-15)
        machine.add_turn("user", message)
        cpack = self.context_builder.build(machine, policy, memory_hits=memory_hits,
                                           evidence=([p.citation_id for p in (pack.passages if pack else [])]))
        prompt_block = context_to_prompt(cpack)

        # 7. MODEL (§23: routed by config task class, harness stays model-free)
        machine.respond()
        system = COMPANION_SYSTEM_PROMPT
        parts = [system, prompt_block]
        if pack is not None and pack.passages:
            parts.append(f"<evidence>\n{pack.to_prompt_block()}\n</evidence>")
        messages = [
            ChatMessage(role="system", content="\n\n".join(parts)),
            ChatMessage(role="user", content=f"User says: {message}"),
        ]
        resp = self.router.chat(
            "simple_chat" if state.mode is Mode.COMPANION else "complex_rag",
            messages, max_tokens=max_tokens,
        )
        text = resp.content.strip() or (
            "I hear you. If you want to tell me more, I'm listening."
        )

        # 8. VALIDATION (§21-22 + §25): religious + companion, layered
        citations: list[str] = []
        unsupported: list[str] = []
        if pack is not None:
            v = self.citation_validator.validate(text, pack)
            citations = v.verified_citations
            unsupported = v.unsupported_citations
            if unsupported or v.misattributed_grades:
                text = self._clean_unsupported(text, unsupported)
                v2 = self.citation_validator.validate(text, pack)
                unsupported = v2.unsupported_citations
                if unsupported:
                    text = (text.split("\n")[0][:200]
                            + "\n\nI could not verify this from the approved source corpus.")
                    unsupported = []
        companion_v = self.validator.validate(text, policy)
        if not companion_v.ok:
            # deterministic repairs for the two worst classes
            text = self._repair_dependency(text)
            companion_v = self.validator.validate(text, policy)

        # 9. bookkeeping + follow-up phase
        if "?" in text:
            machine.follow_up()
        else:
            machine.continue_()
        machine.add_turn("assistant", text)
        trace.mark_validation(
            companion_v.ok and not unsupported,
            "; ".join(companion_v.policy_problems + companion_v.companion_problems),
        )
        trace.latency_s = _t.time() - trace.started_at
        result = self._result(text, state, policy, trace, citations, unsupported)
        result.companion_validation = companion_v.to_dict()
        return result

    # ------------------------------------------------------------- internals
    def _result(self, text, state, policy, trace, citations, unsupported) -> HarnessResult:
        return HarnessResult(
            answer=text, mode=state.mode, policy=policy.to_dict(),
            citations=citations, unsupported_citations=unsupported,
            trace=trace.to_dict(), state=state.to_dict(),
        )

    @staticmethod
    def _memory_pref(state) -> str:
        # guidance preference could live in long-term profile memory; default unknown
        return "unknown"

    @staticmethod
    def _clean_unsupported(text: str, unsupported: list[str]) -> str:
        for citation in unsupported:
            pattern = re.compile(
                r"[^.!?\n]*" + re.escape(citation) + r"[^.!?\n]*[.!?]?", re.IGNORECASE
            )
            text = pattern.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _repair_dependency(text: str) -> str:
        from agent.validators.companion_validator import DEPENDENCY_RE

        return DEPENDENCY_RE.sub("I'm here to listen", text)

    @staticmethod
    def _lang(message: str) -> str:
        return "ms" if re.search(
            r"\b(saya|aku|tak|nak|dengan|yang)\b", message, re.IGNORECASE
        ) else "en"
