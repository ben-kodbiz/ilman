"""Companion engine (fix_me.md §1, §2, §7, §8, §10, §12, §14, §15).

Three-layer separation (§1):
  Companion layer  — emotion + state + planning (this module)
  Islamic layer    — RAG evidence via the EXISTING retrieval (unchanged rules)
  Knowledge layer  — existing validators (citations, provenance, §22)

Behavioral rules enforced here:
- Empathy FIRST; Islamic evidence only when the router decides it helps (§10)
- ONE follow-up question max, only when clarification genuinely helps (§15)
- Response-sections split: empathy needs no citation; ANY religious claim
  still passes the deterministic CitationValidator (§12, §22)
- Forbidden dependency/relationship-simulation language list-checked (§7, §8)
- Crisis severity short-circuits to the canned safety response, model-free
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.companion.intent import CompanionIntent, classify_companion
from agent.companion.safety import (
    Severity,
    safety_response,
)
from agent.companion.state import Mode, Phase, StateManager
from agent.core.model import ChatMessage
from agent.validators.pipeline import CitationValidator, EvidencePack

FORBIDDEN_PHRASES = [
    "i know exactly how you feel",
    "i am all you need",
    "you only need me",
    "i'm always here instead of",
    "always here instead of other people",
    "i understand you better than anyone",
    "you don't need anyone else",
    "no one else will understand",
]

DEPENDENCY_ENCOURAGEMENT = re.compile(
    r"\b(talk to me instead of|rather than (your|talking to) (friends|family))\b",
    re.IGNORECASE,
)

COMPANION_SYSTEM_PROMPT = (
    "You are Ilman, a warm, calm, and humble Islamic companion. You are NOT a "
    "therapist, doctor, or mufti — never diagnose, never issue rulings.\n\n"
    "HOW YOU SPEAK:\n"
    "- Natural and concise. Short paragraphs. No lectures, no lists of advice "
    "unless asked.\n"
    "- Acknowledge the person's feeling plainly before anything else. Never "
    "judge, never minimize ('at least...'), never preach.\n"
    "- You are an AI companion; do not pretend to be human, do not simulate a "
    "personal relationship, and never encourage the person to depend on you "
    "instead of real people. Where natural, gently encourage real human "
    "connection.\n"
    "- Do not start with 'Allah says' by default. Bring Islamic guidance only "
    "when the context below marks it as offered/welcome, and present it gently.\n\n"
    "RESPONSE SHAPE (use exactly these parts, in order, when comforting):\n"
    "1. one or two sentences acknowledging the feeling\n"
    "2. at most ONE gentle follow-up question — only if it helps; it is fine "
    "to end without a question. NEVER ask two questions. Count your "
    "question marks before finishing: there must be at most one '?'\n"
    "3. if Islamic guidance is marked welcome: a brief, gentle offering "
    "(e.g. 'there is a verse about Allah's nearness that some find comforting')\n"
    "4. optionally one small practical next step\n"
    "Keep the WHOLE reply under 90 words in companion mode. Never use "
    "'Would you like me to...' more than once.\n\n"
    "IF EVIDENCE IS PROVIDED below <evidence>, any religious statement MUST "
    "quote from it and cite as shown; NEVER invent Qur'an, hadith, or "
    "gradings. If evidence is absent, you may still empathize — but make NO "
    "religious claims at all."
)

QUESTION_RE = re.compile(r"[?]")


@dataclass
class CompanionResponse:
    text: str
    mode: Mode
    intent: CompanionIntent
    sections: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    religious_claims_valid: bool = True
    followup_questions: int = 0
    used_evidence: bool = False
    state_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "mode": self.mode.value,
            "intent": self.intent.to_dict(),
            "citations": self.citations,
            "unsupported_citations": self.unsupported_citations,
            "followup_questions": self.followup_questions,
            "used_evidence": self.used_evidence,
            "conversation_state": self.state_snapshot,
        }


class CompanionEngine:
    def __init__(self, router, retrieval, tools=None, memory=None,
                 state_manager: StateManager | None = None):
        self.router = router
        self.retrieval = retrieval
        self.tools = tools
        self.memory = memory
        self.states = state_manager or StateManager()
        self.validator = CitationValidator()

    # ------------------------------------------------------------------ api
    def respond(self, session_id: str, message: str,
                max_tokens: int = 1500) -> CompanionResponse:
        state = self.states.get(session_id)
        intent = classify_companion(message)

        # §9 crisis short-circuit — model-free, deterministic, safe
        if intent.severity is Severity.HIGH_RISK:
            state.mode = Mode.CRISIS
            state.phase = Phase.RESPOND
            state.add_turn("user", message)
            text = safety_response(self._detect_lang(message, state))
            state.add_turn("assistant", text)
            return CompanionResponse(
                text=text, mode=Mode.CRISIS, intent=intent,
                state_snapshot=state.to_dict(),
            )

        state.phase = Phase.UNDERSTAND
        state.add_turn("user", message)
        if intent.emotion:
            state.set_emotion(intent.emotion, intent.emotion_confidence)
            if intent.needs_clarification:
                state.note_open_thread(f"What lies behind the {intent.emotion}")

        # mode selection (§13)
        state.mode = self._select_mode(intent, state)

        # §6: compact context — relevant memories only
        memory_block = ""
        if self.memory is not None and self.memory.memory_enabled:
            facts = self.memory.relevant_facts(message, limit=2)
            if facts:
                memory_block = "Known about this person (their words, saved with consent):\n" + "\n".join(
                    f"- {f['fact']}" for f in facts
                )

        # §10: does this turn need Islamic evidence?
        guidance = self._guidance_decision(intent, state)
        pack = None
        if guidance and self.retrieval is not None:
            passages = self.retrieval.search(
                message, limit=4,
                concept_expansions=(intent.core.concept_expansions or None)
                if intent.core else None,
                semantic_only=bool(intent.emotion),
            )
            pack = EvidencePack(query=message, passages=passages)

        state.phase = Phase.RESPOND
        text = self._generate(message, intent, state, memory_block, pack, max_tokens)

        # §12/§22: any religious claim must validate against evidence
        citations: list[str] = []
        unsupported: list[str] = []
        if pack is not None:
            validation = self.validator.validate(text, pack)
            citations = validation.verified_citations
            unsupported = validation.unsupported_citations
            if unsupported:
                text = self._strip_unsupported(text, unsupported)
                # re-validate; if the model keeps fabricating, fall to notice
                residual = self.validator.validate(text, pack).unsupported_citations
                if residual:
                    empathic_prefix = text.split("\n")[0][:200]
                    text = (
                        f"{empathic_prefix}\n\nI could not verify this from the "
                        "approved source corpus."
                    )
                    unsupported = self.validator.validate(text, pack).unsupported_citations

        # §7/§8: dependency-simulation guard (deterministic post-check)
        text = self._dependency_guard(text)

        state.phase = Phase.FOLLOW_UP if QUESTION_RE.search(text) else Phase.CONTINUE
        state.add_turn("assistant", text)
        return CompanionResponse(
            text=text, mode=state.mode, intent=intent,
            citations=citations, unsupported_citations=unsupported,
            used_evidence=pack is not None and bool(pack.passages),
            followup_questions=len(QUESTION_RE.findall(text)),
            state_snapshot=state.to_dict(),
        )

    # ------------------------------------------------------------- internals
    def _select_mode(self, intent: CompanionIntent, state) -> Mode:
        if intent.intent == "crisis_signal":
            return Mode.CRISIS
        if intent.intent in ("quran_request", "quran_question"):
            return Mode.QA if intent.is_question else Mode.REFLECTION
        if intent.intent in ("hadith_question", "islamic_question", "fiqh_question"):
            return Mode.QA
        if intent.intent == "dua_request":
            return Mode.DUA
        if intent.intent == "reflection_request":
            return Mode.REFLECTION
        if intent.emotion or intent.intent == "emotional_support":
            return Mode.COMPANION
        if state.mode is Mode.COMPANION and not intent.is_question and intent.first_person:
            return Mode.COMPANION  # stay in companion across a support thread
        if intent.core and intent.core.intent in ("study_note", "history"):
            return Mode.STUDY
        return Mode.QA

    def _guidance_decision(self, intent: CompanionIntent, state) -> bool:
        """§10 router: explicit islamic ask -> RAG now. Emotional statement ->
        offer only; guidance preference can flip the default."""
        if state.religious_guidance_preference == "welcome":
            return True
        if state.religious_guidance_preference == "hold":
            return intent.needs_islamic_guidance and not intent.emotion
        return intent.needs_islamic_guidance

    def _generate(self, message: str, intent: CompanionIntent, state,
                  memory_block: str, pack: EvidencePack | None,
                  max_tokens: int = 1500) -> str:
        parts = [COMPANION_SYSTEM_PROMPT]
        ctx = state.to_prompt_context()
        if ctx:
            parts.append(f"CONVERSATION CONTEXT:\n{ctx}")
        if memory_block:
            parts.append(f"MEMORY:\n{memory_block}")
        guidance_line = (
            "GUIDANCE: the person has not asked for religious content. Respond "
            "with empathy first. You may BRIEFLY offer that Islamic perspective "
            "exists ('there's a verse about...') without quoting or citing, "
            "unless they ask."
            if (pack is None or not pack.passages)
            else "GUIDANCE: Islamic guidance is welcome in this reply. Quote the "
                 "evidence gently, cite as shown, keep the empathy first."
        )
        parts.append(guidance_line)
        if pack is not None and pack.passages:
            parts.append(f"<evidence>\n{pack.to_prompt_block()}\n</evidence>")
        user_content = f"User says: {message}\n\nRespond as Ilman."
        messages = [
            ChatMessage(role="system", content="\n\n".join(parts)),
            ChatMessage(role="user", content=user_content),
        ]
        resp = self.router.chat("simple_chat", messages, max_tokens=max_tokens)
        if not resp.content.strip():
            return (
                "That sounds heavy. I'm here — if you want to tell me more "
                "about what's going on, I'm listening."
            )
        return resp.content.strip()

    def _strip_unsupported(self, text: str, unsupported: list[str]) -> str:
        """§22: remove SENTENCES carrying unsupported religious citations —
        keeping the citation marker but dropping the claim is not enough."""
        for citation in unsupported:
            # remove any sentence containing the citation (in any bracket form)
            pattern = re.compile(
                r"[^.!?\n]*" + re.escape(citation) + r"[^.!?\n]*[.!?]?",
                re.IGNORECASE,
            )
            text = pattern.sub("", text)
            # also catch bare marker leftovers
            text = text.replace(f"[{citation}]", "").replace(f"({citation})", "")
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _dependency_guard(self, text: str) -> str:
        """§7/§8: rewrite forbidden dependency-simulation phrasing."""
        lowered = text.lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in lowered:
                text = re.sub(re.escape(phrase), "I'm here to listen", text, flags=re.IGNORECASE)
        if DEPENDENCY_ENCOURAGEMENT.search(text):
            text = DEPENDENCY_ENCOURAGEMENT.sub(
                "it may also help to talk with someone you trust", text
            )
        return text

    @staticmethod
    def _detect_lang(message: str, state) -> str:
        ms = re.search(r"\b(saya|aku|tak|nak|dengan|yang|untuk)\b", message, re.IGNORECASE)
        return "ms" if ms else "en"
