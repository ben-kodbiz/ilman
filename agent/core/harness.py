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
from agent.core.query_planner import plan_query
from agent.memory.router import MemoryRouter
from agent.policy.companion_policy import CompanionPolicyEngine, ResponsePolicy
from agent.safety.router import canned_safety_response, safety_route
from agent.state.manager import StateManager
from agent.state.models import Mode, Route, UserGoal
from agent.validators.companion_validator import ResponseValidator
from agent.validators.evidence_judge import (
    EvidenceJudge,
    Verdict,
    language_strength_ok,
)
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

        # 5. RAG ROUTER (§16-17) — fixme_v3 §6: plan the query first, then
        # retrieve on the planned information need (modern terms expanded to
        # classical concepts, requested object deciding evidence shape)
        pack: EvidencePack | None = None
        plan = plan_query(message, ci.intent)
        trace.planned_query = plan.to_dict()
        if policy.requires_rag and self.retrieval is not None:
            passages: list = []
            for term in plan.retrieval_terms:
                for p in self.retrieval.search(
                    term, limit=max(3, policy.evidence_limit // 2),
                    concept_expansions=(ci.core.concept_expansions or None) if ci.core else None,
                    semantic_only=bool(ci.emotion),
                ):
                    if p.citation_id not in {x.citation_id for x in passages}:
                        passages.append(p)
            # source preference ordering (§6): preferred source types first
            def _pref_rank(p):
                if plan.source_preference[0] == "hadith":
                    return (0 if p.citation_id.startswith("hadith:") else 1, -p.score)
                if plan.source_preference[0] == "quran":
                    return (0 if p.citation_id.startswith("quran:") else 1, -p.score)
                return (0, -p.score)
            passages.sort(key=_pref_rank)
            pack = EvidencePack(query=message, passages=passages[: policy.evidence_limit])

            # §12 EVIDENCE QUARANTINE: grade each passage against the query
            # BEFORE the LLM sees it; IRRELEVANT passages are removed so the
            # model cannot stitch them into claims.
            if pack.passages:
                pack.passages = self._quarantine_irrelevant(pack.passages, plan)
            trace.rag_used = bool(pack.passages)
            trace.evidence_status = "insufficient" if not pack.passages else "graded"

        # Dua/prayer requests get a targeted corpus search for ACTUAL
        # supplication texts (translations that begin with/repeat dua words:
        # "O Allah, ...", "I seek refuge in You from ..."). Plain retrieval
        # ranks verses ABOUT supplication above the duas themselves; a dua
        # request needs the duas. Deterministic: no invented text.
        if ci.intent == "dua_request" and self.retrieval is not None and self.retrieval.hadith_store is not None:
            dua_passages = self.retrieval.hadith_store.search_fts(
                "O Allah I seek refuge in You from anxiety and grief sorrow",
                limit=4,
            )
            if pack is None:
                pack = EvidencePack(query=message, passages=[])
            existing = {p.citation_id for p in pack.passages}
            for row in dua_passages:
                if row["citation_id"] in existing:
                    continue
                pack.passages.append(self._row_to_hadith_passage(row))
            if len(pack.passages) > policy.evidence_limit:
                # dua-text hits outrank generic matches for a dua request
                dua_first = [p for p in pack.passages if self._looks_like_dua_text(p)]
                rest = [p for p in pack.passages if p not in dua_first]
                pack.passages = (dua_first + rest)[: policy.evidence_limit]

        # Dua/prayer requests: if the retrieved evidence contains no actual
        # supplication text, do NOT improvise — acknowledge honestly and
        # offer to search for the specific dua (user-requested behavior).
        # Detection requires quoted supplication WORDS, not verses about
        # supplication (which false-matched 'Duha prayer' / 41:49).
        if ci.intent == "dua_request":
            has_dua_evidence = bool(
                pack is not None and pack.passages
                and any(self._looks_like_dua_text(p) for p in pack.passages)
            )
            if not has_dua_evidence:
                ack = "That sounds heavy to carry."
                offer = (
                    "I don't want to guess at words attributed to the Prophet ﷺ. "
                    "Would you like me to find a specific dua for easing this "
                    "— like the Prophet's ﷺ supplications for relief from grief "
                    "and anxiety?"
                )
                machine.add_turn("user", message)
                text = f"{ack} {offer}"
                machine.follow_up()
                machine.add_turn("assistant", text)
                state.note_thread("dua search offered")
                trace.rag_used = bool(pack is not None and pack.passages)
                trace.mark_validation(True, "dua weak-evidence: honest offer")
                trace.latency_s = _t.time() - trace.started_at
                result = self._result(text, state, policy, trace, [], [])
                result.companion_validation = {
                    "ok": True, "policy_problems": [],
                    "companion_problems": [], "uncited_religious_claims": [],
                }
                return result

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
            parts.append(
                "EVIDENCE IS PROVIDED: build the religious substance of your answer "
                "from it and cite as shown ([quran:surah:ayah], [hadith:collection:"
                "number]). Statements like 'The Quran says...' or 'The Prophet "
                "said...' MUST quote this evidence with a citation — an uncited "
                "religious quote will be removed."
            )
        else:
            parts.append(
                "NO EVIDENCE IS PROVIDED: make NO religious quotes, claims, or "
                "attributions — empathy and general warmth only."
            )
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

        # 8. VALIDATION (§21-22 + §25) + fixme_v3 CLAIM→EVIDENCE ENTAILMENT:
        # existence checks, then the judge decides what the answer is
        # ALLOWED to claim (§4-5, §9-13).
        citations: list[str] = []
        unsupported: list[str] = []
        if pack is not None:
            v = self.citation_validator.validate(text, pack)
            citations = v.verified_citations
            unsupported = v.unsupported_citations
            misquoted = v.misquoted_citations
            if unsupported or v.misattributed_grades or misquoted:
                # Strip ONLY the offending sentences; good cited content
                # survives for the judge to verify. The notice fallback no
                # longer fires here — one bad citation must not nuke an
                # answer whose other claims are entailed (that decision
                # belongs to the judge, not the existence checker).
                for mq in misquoted:
                    text = self._strip_citation_sentence(text, mq["citation"])
                text = self._clean_unsupported(text, unsupported)
                v2 = self.citation_validator.validate(text, pack)
                citations = v2.verified_citations or citations
                unsupported = v2.unsupported_citations

        # EVIDENCE JUDGE (fixme_v3 §4): claim→evidence entailment on every
        # claim sentence; verdicts feed the §16 language gate and the §13
        # sufficiency state; failures trigger the §8 careful-scope fallback.
        judgement = None
        if pack is not None and pack.passages and policy.requires_rag:
            judge = EvidenceJudge(embed=self._embed_fn())
            judgement = judge.judge_answer(
                text, pack, topic=plan.topic if plan else None,
                requested_object=plan.requested_object if plan else None,
            )
            trace.evidence_status = judgement.sufficiency.value
            trace.evidence_sufficiency = judgement.evidence_sufficiency
            # §16: strong language only on SUPPORTS claims
            lang_violations = language_strength_ok(text, judgement)
            unsupported_claims = [
                j for j in judgement.claim_support
                if j.verdict in (Verdict.IRRELEVANT, Verdict.UNKNOWN)
            ]
            if lang_violations or unsupported_claims:
                text = self._repair_entailment_failures(
                    text, judgement, lang_violations, policy, machine,
                )
                # re-judge after repair
                judgement = judge.judge_answer(
                    text, pack, topic=plan.topic if plan else None,
                    requested_object=plan.requested_object if plan else None,
                )
                trace.evidence_status = judgement.sufficiency.value
                trace.evidence_sufficiency = judgement.evidence_sufficiency
        companion_v = self.validator.validate(
            text, policy, evidence_present=bool(pack is not None and pack.passages)
        )
        if not companion_v.ok:
            # deterministic repairs for the worst classes
            text = self._repair_dependency(text)
            if companion_v.uncited_religious_claims:
                text = self._strip_religious_claims(text)
            companion_v = self.validator.validate(
                text, policy, evidence_present=bool(pack is not None and pack.passages)
            )
            if not companion_v.ok and companion_v.uncited_religious_claims:
                # still failing -> keep only the empathic opening + honest notice
                first_line = text.split("\n")[0][:200]
                text = (
                    f"{first_line}\n\nI could not verify this from the approved "
                    "source corpus."
                )
                companion_v = self.validator.validate(
                    text, policy, evidence_present=bool(pack is not None and pack.passages)
                )

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
    def _embed_fn(self):
        """Semantic signal for the judge: the corpus vector store's embedder,
        or None when unavailable (judge degrades to lexical+type signals)."""
        if self.retrieval is None or self.retrieval.vector_store is None:
            return None
        try:
            client = self.retrieval.vector_store.client
            if client is None:
                from agent.core.config import load_config

                client = __import__(
                    "agent.core.embeddings", fromlist=["EmbeddingClient"]
                ).EmbeddingClient(load_config())
            return client.embed_one
        except Exception:
            return None

    def _quarantine_irrelevant(self, passages, plan) -> list:
        """fixme_v3 §12 evidence filter: grade candidate passages against
        the query BEFORE the LLM sees them; drop irrelevant ones so the model
        cannot stitch them into claims (small models happily connect anything)."""
        from agent.validators.evidence_judge import _content_stems

        if not passages:
            return passages
        query_terms = set()
        for term in (plan.retrieval_terms if plan else []):
            query_terms |= _content_stems(term)
        kept = []
        for p in passages:
            text = (p.translation or p.arabic or "")
            p_stems = _content_stems(text)
            overlap = len(query_terms & p_stems)
            # passage must share at least one content stem with the planned
            # need OR be a tier-0/1 reference hit (deterministic anchors)
            if overlap >= 1 or p.leg == "reference":
                kept.append(p)
        return kept if kept else passages[:1]  # never drop everything

    def _repair_entailment_failures(self, text, judgement, lang_violations,
                                    policy, machine) -> str:
        """fixme_v3 §8/§16: entailment failures repair.
        - SUPPORTS claims keep their sentences
        - PARTIAL claims must be re-worded conservatively -> drop if they
          carried strong language
        - IRRELEVANT/UNKNOWN claim sentences are removed
        - if nothing strong remains, produce the careful-scope fallback:
          honest partial answer + no false guarantee."""

        if lang_violations or any(
            j.verdict in (Verdict.IRRELEVANT, Verdict.UNKNOWN)
            for j in judgement.claim_support
        ):
            # rebuild from the sentences that survived judging
            bad_sentences = {j.claim for j in judgement.claim_support
                              if j.verdict in (Verdict.IRRELEVANT, Verdict.UNKNOWN)}
            out_lines: list[str] = []
            for paragraph in text.split("\n"):
                kept_parts: list[str] = []
                for sentence in re.split(r"(?<=[.!?\n])\s+", paragraph):
                    if any(bad in sentence or sentence in bad for bad in bad_sentences):
                        continue
                    kept_parts.append(sentence)
                if kept_parts:
                    out_lines.append(" ".join(kept_parts))
            rebuilt = "\n".join(out_lines).strip()
            # §8 fallback when the rebuild is empty or the judge still fails
            supported = [
                j for j in judgement.claim_support
                if j.verdict is Verdict.SUPPORTS
            ]
            if supported and len(rebuilt) > 40:
                return rebuilt
            return self._careful_scope_fallback(judgement)
        return text

    @staticmethod
    def _careful_scope_fallback(judgement) -> str:
        """fixme_v3 §8: SUPPORTED CLAIM + CAREFUL SCOPE + NO FALSE GUARANTEE."""
        supported = [
            j for j in judgement.claim_support if j.verdict is Verdict.SUPPORTS
        ]
        if supported:
            best = supported[0]
            return (
                "There are supplications taught in the Sunnah for distress, "
                "worry and grief. I can share one of those with you. "
                "I wouldn't describe it as a guaranteed way to remove "
                "depression, though — Islamic supplication can be part of "
                "seeking comfort and turning to Allah, alongside getting "
                f"appropriate support. ({best.citation} carries the "
                "supplication itself.)"
            )
        return (
            "I could not verify this from the approved source corpus. "
            "If you're asking about supplications for distress, there are "
            "authentic ones in the Sunnah — would you like me to find one "
            "for you?"
        )

    @staticmethod
    def _looks_like_dua_text(p) -> bool:
        """Actual supplication words (quoted dua), not text ABOUT dua."""
        import re as _re

        t = (p.translation or "").lower()
        return bool(
            _re.search(r"\bo allah\b", t)
            or _re.search(r"\ballahum(a|ma)\b", t)
            or "i seek refuge" in t
            or "seek refuge in you" in t
            or "i seek protection" in t
        )

    def _row_to_hadith_passage(self, row: dict):
        from agent.state.models import Risk  # noqa: F401  (docstring only)
        from retrieval.hybrid import RetrievedPassage

        return RetrievedPassage(
            citation_id=row["citation_id"], surah=0, ayah=0,
            arabic=row["arabic"], source_id=row["source_id"], tier=1,
            leg="hadith", score=row.get("rank", -1.0),
            translation=row.get("english") or "",
            collection=row["source_id"], hadithnumber=row.get("hadithnumber"),
            grades=row.get("grades") or None,
        )

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
    def _strip_citation_sentence(text: str, citation: str) -> str:
        """Remove every sentence carrying this citation marker (misquote or
        unsupported repair): keeping the marker but dropping the claim is
        not acceptable — the whole claim sentence goes."""
        pattern = re.compile(
            r"[^.!?\n]*" + re.escape(citation) + r"[^.!?\n]*[.!?]?",
            re.IGNORECASE,
        )
        text = pattern.sub("", text)
        text = text.replace(f"[{citation}]", "").replace(f"({citation})", "")
        text = re.sub(r"[ \t]{2,}", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

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
    def _strip_religious_claims(text: str) -> str:
        """§22: remove sentences carrying uncited religious claims, keep the
        empathy and everything else."""
        from agent.validators.companion_validator import CITATION_MARKER_RE, RELIGIOUS_CLAIM_RE

        out_lines: list[str] = []
        for paragraph in text.split("\n"):
            kept: list[str] = []
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                if RELIGIOUS_CLAIM_RE.search(sentence) and not CITATION_MARKER_RE.search(sentence):
                    continue  # drop the uncited claim sentence
                kept.append(sentence)
            out_lines.append(" ".join(kept))
        cleaned = "\n".join(out_lines)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _repair_dependency(text: str) -> str:
        from agent.validators.companion_validator import DEPENDENCY_RE

        return DEPENDENCY_RE.sub("I'm here to listen", text)

    @staticmethod
    def _lang(message: str) -> str:
        return "ms" if re.search(
            r"\b(saya|aku|tak|nak|dengan|yang)\b", message, re.IGNORECASE
        ) else "en"
