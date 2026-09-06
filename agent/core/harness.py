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
from agent.companion.logging import CompanionLogger
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
from agent.validators.pipeline import (
    UNVERIFIABLE_NOTICE,
    CitationValidator,
    EvidencePack,
)

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
    "warmth only.\n\n"
    "When the user asks how to BEGIN learning about Islam, do not list "
    "generic study advice: use the evidence as the answer — present the "
    "framework it contains (e.g. the pillars of Islam, the dimensions of "
    "the religion) as the starting map, quoting and citing it, then invite "
    "them to explore it together with you."
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
        chat_logger: CompanionLogger | None = None,
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
        # chat logging (owner-approved troubleshooting capture); None-safe
        self.chat_logger = chat_logger or CompanionLogger()

    # ----------------------------------------------------------------- turn
    def respond(self, session_id: str, message: str,
                max_tokens: int | None = None) -> HarnessResult:
        trace = DebugTrace(model=self.model_label)
        # Ling-family models always reason internally (enable_thinking is
        # ignored by LM Studio); the reasoning shares the output budget, so
        # a small budget yields EMPTY answers on evidence-heavy RAG turns.
        # QA/RAG turns carry long evidence -> 4096; companion chat -> 2048.
        if max_tokens is None:
            max_tokens = 4096 if self._next_mode_is_qa(session_id, message) else 2048
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
            self.chat_logger.log_turn(session_id, state.turn_count, message, text, result, self.model_label)
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
            # deterministic concept anchors (fixme_v3 §6): the canonical
            # narrations for well-known concepts go straight to the pack head
            if plan.anchor_citations:
                anchor_rows = []
                for cid in plan.anchor_citations:
                    if cid.startswith("hadith:"):
                        _, collection, number = cid.split(":")
                        if self.retrieval.hadith_store is not None:
                            row = self.retrieval.hadith_store.get_hadith(
                                collection, int(number)
                            )
                            if row:
                                anchor_rows.append(self._row_to_hadith_passage(row))
                    elif cid.startswith("quran:"):
                        _, surah, ayah = cid.split(":")
                        row = self.retrieval.store.get_ayah(int(surah), int(ayah), lang="en")
                        if row:
                            anchor_rows.append(self.retrieval._to_passage(row, "reference", 1.0))
                anchor_ids = {r.citation_id for r in anchor_rows}
                passages = anchor_rows + [
                    p for p in passages if p.citation_id not in anchor_ids
                ]
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
                self.chat_logger.log_turn(session_id, state.turn_count, message, text, result, self.model_label)
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
                "religious quote will be removed. CRITICAL: state ONLY what the "
                "quoted passage itself says — do not add details from your own "
                "knowledge (numbers, times, conditions) that the quoted text does "
                "not contain. If the evidence lists items, present those items "
                "exactly as the evidence words them. Unsupported additions will "
                "be detected and removed."
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
        text = resp.content.strip()
        # Empty/length-cut model output: RAG route (a question was asked,
        # evidence exists) -> one doubled-budget retry, then the honest §12
        # notice — NEVER the companion-listening line, which answers a
        # different question ("I hear you" to "what is wudoo?" is wrong).
        # Companion route keeps the empathic fallback: listening IS valid.
        if not text:
            if state.mode is Mode.COMPANION:
                text = "I hear you. If you want to tell me more, I'm listening."
            else:
                retry = self.router.chat(
                    "complex_rag", messages, max_tokens=max_tokens * 2,
                )
                text = retry.content.strip()
                trace.notes.append("empty QA output: retried with doubled budget")
        if not text:
            if state.mode is Mode.COMPANION:
                text = "I hear you. If you want to tell me more, I'm listening."
            else:
                text = UNVERIFIABLE_NOTICE
                trace.notes.append("QA empty after retry: honest notice")

        # 8. VALIDATION (§21-22 + §25) + fixme_v3 CLAIM→EVIDENCE ENTAILMENT:
        # existence checks, then the judge decides what the answer is
        # ALLOWED to claim (§4-5, §9-13).
        citations: list[str] = []
        unsupported: list[str] = []
        if pack is not None:
            # first: rebind malformed-but-genuine quotes to their real source
            text = self._repair_malformed_citations(text, pack)
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

        # EVIDENCE JUDGE (fixme_v3 §4 + v3.1 §31-33): claim→evidence
        # entailment on every claim; typed verdicts feed the language gate;
        # repair is BOUNDED (max 2 rounds §32) and ALWAYS re-validated —
        # repair success is never assumed. The final answer gate (§33) is
        # enforced before anything reaches the user.
        judgement = None
        if pack is not None and pack.passages and policy.requires_rag:
            judge = EvidenceJudge(embed=self._embed_fn())
            judgement = judge.judge_answer(
                text, pack, topic=plan.topic if plan else None,
                requested_object=plan.requested_object if plan else None,
            )
            trace.evidence_status = judgement.sufficiency.value
            trace.evidence_sufficiency = judgement.evidence_sufficiency
            trace.validation_trace = [j.to_dict() for j in judgement.claim_support]

            MAX_REPAIR_ROUNDS = 2  # v3.1 §32
            for round_no in range(MAX_REPAIR_ROUNDS):
                lang_violations = language_strength_ok(text, judgement)
                unsupported_claims = [
                    j for j in judgement.claim_support
                    if j.verdict in (Verdict.IRRELEVANT, Verdict.UNKNOWN)
                ]
                if not lang_violations and not unsupported_claims:
                    break  # clean; no repair needed
                text = self._repair_entailment_failures(
                    text, judgement, lang_violations, policy, machine,
                )
                # §31: repair must revalidate — claim extraction and judging
                # run AGAIN on the repaired text; the loop exits either when
                # clean or after the bounded rounds
                judgement = judge.judge_answer(
                    text, pack, topic=plan.topic if plan else None,
                    requested_object=plan.requested_object if plan else None,
                )
                trace.evidence_status = judgement.sufficiency.value
                trace.evidence_sufficiency = judgement.evidence_sufficiency
                trace.notes.append(f"repair round {round_no + 1} revalidated")

            # §33 FINAL ANSWER GATE: after bounded repair, any surviving
            # high-risk unsupported claim forces the safe fallback — it may
            # NEVER ship as part of a confident answer
            survivors = [
                j for j in judgement.claim_support
                if j.verdict in (Verdict.IRRELEVANT, Verdict.UNKNOWN)
                and j.is_high_risk
            ]
            if survivors:
                text = self._careful_scope_fallback(judgement)
                judgement = judge.judge_answer(
                    text, pack, topic=plan.topic if plan else None,
                    requested_object=plan.requested_object if plan else None,
                )
                trace.evidence_status = judgement.sufficiency.value
                trace.notes.append(
                    f"final gate: {len(survivors)} high-risk claims -> fallback"
                )
            trace.validation_trace = [j.to_dict() for j in judgement.claim_support]
        companion_v = self.validator.validate(
            text, policy, evidence_present=bool(pack is not None and pack.passages)
        )
        if not companion_v.ok:
            # deterministic repairs for the worst classes (v3.1 §31-33:
            # bounded, revalidated; the final gate below enforces the §33
            # checklist — a still-failing answer ships only de-fanged)
            text = self._repair_dependency(text)
            if companion_v.uncited_religious_claims:
                text = self._strip_religious_claims(text)
            if any("diagnosis" in p for p in companion_v.companion_problems):
                text = self._strip_diagnosis_sentences(text)
            if any("too many questions" in p for p in companion_v.policy_problems):
                text = self._trim_extra_questions(text, policy)
            companion_v = self.validator.validate(
                text, policy, evidence_present=bool(pack is not None and pack.passages)
            )
            if not companion_v.ok and (
                companion_v.uncited_religious_claims
                or any("diagnosis" in p for p in companion_v.companion_problems)
            ):
                # still failing -> keep only the empathic opening + notice
                first_line = text.split("\n")[0][:200]
                text = (
                    f"{first_line}\n\nI could not verify this from the approved "
                    "source corpus."
                )
                companion_v = self.validator.validate(
                    text, policy,
                    evidence_present=bool(pack is not None and pack.passages),
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
        self.chat_logger.log_turn(session_id, state.turn_count, message, text, result, self.model_label)
        return result

    # ------------------------------------------------------------- internals
    @staticmethod
    def _next_mode_is_qa(session_id: str, message: str) -> bool:
        """Budget heuristic BEFORE classification runs: an islamic question
        becomes a QA/RAG turn (long evidence in context); anything else stays
        companion chat. Mirrors the policy engine's QA-mode criteria."""
        from agent.companion.intent import classify_companion

        try:
            ci = classify_companion(message)
            return bool(ci.is_question and ci.needs_islamic_guidance)
        except Exception:
            return False

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
        # ubiquitous stems carry no relevance signal (nearly every Islamic
        # text contains allah/quran/prophet — overlap on those alone is
        # indistinguishable from chance)
        UBIQUITOUS = {"allah", "quran", "qur_a", "prophe", "messa", "hadit", "islam"}
        query_terms = set()
        for term in (plan.retrieval_terms if plan else []):
            query_terms |= _content_stems(term)
        kept = []
        for p in passages:
            text = (p.translation or p.arabic or "")
            p_stems = _content_stems(text)
            overlap = len((query_terms & p_stems) - UBIQUITOUS)
            # passage must share at least one DISCRIMINATIVE stem with the
            # planned need OR be a tier-0/1 reference hit (deterministic anchors)
            if overlap >= 1 or p.leg == "reference":
                kept.append(p)
        # fixme_v3.1 §4: if everything is irrelevant, NOTHING re-enters
        # generation. Never deliberately reintroduce quarantined evidence.
        # INSUFFICIENT_EVIDENCE propagates downstream instead.
        return kept

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
        """fixme_v3 §8: SUPPORTED CLAIM + CAREFUL SCOPE + NO FALSE GUARANTEE.
        Generic wording — the topic-specific offer (dua etc.) is handled by
        the intent-specific flows, not here."""
        supported = [
            j for j in judgement.claim_support if j.verdict is Verdict.SUPPORTS
        ]
        if supported:
            best = supported[0]
            return (
                "I can share what the sources say on this, though I could "
                "not fully verify every detail of what I first wrote. "
                f"The most directly relevant passage is {best.citation} — "
                "would you like me to walk through it?"
            )
        return (
            "I could not verify this from the approved source corpus. "
            "If it helps, tell me more about what you're looking for and "
            "I'll search the sources again."
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
    def _strip_diagnosis_sentences(text: str) -> str:
        """Remove sentences carrying diagnosis-like claims (§33: no
        diagnosis ever ships)."""
        from agent.validators.companion_validator import DIAGNOSIS_RE

        out_lines = []
        for paragraph in text.split("\n"):
            kept = []
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                if DIAGNOSIS_RE.search(sentence):
                    continue
                kept.append(sentence)
            if kept:
                out_lines.append(" ".join(kept))
        return re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()

    @staticmethod
    def _trim_extra_questions(text: str, policy) -> str:
        """§33 follow-up gate: keep only max_followups questions."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept = []
        question_budget = max(getattr(policy, "max_followups", 1), 0)
        for sentence in sentences:
            if sentence.rstrip().endswith("?"):
                if question_budget > 0:
                    question_budget -= 1
                    kept.append(sentence)
                continue
            kept.append(sentence)
        return " ".join(kept)

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

    def _repair_malformed_citations(self, text: str, pack) -> str:
        """Models emit malformed citation forms (e.g. 'quran:surah:al-imran:
        3:109'). When the surrounding claim QUOTES a pack passage, rebind the
        marker to that passage's real citation id — the quote is genuine, only
        the marker form is wrong. Surgical: only exact-quote matches repair."""
        import re as _re

        if not pack or not pack.passages:
            return text
        bad_marker = _re.compile(
            # well-formed brackets with invalid ids: [quran:surah:al-imran:3:109]
            r"\[(?P<bad>quran:surah:[a-z'\-]+:\d+:\d+|quran:s[a-z'\-]+:[^\]]*)\]"
            # unterminated/overlapping markers the models leave mid-text
            r"|\[(?:quran|qur'a?n):s(?:urah)?[:-][a-z'\-]*(?:[:-]\d+)*(?![\w\]:])"
            # truncated quran-marker fragments: '[qura' / '[quran' mid-prose,
            # only when NOT followed by more marker chars or a letter+colon
            r"|\[qur?a?n?(?=[\s,.)\]]|$)"
        )
        # replace right-to-left: mutating the string while iterating
        # forward shifted offsets and corrupted following words
        matches = list(bad_marker.finditer(text))
        for m in reversed(matches):
            window = text[max(0, m.start() - 400): m.start()]
            best_id, best_hits = None, 0
            for p in pack.passages:
                ptext = p.translation or p.arabic or ""
                hits = sum(
                    1 for frag in _re.findall(r"[a-z' ]{12,}", window.lower())
                    if frag.strip() in ptext.lower()
                )
                if hits > best_hits:
                    best_hits, best_id = hits, p.citation_id
            if best_id and best_hits > 0:
                text = text[: m.start()] + f"[{best_id}]" + text[m.end():]
        # strip model self-talk leaks (reasoning phrases that survived
        # into the final answer)
        text = _re.sub(
            r"[—-]?\s*wait,? let me check[^.]*\.?|"
            r"[—-]?\s*let me (check|verify|look)[^.]*\.?|"
            r"[—-]?\s*I need to (check|verify|find)[^.]*\.?",
            "",
            text,
        )
        text = _re.sub(r"\[qu\[", "[", text)  # rebind residue cleanup
        text = _re.sub(r"\s{2,}", " ", text)
        return text

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
