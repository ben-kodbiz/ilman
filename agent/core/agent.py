"""Agent orchestrator (agentodo.md §11, §12, §26 Phase 4).

Full agent loop: intent -> tools (model-driven, schema-strict) -> evidence
merge -> grounded answer -> citation validation. The model drives tool calls;
every tool enforces the Sunni source filter; the final answer is validated
deterministically. If validation fails once, one repair round with the
unsupported citations listed; after that the response is downgraded to the
§12 notice rather than ever surfacing an unverified claim (§12: DO NOT GUESS).

Exit condition (§26 Phase 4): agent completes multi-step source-grounded tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent.core.intent import IntentResult, classify
from agent.core.model import ChatMessage, ModelResponse
from agent.memory.store import ConversationMemory, MemoryStore
from agent.tools.layer import TOOL_SCHEMAS, ToolLayer, execute_tool
from agent.validators.pipeline import (
    RESPONSE_SYSTEM_PROMPT,
    UNVERIFIABLE_NOTICE,
    CitationValidator,
    EvidencePack,
)
from retrieval.hybrid import RetrievalOrchestrator, RetrievedPassage

MAX_TOOL_ROUNDS = 4


@dataclass
class AgentTrace:
    """Provenance of decisions — NO chain-of-thought is ever stored (§0)."""

    intent: str = ""
    task_class: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # name + args + ok
    retrieved_citations: list[str] = field(default_factory=list)
    rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class AgentResult:
    answer: str
    verified: bool
    refused: bool
    citations: list[str]
    unsupported_citations: list[str]
    evidence: EvidencePack
    trace: AgentTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "verified": self.verified,
            "refused": self.refused,
            "citations": self.citations,
            "unsupported_citations": self.unsupported_citations,
            "evidence": [
                {
                    "citation_id": p.citation_id,
                    "source_id": p.source_id,
                    "tier": p.tier,
                    "collection": p.collection or None,
                    "hadithnumber": p.hadithnumber,
                    "grades": p.grades if p.citation_id.startswith("hadith:") else None,
                }
                for p in self.evidence.passages
            ],
            "trace": self.trace.to_dict(),
        }


class AgentOrchestrator:
    def __init__(
        self,
        router,
        orchestrator: RetrievalOrchestrator,
        tools: ToolLayer,
        memory: MemoryStore | None = None,
        conversation: ConversationMemory | None = None,
    ):
        self.router = router
        self.retrieval = orchestrator
        self.tools = tools
        self.memory = memory
        self.conversation = conversation or ConversationMemory()
        self.validator = CitationValidator()

    def answer(self, query: str, limit: int = 6, max_tokens: int = 4096) -> AgentResult:
        intent: IntentResult = classify(query)
        trace = AgentTrace(intent=intent.intent, task_class=intent.routed_task_class)

        # 1. Seed evidence deterministically (§14 references never come from the model)
        passages: list[RetrievedPassage] = []
        for ref in intent.quran_refs[:3]:
            row = self.retrieval.store.get_ayah(ref["surah"], ref["ayah"])
            if row:
                passages.append(self.retrieval._to_passage(row, "reference", 1.0))
        for href in intent.hadith_refs[:3]:
            row = self.retrieval.hadith_store.get_hadith(href["collection"], href["number"]) \
                if self.retrieval.hadith_store else None
            if row:
                passages.append(self._hadith_row_to_passage(row, "reference", 1.0))
        # 2. Retrieval leg for search-ish intents (with emotional concept
        # expansions when the intent router detected them; emotional queries
        # go semantic-only — lexical matching on feelings is pure noise)
        seeded_ids = {p.citation_id for p in passages}
        retrieved: list[RetrievedPassage] = []
        if intent.intent in ("quran_search", "hadith_search", "question", "quran_lookup", "hadith_lookup"):
            for p in self.retrieval.search(
                query, limit=limit,
                concept_expansions=intent.concept_expansions or None,
                semantic_only=intent.emotional,
            ):
                if p.citation_id not in seeded_ids:
                    retrieved.append(p)
                    seeded_ids.add(p.citation_id)
        # Reference-seeded passages are DETERMINISTIC evidence (§14) — they
        # keep guaranteed top slots; retrieval fills the remainder by rank.
        # A large diluted pool buries the anchor verse and invites drift.
        max_pack = max(limit + 2, 8)
        pack = EvidencePack(
            query=query, passages=(passages + retrieved[:max(0, max_pack - len(passages))])[:max_pack]
        )
        if not pack.passages:
            self._record(query, intent, trace, [])
            return self._refuse(query, pack, trace)

        # 3. Model loop: answer with evidence, allow tool calls for more
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=RESPONSE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=self._user_prompt(query, pack)),
        ]
        final_text = ""
        pending_tool_results: list[ChatMessage] = []
        for round_no in range(MAX_TOOL_ROUNDS):
            if pending_tool_results:
                messages.extend(pending_tool_results)
                pending_tool_results = []
            resp = self.router.chat(
                intent.routed_task_class, messages,
                tools=TOOL_SCHEMAS, max_tokens=max_tokens,
            )
            trace.rounds = round_no + 1
            if resp.tool_calls:
                tool_result_messages = self._handle_tool_calls(resp, trace, pack, passages)
                messages.append(ChatMessage(role="assistant", content=resp.content or "",
                                            tool_calls=self._raw_tool_calls(resp)))
                messages.extend(tool_result_messages)
                continue
            final_text = resp.content
            break

        if not final_text.strip():
            final_text = UNVERIFIABLE_NOTICE

        # 4. Deterministic citation validation + one repair round (§12).
        # Repair runs in a FRESH context: after tool rounds the conversation
        # contains tool results with citations outside the pack, and a repair
        # appended to that context makes the model mix both citation sets.
        validation = self.validator.validate(final_text, pack)
        if validation.unsupported_citations:
            repair_prompt = (
                "Rewrite this answer using ONLY the citations listed in the "
                "evidence block below. Remove any citation not in that list; "
                "if that removes the substance, reply exactly: "
                f"{UNVERIFIABLE_NOTICE}\n\n"
                f"ANSWER TO REWRITE:\n{final_text[:2000]}\n\n"
                f"<evidence>\n{pack.to_prompt_block()}\n</evidence>\n\n"
                f"Allowed citations: {', '.join(sorted(pack.citation_ids))}. Keep it brief."
            )
            repair_messages = [
                ChatMessage(role="system", content=RESPONSE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=repair_prompt),
            ]
            resp = self.router.chat(
                intent.routed_task_class, repair_messages, max_tokens=max_tokens
            )
            if resp.content.strip():
                final_text = resp.content
                validation = self.validator.validate(final_text, pack)

        refused = UNVERIFIABLE_NOTICE in final_text
        verified = not validation.unsupported_citations and (
            validation.had_any_citation or refused
        )
        trace.retrieved_citations = sorted(pack.citation_ids)
        self._record(query, intent, trace, validation.verified_citations)

        # 5. Conversation memory (short-lived, content only)
        self.conversation.add("user", query)
        self.conversation.add("assistant", final_text)

        return AgentResult(
            answer=final_text,
            verified=verified,
            refused=refused,
            citations=validation.verified_citations,
            unsupported_citations=validation.unsupported_citations,
            evidence=pack,
            trace=trace,
        )

    def _handle_tool_calls(self, resp: ModelResponse, trace: AgentTrace,
                           pack: EvidencePack, passages: list[RetrievedPassage]) -> list[ChatMessage]:
        """Execute requested tools; merge retrieved evidence; build tool messages."""
        tool_messages: list[ChatMessage] = []
        for tc in resp.tool_calls:
            result = execute_tool(self.tools, tc.name, tc.arguments)
            trace.tool_calls.append({
                "name": tc.name,
                "args": tc.arguments,
                "ok": result.ok,
                "error": result.error or None,
            })
            tool_messages.append(ChatMessage(
                role="tool",
                content=json.dumps(
                    {"ok": result.ok, "data": result.data, "error": result.error},
                    ensure_ascii=False,
                )[:4000],
            ))
            if result.ok and result.data:
                self._merge_tool_evidence(tc.name, result.data, pack, passages)
        return tool_messages

    def _merge_tool_evidence(self, name: str, data: dict, pack: EvidencePack,
                             passages: list[RetrievedPassage]) -> None:
        if name in ("get_ayah", "verify_quran_reference") and "citation_id" in data:
            if data["citation_id"] not in pack.citation_ids:
                row = {
                    "citation_id": data["citation_id"], "surah": data["surah"],
                    "ayah": data["ayah"], "arabic": data["arabic"],
                    "translation": data.get("translation", ""),
                    "source_id": data["source_id"],
                }
                passages.append(self.retrieval._to_passage(row, "reference", 1.0))
        elif name in ("search_hadith",):
            for h in data.get("results", [])[:4]:
                if h["citation_id"] not in pack.citation_ids:
                    passages.append(self._hadith_row_to_passage(h, "hadith", h.get("rank", -1.0)))
        elif name == "get_hadith" and "citation_id" in data:
            if data["citation_id"] not in pack.citation_ids:
                passages.append(self._hadith_row_to_passage(data, "reference", 1.0))
        # rebuild pack passages from the working list
        pack.passages = passages[:12]

    def _hadith_row_to_passage(self, row: dict, leg: str, score: float) -> RetrievedPassage:
        return RetrievedPassage(
            citation_id=row["citation_id"], surah=0, ayah=0,
            arabic=row["arabic"], source_id=row["source_id"], tier=1,
            leg=leg, score=score, translation=row.get("english") or "",
            collection=row["source_id"], hadithnumber=row.get("hadithnumber"),
            grades=row.get("grades") or None,
        )

    def _user_prompt(self, query: str, pack: EvidencePack) -> str:
        return (
            f"<evidence>\n{pack.to_prompt_block()}\n</evidence>\n\n"
            f"Question: {query}\n\n"
            "Answer the question using ONLY the evidence above. You may call "
            "tools (get_ayah, search_hadith, get_hadith) for more evidence "
            "first if the question asks about a specific reference not in the "
            "evidence. Quote exactly when citing. Cite as "
            "[quran:surah:ayah] or [hadith:collection:number]. If the "
            "evidence contains nothing relevant, reply exactly: "
            f"{UNVERIFIABLE_NOTICE}"
        )

    def _refuse(self, query: str, pack: EvidencePack, trace: AgentTrace) -> AgentResult:
        self.conversation.add("user", query)
        self.conversation.add("assistant", UNVERIFIABLE_NOTICE)
        return AgentResult(
            answer=UNVERIFIABLE_NOTICE, verified=True, refused=True,
            citations=[], unsupported_citations=[], evidence=pack, trace=trace,
        )

    def _record(self, query: str, intent: IntentResult, trace: AgentTrace, citations: list[str]) -> None:
        if self.memory is not None:
            self.memory.record_query(query, intent.intent, citations)

    @staticmethod
    def _raw_tool_calls(resp: ModelResponse) -> list[dict]:
        """Re-serialize tool calls for the next round's messages."""
        out = []
        for tc in resp.tool_calls:
            out.append({
                "type": "function",
                "id": f"call_{len(out)}",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            })
        return out
