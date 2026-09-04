"""Citation validation + response pipeline (agentodo.md §12, §13, §14).

The model NEVER produces religious text from memory. It receives an evidence
pack, and every citation in its answer is checked deterministically against
that pack. If a claim cannot be supported: DO NOT GUESS — return the
unverifiable notice instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.tools.quran_refs import extract_references
from retrieval.hybrid import RetrievedPassage

UNVERIFIABLE_NOTICE = "I could not verify this from the approved source corpus."

# Citation forms the composer may emit: quran:2:255 / hadith:collection:123 /
# tafsir:tafsir-kemenag:2:255 / tafsir-en:<chunk_id>.
CITATION_RE = re.compile(
    r"\bquran:(\d{1,3}):(\d{1,3})\b"
    r"|\b(\d{1,3}):(\d{1,3})\b"
    r"|\bhadith:([a-z0-9\-]+):(\d+)\b"
    r"|\btafsir:([a-z0-9\-]+):(\d{1,3}):(\d{1,3})\b"
    r"|\btafsir-en:([a-z0-9_\-]+)\b"
)


@dataclass
class EvidencePack:
    """§8: provenance pack every source-dependent answer must carry."""

    query: str
    passages: list[RetrievedPassage] = field(default_factory=list)

    @property
    def citation_ids(self) -> set[str]:
        return {p.citation_id for p in self.passages}

    def to_prompt_block(self) -> str:
        if not self.passages:
            return "NO EVIDENCE AVAILABLE."
        lines = []
        for p in self.passages:
            # Long tafsir/hadith passages are truncated for prompt context;
            # the model quotes what it needs, full text is retrievable via tools.
            def _trim(text: str, limit: int = 1200) -> str:
                text = text or ""
                return text if len(text) <= limit else text[:limit] + " [...]"
            if p.citation_id.startswith("hadith:"):
                grades = "; ".join(
                    f"{g.get('name', '?')}: {g.get('grade', '?')}" for g in (p.grades or [])
                )
                lines.append(
                    f"[{p.citation_id}] ({p.collection or p.source_id}, hadith {p.hadithnumber})\n"
                    f"{_trim(p.arabic, 400)}"
                    + (f"\nEN: {_trim(p.translation)}" if p.translation else "")
                    + (f"\nGRADES (dataset metadata, cite verbatim): {grades}" if grades else "")
                )
            elif p.citation_id.startswith("tafsir"):
                scholar = f" by {p.scholar}" if p.scholar else ""
                label = (
                    f"Tafsir Kemenag on Qur'an {p.surah}:{p.ayah}"
                    if not p.scholar else f"{p.scholar}'s tafsir (Qur'an {p.surah}:{p.ayah} area)"
                )
                lines.append(
                    f"[{p.citation_id}] ({label}{scholar if not p.scholar else ''}; "
                    "interpretation, TIER 2 — present as tafsir, never as Qur'an text)\n"
                    f"{_trim(p.translation)}"
                )
            else:
                lines.append(
                    f"[{p.citation_id}] (Qur'an {p.surah}:{p.ayah})\n{p.arabic}"
                    + (f"\nEN: {p.translation}" if p.translation else "")
                )
        return "\n\n".join(lines)


@dataclass
class ValidationResult:
    verified_citations: list[str] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    had_any_citation: bool = False

    @property
    def ok(self) -> bool:
        return not self.unsupported_citations


class CitationValidator:
    """Checks that every citation in model output exists in the evidence pack.

    Deterministic string/regex work — no model in the loop (§14).
    """

    def validate(self, answer: str, pack: EvidencePack) -> ValidationResult:
        allowed = pack.citation_ids
        allowed_refs = {tuple(p.citation_id.split(":")[1:3]) for p in pack.passages}
        verified: list[str] = []
        unsupported: list[str] = []
        found_any = False
        for m in CITATION_RE.finditer(answer):
            found_any = True
            if m.group(1):  # quran:s:a form
                surah, ayah = m.group(1), m.group(2)
                citation = f"quran:{surah}:{ayah}"
            elif m.group(5):  # hadith:collection:number form
                citation = f"hadith:{m.group(5)}:{m.group(6)}"
                if citation in allowed:
                    if citation not in verified:
                        verified.append(citation)
                elif citation not in unsupported:
                    unsupported.append(citation)
                continue
            elif m.group(7):  # tafsir:source:surah:ayah form
                citation = f"tafsir:{m.group(7)}:{m.group(8)}:{m.group(9)}"
                if citation in allowed:
                    if citation not in verified:
                        verified.append(citation)
                elif citation not in unsupported:
                    unsupported.append(citation)
                continue
            elif m.group(10):  # tafsir-en:<chunk_id> form
                citation = f"tafsir-en:{m.group(10)}"
                if citation in allowed:
                    if citation not in verified:
                        verified.append(citation)
                elif citation not in unsupported:
                    unsupported.append(citation)
                continue
            else:  # plain surah:ayah
                surah, ayah = m.group(3), m.group(4)
                citation = f"quran:{surah}:{ayah}"
            if citation in allowed or (surah, ayah) in allowed_refs:
                if citation not in verified:
                    verified.append(citation)
            else:
                if citation not in unsupported:
                    unsupported.append(citation)
        # plain "2:255" refs count as citations too
        if not found_any:
            for ref in extract_references(answer):
                found_any = True
                key = (str(ref["surah"]), str(ref["ayah"]))
                citation = f"quran:{ref['surah']}:{ref['ayah']}"
                if key in allowed_refs:
                    if citation not in verified:
                        verified.append(citation)
                elif citation not in unsupported:
                    unsupported.append(citation)
        return ValidationResult(verified, unsupported, found_any)


RESPONSE_SYSTEM_PROMPT = (
    "You are a careful Islamic study assistant. You answer ONLY from the evidence "
    "provided between <evidence> tags. Cite Qur'an quotations as [quran:surah:ayah] "
    "and hadith quotations as [hadith:collection:number]. Report hadith gradings "
    "only as given in the GRADES lines, verbatim; never invent gradings, and never "
    "state your own. If evidence answers the question only partially, answer with "
    "what the evidence supports and state plainly what is not covered (e.g. 'this "
    "edition carries no grading metadata for this hadith'). If the user shares a "
    "feeling or personal situation, respond with warmth and acknowledge them as a "
    "person first; then share ONLY evidence genuinely relevant to their situation — "
    "if the evidence does not truly speak to their situation, say so honestly "
    "rather than stretching a citation to fit. If the user asks you to VERIFY "
    "a specific claim (a quote, a reference, a grading), match it EXACTLY "
    "against the evidence: a passage merely about the same TOPIC does not "
    "verify the claim — say what matches and what does not. Never present an "
    "interpretation as "
    "Qur'an text itself. Refuse with the exact notice ONLY when the evidence "
    "contains nothing relevant to the question: "
    f"{UNVERIFIABLE_NOTICE} Never invent or alter Qur'an text, references, hadith, "
    "or scholar attributions. Never present your answer as a fatwa. Keep reasoning "
    "brief: decide which evidence passages answer the question, then write the "
    "final answer immediately."
)


@dataclass
class GroundedResponse:
    answer: str
    evidence: EvidencePack
    validation: ValidationResult | None = None
    verified: bool = False
    refused: bool = False

    def to_dict(self) -> dict[str, Any]:
        v = self.validation
        return {
            "answer": self.answer,
            "citations": v.verified_citations if v else [],
            "unsupported_citations": v.unsupported_citations if v else [],
            "verified": self.verified,
            "refused": self.refused,
            "evidence": [
                {"citation_id": p.citation_id, "surah": p.surah, "ayah": p.ayah,
                 "source_id": p.source_id, "tier": p.tier,
                 "translation": p.translation,
                 "translation_source_id": (
                     "quran-en-saheeh-json"
                     if p.translation and not p.citation_id.startswith("hadith:") else None
                 ),
                 "collection": p.collection or None,
                 "hadithnumber": p.hadithnumber,
                 "grades": p.grades if p.citation_id.startswith("hadith:") else None}
                for p in self.evidence.passages
            ],
        }


class ResponsePipeline:
    """Retrieval -> evidence pack -> model -> citation validation -> user.

    The model backend is the OpenAI-compatible abstraction (§3); this class
    never trusts the model for religious content, only for phrasing over the
    evidence pack.
    """

    def __init__(self, router):
        self.router = router  # agent.core.model.ModelRouter
        self.validator = CitationValidator()

    def answer(self, query: str, orchestrator, task_class: str = "complex_rag",
               limit: int = 6, max_tokens: int = 4096) -> GroundedResponse:
        from agent.core.model import ChatMessage

        passages = orchestrator.search(query, limit=limit)
        pack = EvidencePack(query=query, passages=passages)
        if not pack.passages:
            # No evidence -> unverifiable notice. DO NOT GUESS (§12).
            return GroundedResponse(UNVERIFIABLE_NOTICE, pack, None, verified=False, refused=True)
        user = (
            f"<evidence>\n{pack.to_prompt_block()}\n</evidence>\n\n"
            f"Question: {query}\n\n"
            "Answer the question using ONLY the evidence above. Quote exactly "
            "when citing. Cite as [quran:surah:ayah] or [hadith:collection:number]. "
            "If the evidence contains nothing relevant to the question, reply "
            f"exactly: {UNVERIFIABLE_NOTICE}"
        )
        messages = [
            ChatMessage(role="system", content=RESPONSE_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user),
        ]
        resp = self.router.chat(task_class, messages, max_tokens=max_tokens)
        # Reasoning-first backends can burn the whole budget on hidden CoT and
        # return empty content (known Ling-on-LMStudio behavior). One retry
        # with a doubled budget; if still empty, treat as unverifiable rather
        # than ever surfacing a guess.
        if not resp.content.strip() and resp.finish_reason == "length":
            resp = self.router.chat(task_class, messages, max_tokens=max_tokens * 2)
        if not resp.content.strip():
            return GroundedResponse(UNVERIFIABLE_NOTICE, pack, None, verified=False, refused=True)
        validation = self.validator.validate(resp.content, pack)
        refused = UNVERIFIABLE_NOTICE in resp.content
        verified = validation.ok and (validation.had_any_citation or refused)
        return GroundedResponse(resp.content, pack, validation, verified, refused)
