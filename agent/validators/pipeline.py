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
    # bare bracketed collection:number form the models often emit;
    # ONLY accepted when it resolves to a pack citation (groups 11/12)
    r"|\[([a-z0-9\-]+):(\d+)\](?![\w:])"
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
    misattributed_grades: list[dict] = field(default_factory=list)
    # {"citation": ..., "grader": ..., "claimed": ..., "available": [...]}
    misquoted_citations: list[dict] = field(default_factory=list)
    # {"citation": ..., "claim": ...} — citation exists in the pack but the
    # cited passage does not content-support the claim sentence (misquote)

    @property
    def ok(self) -> bool:
        return (
            not self.unsupported_citations
            and not self.misattributed_grades
            and not self.misquoted_citations
        )


# Claim-support stopwords: claims share few words with evidence by design;
# only CONTENT stems count (no "the/and/of/allah" style glue).
_SUPPORT_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "with", "for", "as", "by", "at", "from", "not", "no",
    "but", "if", "then", "than", "so", "what", "which", "who", "when",
    "where", "why", "how", "all", "any", "some", "there", "here", "his",
    "her", "its", "their", "our", "your", "my", "me", "us", "them", "also",
    "one", "only", "every", "us", "we", "you", "allah", "god", "he", "him",
}


def _content_stems(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z']{4,}", text.lower())
    return {w[:5] for w in words if w not in _SUPPORT_STOP}


def _check_claim_support(answer: str, pack: EvidencePack) -> list[dict]:
    """Misquote detection: for each in-pack citation in the answer, the claim
    sentence around it must share content stems with the cited passage's own
    text. Citing a real verse that says something unrelated (e.g. 112:4
    'none comparable to Him' for a depression claim) is a misquote even
    though the citation exists. Quoted spans always count as support."""
    problems: list[dict] = []
    by_citation = {p.citation_id: p for p in pack.passages}
    for m in CITATION_RE.finditer(answer):
        if m.group(5):
            citation = f"hadith:{m.group(5)}:{m.group(6)}"
        elif m.group(1):
            citation = f"quran:{m.group(1)}:{m.group(2)}"
        elif m.group(7):
            citation = f"tafsir:{m.group(7)}:{m.group(8)}:{m.group(9)}"
        elif m.group(10):
            citation = f"tafsir-en:{m.group(10)}"
        elif m.group(11):
            citation = f"hadith:{m.group(11)}:{m.group(12)}"
        else:
            citation = f"quran:{m.group(3)}:{m.group(4)}"
        passage = by_citation.get(citation)
        if passage is None:
            continue  # unsupported path handles missing citations
        # claim sentence = the sentence containing the citation
        sent_start = answer.rfind(".", 0, m.start())
        sent_start = answer.rfind("\n", 0, m.start()) if answer.rfind("\n", 0, m.start()) > sent_start else sent_start
        sent_end = answer.find(".", m.end())
        if sent_end == -1:
            sent_end = len(answer)
        claim_sentence = answer[sent_start + 1: sent_end] or answer[sent_start: sent_end]
        # direct quotation of the passage = always supported
        passage_text = (passage.translation or passage.arabic or "")
        norm_claim = re.sub(r"[^a-z\s]", "", claim_sentence.lower())
        norm_passage = re.sub(r"[^a-z\s]", "", passage_text.lower())
        if norm_claim and norm_claim in norm_passage:
            continue
        claim_stems = _content_stems(claim_sentence)
        passage_stems = _content_stems(passage_text)
        if not claim_stems or not passage_stems:
            continue
        overlap = len(claim_stems & passage_stems)
        # threshold: at least 2 shared content stems, or >=25% of the claim's
        # stems, when the sentence asserts (not merely references) — we only
        # flag sentences that look like assertions of religious content
        if overlap == 0 and len(claim_stems) >= 3:
            problems.append({
                "citation": citation,
                "claim": " ".join(claim_sentence.split())[:140],
            })
    return problems


# Grader-name aliases -> canonical dataset names (grades_json "name" values)
_GRADER_ALIASES = {
    "al-albani": ["al albani", "al-albani", "albani", "shaykh al-albani", "sheikh al-albani"],
    "shuaib-al-arnaut": ["shuaib al arnaut", "al arnaut", "al-arnaut", "arnaut", "shu'aib al-arnaut"],
    "abu-ghuddah": ["abu ghuddah", "ghuddah"],
    "zubair-ali-zai": ["zubair ali zai", "zubair 'ali zai", "ali zai", "zai"],
    "ahmad-muhammad-shakir": ["ahmad muhammad shakir", "shakir"],
    "muhammad-muhyi-al-din-abdul-hamid": [
        "muhammad muhyi al-din abdul hamid", "abdul hamid", "muhyi al-din",
    ],
}
_GRADE_WORDS = {
    "sahih": ["sahih"],
    "hasan": ["hasan", "hasan sahih", "sahih hasan", "hasan lighairihi", "sahih lighairihi"],
    "daif": ["da'if", "daif", "weak"],
    "munkar": ["munkar"],
    "fabricated": ["fabricated", "mawdu"],
    "isnaad-hasan": ["isnaad hasan", "isnad hasan"],
}

# "Al-Albani ... Sahih" / "graded sahih by Al-Albani" / "Al-Albani: Hasan"
_GRADER_CLAIM_RE = re.compile(
    r"(?P<grader>al[-\s]?albani|al[-\s]?arnaut|abu\s+ghuddah|zubair.{0,3}ali\s+zai|"
    r"ahmad\s+muhammad\s+shakir|shakir|abdul\s+hamid|muhyi\s+al[-\s]?din)"
    r"[^.\n]{0,140}?\b(?P<grade>sahih|hasan|da'?if|weak|fabricated|mawdu|munkar)\b"
    r"|[^.\n]{0,60}\b(?P<grade2>sahih|hasan|da'?if|weak|fabricated|mawdu|munkar)\b"
    r"[^.\n]{0,20}?\b(?:by|according to|from)\s+(?P<grader2>al[-\s]?albani|al[-\s]?arnaut|"
    r"abu\s+ghuddah|zubair.{0,3}ali\s+zai|shakir|abdul\s+hamid)",
    re.IGNORECASE,
)


def _canonical_grader(name: str) -> str:
    lowered = re.sub(r"[^a-z\s'-]", "", name.lower()).strip()
    for canonical, aliases in _GRADER_ALIASES.items():
        if lowered in [re.sub(r"[^a-z\s'-]", "", a) for a in aliases]:
            return canonical
    return ""


def _canonical_grade(grade: str) -> str:
    lowered = grade.lower().strip()
    for canonical, variants in _GRADE_WORDS.items():
        if lowered in variants:
            return canonical
    # grade combos like "Hasan Sahih" -> the closest single key
    for canonical, variants in _GRADE_WORDS.items():
        if any(v in lowered for v in variants):
            return canonical
    return lowered


def _check_grade_attribution(answer: str, pack: EvidencePack) -> list[dict]:
    """§6/§13: a claimed grader+grade must exist in the GRADES metadata of the
    hadith the model actually cited. Attributing a parallel narration's grading
    to the asked-about hadith is a misattribution, deterministic to catch."""
    problems: list[dict] = []
    hadith_citations = [p for p in pack.passages if p.citation_id.startswith("hadith:")]
    if not hadith_citations or not re.search(r"(albani|arnaut|ghuddah|zai|shakir|hamid)", answer, re.IGNORECASE):
        return problems
    # grades available per cited hadith
    per_hadith: dict[str, dict[str, set[str]]] = {}
    for p in hadith_citations:
        canonical: dict[str, set[str]] = {}
        for g in p.grades or []:
            grader = _canonical_grader(g.get("name", ""))
            grades = canonical.setdefault(grader, set())
            grades.add(_canonical_grade(g.get("grade", "")))
        per_hadith[p.citation_id] = canonical
    any_grader_grades: dict[str, set[str]] = {}
    for grader_map in per_hadith.values():
        for grader, grades in grader_map.items():
            any_grader_grades.setdefault(grader, set()).update(grades)

    for m in _GRADER_CLAIM_RE.finditer(answer):
        grader_raw = m.group("grader") or m.group("grader2")
        grade_raw = m.group("grade") or m.group("grade2")
        grader = _canonical_grader(grader_raw or "")
        claimed = _canonical_grade(grade_raw or "")
        if not grader:
            continue
        # nearest hadith citation by distance (prose may cite before or after
        # the grader mention: "X is graded Sahih by Al-Albani [hadith:...]"
        # or "[hadith:...] graded Sahih by Al-Albani")
        mid = (m.start() + m.end()) // 2
        nearest = None
        nearest_dist = None
        for cm in CITATION_RE.finditer(answer):
            if not cm.group(5):
                continue
            citation = f"hadith:{cm.group(5)}:{cm.group(6)}"
            if citation not in per_hadith:
                continue
            dist = abs(mid - (cm.start() + cm.end()) // 2)
            if nearest_dist is None or dist < nearest_dist:
                nearest = citation
                nearest_dist = dist
        # attribution is checked against the hadith actually being discussed;
        # only when no hadith citation exists anywhere do we fall back to all
        candidates = [nearest] if nearest else list(per_hadith)
        supported = any(
            claimed in per_hadith[c].get(grader, set())
            for c in candidates
        )
        if not supported:
            available = sorted(
                g for c in candidates for g in per_hadith[c].get(grader, set())
            ) or sorted(
                f"{g}:{','.join(sorted(gs))}" for c in candidates for g, gs in per_hadith[c].items()
            )
            problems.append({
                "citation": nearest or "(unspecified hadith)",
                "grader": grader,
                "claimed": claimed,
                "available": available[:4],
            })
    return problems


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
            elif m.group(11):  # bare [collection:number] bracket form:
                # accepted ONLY when it resolves to a pack hadith citation
                bare = f"hadith:{m.group(11)}:{m.group(12)}"
                if bare in allowed:
                    if bare not in verified:
                        verified.append(bare)
                elif bare not in unsupported:
                    unsupported.append(bare)
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
        misattributed = _check_grade_attribution(answer, pack)
        # NOTE: claim-SUPPORT (does the cited passage entail the claim?) moved
        # to the EvidenceJudge (agent/validators/evidence_judge.py, fixme_v3
        # §4-5) — the old single-sentence stem heuristic was cruder and
        # over-stripped legitimate quoted-dua answers. The judge is the single
        # entailment authority; misquoted_citations stays for API compat.
        misquoted = []
        return ValidationResult(
            verified, unsupported, found_any, misattributed, misquoted
        )


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
