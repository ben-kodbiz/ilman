"""Source authority matrix (enhance_v1 §9-11).

Two-dimensional evidence validation (§11): ENTAILMENT (does the source
support the claim?) × AUTHORITY (is the source QUALIFIED for this claim
type?). These are separate checks with separate failure results:

    AUTHORITY_FAIL  — semantically relevant but wrong source class
                      (a lecture cannot establish 'The Prophet said X')
    ENTAILMENT_FAIL — right source class, doesn't support the claim

Authority model (§9.1):
    QURAN            → Quranic claims
    SAHIH_HADITH     → prophetic attribution
    TAFSIR           → tafsir interpretation (never as Qur'an text)
    APPROVED_FIQH    → fiqh rulings (none ingested yet — registry-pending)
    SCHOLAR/SECONDARY→ secondary explanation
    LECTURE          → secondary material (NOT authoritative for claims)
    LLM              → NO religious authority (§9.1: the LLM can formulate
                       language but never promote itself to a source)

Deterministic policy tables — no model (§45).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto


class AuthorityLevel(StrEnum):
    QURAN = auto()
    SAHIH_HADITH = auto()
    TAFSIR = auto()
    APPROVED_FIQH = auto()
    SCHOLARLY_SECONDARY = auto()
    LECTURE = auto()          # secondary material (§22 agentodo)
    EXTERNAL_UNTRUSTED = auto()  # §32: external web sources, never authoritative
    LLM = auto()             # NO religious authority


#: §10 claim-type authority requirements.
CLAIM_AUTHORITY_REQUIREMENTS: dict[str, set[AuthorityLevel]] = {
    # "The Quran says/mentions/commands X" — §24 'tafsir treated as Quran'
    # is an authority trap: a DIRECT_FACT whose speaker is the QURAN may only
    # be established by a quran citation. (Tafsir chunks quoting verses are
    # secondary carriers; the claim must cite the verse, or be typed TAFSIR.)
    "DIRECT_FACT": {AuthorityLevel.QURAN},
    "QURAN_CLAIM": {AuthorityLevel.QURAN},
    # "The Prophet ﷺ said/taught X"
    "HADITH_CLAIM": {AuthorityLevel.SAHIH_HADITH},
    "ATTRIBUTION": {AuthorityLevel.SAHIH_HADITH, AuthorityLevel.QURAN},
    # "Tafsir holds that X" — tafsir may interpret, never become Qur'an
    "TAFSIR": {AuthorityLevel.TAFSIR, AuthorityLevel.SCHOLARLY_SECONDARY},
    "PARAPHRASE": {AuthorityLevel.QURAN, AuthorityLevel.TAFSIR,
                  AuthorityLevel.SAHIH_HADITH},
    "INTERPRETATION": {AuthorityLevel.TAFSIR, AuthorityLevel.SCHOLARLY_SECONDARY},
    "FIQH_RULING": {AuthorityLevel.APPROVED_FIQH},
    "RULING": {AuthorityLevel.APPROVED_FIQH},
    # causal/guarantee/prediction/diagnosis/advice: no source class can
    # establish these as religious certainty (fixme_v3.1 §8 + enhance §42)
    "CAUSAL_CLAIM": set(),  # no authority level suffices for causal claims
    "GUARANTEE": set(),
    "DIAGNOSIS": set(),
    "PREDICTION": set(),
    # generalization: primary sources only ("Islam teaches X")
    "GENERALIZATION": {AuthorityLevel.QURAN, AuthorityLevel.SAHIH_HADITH},
    # inference must be labeled as inference (§8) — any class, but language-
    # gated downstream, never asserted
    "INFERENCE": {AuthorityLevel.QURAN, AuthorityLevel.TAFSIR,
                  AuthorityLevel.SAHIH_HADITH},
    # opinion/advice/Plain: non-religious, no authority requirement
    "OPINION": set(),
    "ADVICE": set(),
    "PLAIN": set(),
}

#: source_id / citation-prefix → authority level. The LLM is deliberately
#: absent: it is not a retrievable source.
SOURCE_AUTHORITY_BY_PREFIX: dict[str, AuthorityLevel] = {
    "quran:": AuthorityLevel.QURAN,
    "hadith:": AuthorityLevel.SAHIH_HADITH,  # registry-approved collections only
    "tafsir:": AuthorityLevel.TAFSIR,
    "tafsir-en:": AuthorityLevel.TAFSIR,
    "webfatwa:": AuthorityLevel.SCHOLARLY_SECONDARY,  # TIER 4 contemporary fatwa
}

#: registry-approved hadith collections (sahih/musanad class per registry)
_HADITH_COLLECTIONS = {
    "sahih-bukhari", "sahih-muslim", "sunan-abu-dawud",
    "jami-at-tirmidhi", "sunan-an-nasai", "sunan-ibn-majah",
}


@dataclass
class SourceAuthority:
    """§9 structure — resolved authority of one evidence source."""

    source_id: str
    authority_level: AuthorityLevel
    permitted_claim_types: set[str] = field(default_factory=set)
    restrictions: list[str] = field(default_factory=list)
    provenance: str = ""


class AuthorityResult(StrEnum):
    SUPPORTED = auto()
    AUTHORITY_FAIL = auto()   # §10: relevant but unauthorized source class
    ENTAILMENT_FAIL = auto()  # right class, doesn't support (judge's domain)
    NO_EVIDENCE = auto()


def resolve_authority(source_id_or_citation: str) -> SourceAuthority:
    citation = source_id_or_citation
    for prefix, level in SOURCE_AUTHORITY_BY_PREFIX.items():
        if citation.startswith(prefix):
            if level is AuthorityLevel.SAHIH_HADITH:
                parts = citation.split(":")
                if len(parts) >= 2 and parts[1] not in _HADITH_COLLECTIONS:
                    return SourceAuthority(
                        source_id=citation,
                        authority_level=AuthorityLevel.EXTERNAL_UNTRUSTED,
                        restrictions=["collection not in approved Kutub al-Sittah registry"],
                    )
            permitted = {
                ct for ct, reqs in CLAIM_AUTHORITY_REQUIREMENTS.items()
                if level in reqs
            }
            return SourceAuthority(
                source_id=citation, authority_level=level,
                permitted_claim_types=permitted,
                provenance=f"registry-approved {level.value} source",
            )
    return SourceAuthority(
        source_id=citation,
        authority_level=AuthorityLevel.EXTERNAL_UNTRUSTED,
        restrictions=["not an approved corpus citation"],
    )


def check_authority(claim_type: str, source_id: str) -> AuthorityResult:
    """§10: claim-type → required source classes. A semantically relevant
    but unauthorized source yields AUTHORITY_FAIL (distinct from the
    entailment check, which stays in the EvidenceJudge)."""
    if not source_id:
        return AuthorityResult.NO_EVIDENCE
    # claim-type values arrive lowercase (StrEnum .value) while the table
    # keys are uppercase — normalize before lookup
    required = CLAIM_AUTHORITY_REQUIREMENTS.get(claim_type) or CLAIM_AUTHORITY_REQUIREMENTS.get(
        claim_type.upper()
    )
    if required is None:
        # unknown claim type: conservative — treat as needing primary sources
        required = {AuthorityLevel.QURAN, AuthorityLevel.SAHIH_HADITH,
                    AuthorityLevel.TAFSIR}
    authority = resolve_authority(source_id)
    # never-allowed claim types (causal/guarantee/diagnosis/prediction with
    # empty requirement sets) fail for EVERY source class
    if not required:
        return AuthorityResult.AUTHORITY_FAIL
    if authority.authority_level in required:
        return AuthorityResult.SUPPORTED
    return AuthorityResult.AUTHORITY_FAIL
