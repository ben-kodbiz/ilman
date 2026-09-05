"""Claim extraction (fixme_v3 §4).

Deterministic, sentence-level extraction of RELIGIOUS CLAIMS from model
output: sentences that assert Islamic content (quotes, attributions, doctrine,
rulings, guarantees). Empathy/plain prose is not a claim and needs no evidence.

A claim is whatever sentence carries:
  - an explicit citation (the sentence it sits in), OR
  - religious assertion language ("The Quran says...", "The Prophet taught...",
    "Allah is...", "Islam teaches...", "X is haram", "this dua cures...")

Output feeds the Evidence Judge (entailment); nothing here talks to a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# citation markers (any form the validators accept)
CITATION_MARKER_RE = re.compile(
    r"\[(?:quran|hadith|tafsir|tafsir-en):[^\]]+\]"
    r"|\[(?:[a-z0-9\-]+):(\d+)\](?![\w:])"
    r"|\b(?:quran|hadith|tafsir|tafsir-en):[a-z0-9:\-_]+",
    re.IGNORECASE,
)

# religious assertion language (subset of companion_validator's claim regex,
# tuned for answer text: quotes + attributions + doctrine + guarantees)
ASSERTION_RE = re.compile(
    r"\b(the\s+)?(prophet|quran|qur'?an|allah|rasul|messenger)\b[^.\n]{0,80}"
    r"\b(said|says|stated|states|tells|told|teaches|taught|describes|mentions|"
    r"reminds|reminded|promises|promised|commands|commanded|warns|warned|"
    r"guides|guided|advises|advised|taught)\b"
    r"|\baccording\s+to\s+(the\s+)?(quran|prophet|sunnah|hadith)\b"
    r"|\bislam\s+(?:also\s+)?(teaches|teach|says|states|tells|reminds|"
    r"emphasizes|promises|commands|encourages)\b"
    r"|\ballah\s+(is|alone|does not|will not|never|does)\b[^.\n]{0,80}"
    r"|\b(?:it|this|that)\s+is\s+(?:absolutely\s+|completely\s+)?"
    r"(haram|halal|obligatory|fard|wajib|sunnah|recommended)\b"
    r"|\b(?:guarantees?|cures?|removes?|treats?|heals?)\b"
    r"|\bin\s+islam,?\s+[^.\n]{0,80}",
    re.IGNORECASE,
)

# authoritative connectives (fixme_v3 §16): strong language that requires
# strong evidence
STRONG_CONNECTIVE_RE = re.compile(
    r"\b(this\s+proves|the\s+evidence\s+shows|the\s+evidence\s+points|"
    r"therefore|thus\s+it\s+is\s+clear|it\s+is\s+certain\s+that|"
    r"allah\s+alone|only\s+allah\s+can|this\s+means\s+that|"
    r"guarantees?|definitely|certainly\s+cures?|will\s+certainly)\b",
    re.IGNORECASE,
)


@dataclass
class Claim:
    sentence: str
    has_citation: bool = False
    citations: list[str] = field(default_factory=list)
    is_religious: bool = False
    uses_strong_language: bool = False

    @property
    def needs_evidence(self) -> bool:
        """A sentence must be judged when it asserts religious content —
        cited or not. Empathy prose never needs evidence."""
        return self.is_religious or self.has_citation


def _sentences(text: str) -> list[str]:
    """Quote-aware sentence split: enders inside "..." spans do not end the
    sentence (dua texts are full of 'O Allah!' — splitting there breaks
    quote-entailment)."""
    out: list[str] = []
    buf: list[str] = []
    in_quote = False
    quote_chars = ""
    i = 0
    while i < len(text):
        ch = text[i]
        buf.append(ch)
        if ch in "\"'“”*«»":
            if in_quote and ch == quote_chars:
                in_quote = False
            elif not in_quote and ch in "\"“«":
                in_quote = True
                quote_chars = ch
        elif ch in "!?.\n" and not in_quote:
            # sentence ender (newline only when followed by non-quote prose)
            "".join(buf)
            if ch == "\n":
                out.append("".join(buf).strip())
                buf = []
            else:
                # look ahead: ender followed by space+capital/quote = boundary
                rest = text[i + 1: i + 3]
                if not rest or rest[0] in " \n\t" or (rest[0] and rest[0].isupper()):
                    out.append("".join(buf).strip())
                    buf = []
        i += 1
    if buf:
        out.append("".join(buf).strip())
    return [s for s in out if s]


def extract_claims(text: str) -> list[Claim]:
    claims: list[Claim] = []
    for sentence in _sentences(text):
        if len(sentence) < 12:
            continue
        citations = [m.group(0) for m in CITATION_MARKER_RE.finditer(sentence)]
        is_religious = bool(ASSERTION_RE.search(sentence))
        strong = bool(STRONG_CONNECTIVE_RE.search(sentence))
        claim = Claim(
            sentence=sentence,
            has_citation=bool(citations),
            citations=citations,
            is_religious=is_religious,
            uses_strong_language=strong,
        )
        # bare citations without assertion language still count: the
        # sentence binds content to a source
        if claim.needs_evidence:
            claims.append(claim)
    return claims
