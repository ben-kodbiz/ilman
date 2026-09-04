"""Arabic search normalization (agentodo.md §7: Arabic/text cleanup).

The canonical Uthmani text is stored EXACTLY as ingested and always displayed
from `quran.arabic`. For *matching only* we fold it into a bare search form:
strip all diacritics (classical harakat + Quranic annotation codepoints),
fold alef/hamza variants onto plain alef, taa marbuta -> ha, alef maqsura -> ya.
Queries go through the same function, so 'قل هو الله أحد' finds
'قُلۡ هُوَ ٱللَّهُ أَحَدٌ'.
"""

from __future__ import annotations

import re

# Ranges stripped entirely (marks, not letters).
_STRIP_RANGES = [
    (0x0610, 0x061A),  # Arabic signs
    (0x064B, 0x065F),  # harakat incl. superscript alef/hamza (maddah etc.)
    (0x0670, 0x0670),  # superscript alef
    (0x06D6, 0x06DC),  # small high signs
    (0x06DF, 0x06E4),  # small signs
    (0x06E5, 0x06E6),  # small waw / ya
    (0x06E7, 0x06E8),  # small ya/yeh
    (0x06EA, 0x06ED),  # low marks
    (0x08D3, 0x08FF),  # Arabic extended-A marks
    (0x0640, 0x0640),  # tatweel
]

# Character folds (search form only; never applied to stored/display text).
_FOLD_MAP = {
    0x0671: 0x0627,  # ٱ alef wasla -> ا
    0x0622: 0x0627,  # آ alef madda -> ا
    0x0623: 0x0627,  # أ alef hamza -> ا
    0x0625: 0x0627,  # إ alef hamza below -> ا
    0x0629: 0x0647,  # ة taa marbuta -> ه
    0x0649: 0x064A,  # ى alef maqsura -> ي
}

_STRIP_RE = None


def _build_strip_re() -> re.Pattern:
    parts = []
    for lo, hi in _STRIP_RANGES:
        if lo == hi:
            parts.append(f"\\u{lo:04X}")
        else:
            parts.append(f"\\u{lo:04X}-\\u{hi:04X}")
    return re.compile("[" + "".join(parts) + "]+")


def search_form(text: str) -> str:
    """Bare search form: no diacritics, folded variants, single spaces."""
    global _STRIP_RE
    if _STRIP_RE is None:
        _STRIP_RE = _build_strip_re()
    out = _STRIP_RE.sub("", text)
    out = "".join(chr(_FOLD_MAP.get(ord(c), ord(c))) for c in out)
    return " ".join(out.split())
