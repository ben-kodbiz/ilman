"""Deterministic Qur'an reference handling (agentodo.md §14).

Accepts forms like "2:255", "Al-Baqarah 255", "Ayat al-Kursi" and normalizes
to {"surah": int, "ayah": int}. The model NEVER invents verse text or
references; this module is the only way references enter the system.
"""

from __future__ import annotations

import re

# Surah lookup by common transliterations. Deliberately minimal; extend as the
# corpus grows. Numbers are canonical and deterministic.
SURAH_NAMES: dict[str, int] = {
    "al-fatihah": 1,
    "al-baqarah": 2,
    "al-baqara": 2,
    "baqarah": 2,
    "al-imran": 3,
    "an-nisa": 4,
    "al-ma'idah": 5,
    "al-kahf": 18,
    "ta ha": 20,
    "al-furqan": 25,
    "ash-shu'ara": 26,
    "an-naml": 27,
    "ar-rahman": 55,
    "al-waqi'ah": 56,
    "al-mulk": 67,
    "al-ikhlas": 112,
    "al-falaq": 113,
    "an-nas": 114,
    "ikhlas": 112,
    "falaq": 113,
    "nas": 114,
    "fatihah": 1,
}

# Well-known ayah aliases, for disambiguation only (never for verse text).
KNOWN_AYAH_ALIASES: dict[str, tuple[int, int]] = {
    "ayat al-kursi": (2, 255),
    "ayatul kursi": (2, 255),
    "throne verse": (2, 255),
    "al-fatihah opening": (1, 1),
}

VALID_REF = re.compile(r"^(?P<surah>\d{1,3}):(?P<ayah>\d{1,3})$")


class QuranRefParseError(ValueError):
    pass


def normalize_reference(text: str) -> dict[str, int]:
    """Normalize any accepted reference form to {"surah": n, "ayah": n}."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)

    # 1. plain "2:255" / "2 : 255"
    m = re.match(r"^(\d{1,3})\s*[:.]\s*(\d{1,3})$", cleaned)
    if m:
        return _make(int(m.group(1)), int(m.group(2)))

    # 2. "surah 2:255", "qur'an 2:255", "verse 2:255"
    m = re.search(r"(\d{1,3})\s*[:.]\s*(\d{1,3})", cleaned)
    if m and _worded_prefix(cleaned):
        return _make(int(m.group(1)), int(m.group(2)))

    # 3. "Al-Baqarah 255" / "Surah Al-Baqarah, 255" / "Ikhlas 1"
    m = re.match(r"^(?:surah |qur'an |quran )?([a-z' \-]+?)[ ,]*(\d{1,3})$", cleaned)
    if m:
        name = m.group(1).strip()
        if name in SURAH_NAMES:
            return _make(SURAH_NAMES[name], int(m.group(2)))

    # 4. alias "Ayat al-Kursi"
    if cleaned in KNOWN_AYAH_ALIASES:
        s, a = KNOWN_AYAH_ALIASES[cleaned]
        return _make(s, a)

    raise QuranRefParseError(f"unrecognized Qur'an reference: '{text}'")


def _make(surah: int, ayah: int) -> dict[str, int]:
    if not 1 <= surah <= 114:
        raise QuranRefParseError(f"surah {surah} out of range 1-114")
    if not 1 <= ayah <= 286:
        raise QuranRefParseError(f"ayah {ayah} out of range")
    return {"surah": surah, "ayah": ayah}


def _worded_prefix(text: str) -> bool:
    return bool(re.search(r"surah|verse|qur|ayah|tafsir|ayat", text))


def extract_references(text: str) -> list[dict[str, int]]:
    """Extract all numeric refs from free text, deterministic order of appearance."""
    refs: list[dict[str, int]] = []
    for match in re.finditer(r"\b(\d{1,3})\s*[:.]\s*(\d{1,3})\b", text):
        s, a = int(match.group(1)), int(match.group(2))
        if 1 <= s <= 114:
            refs.append({"surah": s, "ayah": a})
    return refs
