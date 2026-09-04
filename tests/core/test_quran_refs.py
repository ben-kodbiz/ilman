from __future__ import annotations

import pytest

from agent.tools.quran_refs import (
    QuranRefParseError,
    extract_references,
    normalize_reference,
)


class TestNormalize:
    @pytest.mark.parametrize("text,expected", [
        ("2:255", {"surah": 2, "ayah": 255}),
        ("2 : 255", {"surah": 2, "ayah": 255}),
        ("112:1", {"surah": 112, "ayah": 1}),
        ("Surah 112:1", {"surah": 112, "ayah": 1}),
        ("Qur'an 2:255", {"surah": 2, "ayah": 255}),
        ("Al-Baqarah 255", {"surah": 2, "ayah": 255}),
        ("surah al-ikhlas 1", {"surah": 112, "ayah": 1}),
        ("Ayat al-Kursi", {"surah": 2, "ayah": 255}),
        ("AYATUL KURSI", {"surah": 2, "ayah": 255}),
    ])
    def test_accepted_forms(self, text, expected):
        assert normalize_reference(text) == expected

    @pytest.mark.parametrize("text", [
        "3:300",        # ayah beyond any surah's length guard
        "115:1",        # surah out of range
        "totally unknown",
        "",
        "4:79:3",
    ])
    def test_rejected_forms(self, text):
        with pytest.raises(QuranRefParseError):
            normalize_reference(text)

    def test_ayah_range_guard(self):
        with pytest.raises(QuranRefParseError):
            normalize_reference("1:287")  # al-Fatihah has 7 ayahs; guard is generic 286
        # generic guard allows <=286; per-surah exact bounds come with the
        # full Qur'an dataset in Phase 2.


class TestExtract:
    def test_extracts_in_order(self):
        text = "See 2:255 and also 112:1, then revisit 2:255."
        refs = extract_references(text)
        assert refs == [{"surah": 2, "ayah": 255}, {"surah": 112, "ayah": 1}, {"surah": 2, "ayah": 255}]

    def test_ignores_non_refs(self):
        assert extract_references("chapter 12 verse 34 has no colon ref") == []

    def test_ignores_out_of_range_surah(self):
        assert extract_references("999:1") == []
