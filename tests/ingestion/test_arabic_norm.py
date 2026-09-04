from __future__ import annotations

from ingestion.arabic_norm import search_form


class TestSearchForm:
    def test_strips_classical_harakat(self):
        assert search_form("قُلْ هُوَ") == "قل هو"

    def test_strips_quranic_marks_and_folds_alef_wasla(self):
        # 112:1 in Uthmani: قُلۡ هُوَ ٱللَّهُ أَحَدٌ
        uthmani = "قُلۡ هُوَ ٱللَّهُ أَحَدٌ"
        assert search_form(uthmani) == "قل هو الله احد"

    def test_folds_hamza_alefs(self):
        # superscript alef (U+0670) is a mark: stripped, not unfolded
        assert search_form("أَحَدٌ إِلَٰهٍ آيَة") == "احد اله ايه"

    def test_taa_marbuta_and_maqsura(self):
        assert search_form("صَلَاةٰ مُوسَىٰ") == "صلاه موسي"

    def test_search_form_always_folds_plain_text(self):
        # search_form is for matching only; folding applies to any input
        assert search_form("قل هو الله أحد") == "قل هو الله احد"

    def test_idempotent(self):
        once = search_form("قُلۡ هُوَ ٱللَّهُ أَحَدٌ")
        assert search_form(once) == once

    def test_whitespace_collapse(self):
        assert search_form("  a\tb \n c ") == "a b c"
