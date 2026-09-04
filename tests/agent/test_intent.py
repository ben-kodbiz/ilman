from __future__ import annotations

from agent.core.intent import classify


class TestIntents:
    def test_quran_lookup_numeric(self):
        r = classify("What does 2:255 say about Allah?")
        assert r.intent == "quran_lookup"
        assert r.quran_refs == [{"surah": 2, "ayah": 255}]

    def test_quran_lookup_surah_only(self):
        r = classify("Tell me about Surah 112")
        assert r.intent == "quran_lookup"
        assert r.quran_refs[0]["surah"] == 112

    def test_hadith_lookup_bukhari_number(self):
        r = classify("What is hadith no. 1 in Bukhari?")
        assert r.intent == "hadith_lookup"
        assert r.hadith_refs == [{"collection": "sahih-bukhari", "number": 1}]

    def test_hadith_lookup_muslim(self):
        r = classify("Sahih Muslim number 33 check please")
        assert r.hadith_refs[0]["collection"] == "sahih-muslim"
        assert r.hadith_refs[0]["number"] == 33

    def test_hadith_search_phrase(self):
        r = classify("Find a hadith about the Prophet on patience")
        assert r.intent == "hadith_search"

    def test_quran_search_arabic(self):
        r = classify("قل هو الله أحد")
        assert r.arabic
        assert r.intent in ("quran_search", "question")

    def test_general_question(self):
        r = classify("What does Islam say about honesty in trade?")
        assert r.intent == "question"
        assert r.is_question

    def test_history_intent(self):
        r = classify("show my recent history")
        assert r.intent == "history"

    def test_routing_task_class(self):
        assert classify("What is 2:255?").routed_task_class == "simple_rag"
        assert classify("What does Islam say about patience?").routed_task_class == "complex_rag"


class TestRefsExtraction:
    def test_multiple_quran_refs(self):
        r = classify("Compare 112:1 and 2:255 for me")
        pairs = {(x["surah"], x["ayah"]) for x in r.quran_refs}
        assert pairs == {(112, 1), (2, 255)}

    def test_alias_resolves_to_reference(self):
        r = classify("What is Ayat al-Kursi?")
        # aliases now resolve deterministically (§14) — the improvement the
        # vector-leg regression exposed
        assert r.intent == "quran_lookup"
        assert r.quran_refs == [{"surah": 2, "ayah": 255}]
