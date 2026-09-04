from __future__ import annotations

import pytest

from agent.memory.store import ConversationMemory, MemoryStore


@pytest.fixture()
def memory(tmp_path):
    return MemoryStore(db_path=tmp_path / "memory.db")


class TestProfile:
    def test_set_get_roundtrip(self, memory):
        memory.set_profile("language", "en")
        assert memory.get_profile("language") == "en"

    def test_default_on_missing(self, memory):
        assert memory.get_profile("nonsense", default="x") == "x"

    def test_overwrite(self, memory):
        memory.set_profile("theme", "light")
        memory.set_profile("theme", "dark")
        assert memory.get_profile("theme") == "dark"


class TestStudyNotes:
    def test_save_and_list(self, memory):
        memory.save_note("Ayat al-Kursi protects at night", citation_id="quran:2:255")
        notes = memory.notes()
        assert notes and notes[0]["note"].startswith("Ayat al-Kursi")
        assert notes[0]["citation_id"] == "quran:2:255"

    def test_filter_by_citation(self, memory):
        memory.save_note("about 112", citation_id="quran:112:1")
        memory.save_note("about 2", citation_id="quran:2:255")
        only_112 = memory.notes(citation_id="quran:112:1")
        assert len(only_112) == 1


class TestHistory:
    def test_record_and_retrieve(self, memory):
        memory.record_query("What is Ayat al-Kursi?", "quran_lookup", ["quran:2:255"])
        history = memory.history()
        assert history[0]["query"] == "What is Ayat al-Kursi?"
        assert history[0]["intent"] == "quran_lookup"
        assert history[0]["citations"] == ["quran:2:255"]


class TestBookmarks:
    def test_bookmark_dedup(self, memory):
        memory.bookmark("quran:112:1")
        memory.bookmark("quran:112:1")
        assert len(memory.bookmarks()) == 1


class TestSeparation:
    def test_memory_db_is_not_the_knowledge_db(self, tmp_path):
        """§15: user memory must never merge with authoritative knowledge."""
        mem = MemoryStore(db_path=tmp_path / "memory.db")
        mem.save_note("personal thought about a verse")
        import sqlite3
        con = sqlite3.connect(tmp_path / "memory.db")
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        con.close()
        assert "quran" not in tables and "hadith" not in tables
        assert "study_notes" in tables


class TestConversationMemory:
    def test_cap(self):
        conv = ConversationMemory(max_turns=4)
        for i in range(10):
            conv.add("user", f"m{i}")
        assert len(conv.turns) == 4
        assert conv.turns[-1].content == "m9"

    def test_as_messages(self):
        conv = ConversationMemory()
        conv.add("user", "hi")
        conv.add("assistant", "salam")
        assert conv.as_messages()[0] == {"role": "user", "content": "hi"}

    def test_clear(self):
        conv = ConversationMemory()
        conv.add("user", "x")
        conv.clear()
        assert conv.turns == []
