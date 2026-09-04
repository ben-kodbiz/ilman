"""Agent tool layer (agentodo.md §11).

Every tool enforces the Sunni source filter. Tools return structured results;
the model NEVER writes religious content itself, it reasons over what these
tools return and the validators check its claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.memory.store import MemoryStore
from agent.policy.source_policy import SourcePolicy
from agent.tools.quran_refs import normalize_reference
from ingestion.hadith_ingest import HadithStore
from ingestion.quran_ingest import QuranStore


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str = ""


class ToolLayer:
    """Schema-strict tool implementations shared by the agent and benchmarks."""

    def __init__(
        self,
        policy: SourcePolicy,
        store: QuranStore | None = None,
        hadith_store: HadithStore | None = None,
        memory: MemoryStore | None = None,
    ):
        self.policy = policy
        self.store = store
        self.hadith_store = hadith_store
        self.memory = memory

    def get_hadith(self, collection: str, hadith_number: int) -> ToolResult:
        """§13: returns stored text + grading metadata; never fabricates."""
        if self.hadith_store is None:
            return ToolResult(False, error="hadith corpus not ingested")
        row = self.hadith_store.get_hadith(collection, hadith_number)
        if row is None:
            return ToolResult(
                False,
                error=f"hadith {collection}:{hadith_number} not found in approved corpus",
            )
        return ToolResult(True, data=row)

    def search_hadith(self, query: str, collection: str | None = None) -> ToolResult:
        if self.hadith_store is None:
            return ToolResult(False, error="hadith corpus not ingested")
        hits = self.hadith_store.search_fts(query, source_id=collection, limit=8)
        return ToolResult(True, data={"query": query, "results": hits})

    def save_study_note(self, note: str, citation_id: str | None = None) -> ToolResult:
        """§15: study memory; stores content only, never chain-of-thought."""
        if self.memory is None:
            return ToolResult(False, error="memory not configured")
        if not note.strip():
            return ToolResult(False, error="note is empty")
        note_id = self.memory.save_note(note.strip(), citation_id)
        return ToolResult(True, data={"note_id": note_id, "saved": True})

    def get_study_history(self, limit: int = 10) -> ToolResult:
        if self.memory is None:
            return ToolResult(False, error="memory not configured")
        history = self.memory.history(limit=min(max(limit, 1), 50))
        notes = self.memory.notes(limit=min(max(limit, 1), 50))
        return ToolResult(True, data={"history": history, "notes": notes})

    def get_ayah(self, surah: int, ayah: int) -> ToolResult:
        if not isinstance(surah, int) or not isinstance(ayah, int):
            return ToolResult(False, error="surah and ayah must be integers")
        if self.store is None:
            return ToolResult(False, error="corpus not ingested (run ingestion first)")
        row = self.store.get_ayah(surah, ayah)
        if row is None:
            return ToolResult(False, error=f"ayah {surah}:{ayah} not found in approved corpus")
        return ToolResult(True, data=row)

    def verify_quran_reference(self, reference: str) -> ToolResult:
        """§14: deterministic normalization, never model-generated."""
        try:
            ref = normalize_reference(reference)
        except ValueError as e:
            return ToolResult(False, error=str(e))
        return self.get_ayah(ref["surah"], ref["ayah"])

    def get_source_metadata(self, source_id: str) -> ToolResult:
        if self.policy.must_not_retrieve(source_id):
            return ToolResult(False, error=f"source '{source_id}' is excluded and can never be retrieved")
        try:
            record = self.policy.registry.get(source_id)
        except KeyError:
            return ToolResult(False, error=f"source '{source_id}' not in approved registry")
        if not self.policy.retrieval_filter(record):
            return ToolResult(False, error=f"source '{source_id}' fails retrieval filter")
        return ToolResult(True, data=record.to_dict())


# Tool JSON schemas exposed to models (OpenAI function format, §11).
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_ayah",
            "description": "Get a Qur'an ayah (Arabic + translation) by surah and ayah number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "surah": {"type": "integer", "description": "Surah number 1-114"},
                    "ayah": {"type": "integer", "description": "Ayah number"},
                },
                "required": ["surah", "ayah"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_quran_reference",
            "description": "Deterministically verify a Qur'an reference: '2:255', 'Al-Baqarah 255', 'Ayat al-Kursi'.",
            "parameters": {
                "type": "object",
                "properties": {"reference": {"type": "string"}},
                "required": ["reference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_source_metadata",
            "description": "Get registry metadata (title, author, license, verification) for an approved source ID.",
            "parameters": {
                "type": "object",
                "properties": {"source_id": {"type": "string"}},
                "required": ["source_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hadith",
            "description": "Get a hadith by collection and number. Returns Arabic, English, and grading metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "One of: sahih-bukhari, sahih-muslim, sunan-abu-dawud, "
                        "jami-at-tirmidhi, sunan-an-nasai, sunan-ibn-majah",
                    },
                    "hadith_number": {"type": "integer"},
                },
                "required": ["collection", "hadith_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hadith",
            "description": "Full-text search across the six hadith collections (Arabic or English).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "collection": {"type": "string", "description": "Optional: restrict to one collection ID."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_study_note",
            "description": "Save a study note, optionally attached to a citation_id like quran:2:255.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                    "citation_id": {"type": "string", "description": "Optional citation to attach."},
                },
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_study_history",
            "description": "Get the user's recent study history and saved notes.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "How many entries (default 10)."}},
                "required": [],
            },
        },
    },
]

TOOL_IMPLS = {
    "get_ayah": lambda layer, args: layer.get_ayah(**args),
    "verify_quran_reference": lambda layer, args: layer.verify_quran_reference(**args),
    "get_source_metadata": lambda layer, args: layer.get_source_metadata(**args),
    "get_hadith": lambda layer, args: layer.get_hadith(**args),
    "search_hadith": lambda layer, args: layer.search_hadith(**args),
    "save_study_note": lambda layer, args: layer.save_study_note(**args),
    "get_study_history": lambda layer, args: layer.get_study_history(**args),
}


def execute_tool(layer: ToolLayer, name: str, arguments: dict[str, Any]) -> ToolResult:
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return ToolResult(False, error=f"unknown tool '{name}'")
    try:
        return impl(layer, arguments)
    except TypeError as e:
        return ToolResult(False, error=f"bad arguments for '{name}': {e}")
