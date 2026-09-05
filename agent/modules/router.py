"""Module architecture (enhance_v1 §12-16) — adapters over EXISTING
functionality (§3.1), not a rewrite.

Knowledge modules (§12-13): quran / hadith / tafsir wrap the existing stores
+ retrieval legs; fiqh/seerah are registered but disabled until corpora are
ingested (registry-pending). Knowledge modules provide evidence + provenance;
they never generate answers and never bypass validation.

Capability modules (§14): dua / study / reflection adapt the existing
companion-policy flows (dua candidates, study memory, reflection mode).

ModuleRouter (§16): selects the MINIMUM module set from the intent — a
companion emotional query does NOT invoke the hadith module (§17); a Qur'an
question does not invoke the dua capability.

Security boundary (§34): every module path returns structured results INTO
the core pipeline — none can disable validation, mutate frozen packs, or
mark unsupported claims supported. Enforced structurally: modules never see
the validators or the frozen pack mutators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any

from agent.companion.intent import classify_companion
from agent.core.query_planner import plan_query


class ModuleKind(StrEnum):
    KNOWLEDGE = auto()
    CAPABILITY = auto()


@dataclass
class KnowledgeProvider:
    """§12 interface over an existing corpus store + retrieval leg."""

    name: str
    version: str = "1.0"
    kind: ModuleKind = ModuleKind.KNOWLEDGE
    enabled: bool = True
    # adapter deps (injected; never imported at module scope to keep the
    # boundary structural)
    search_fn: Any = None  # (query, filters) -> list[dict passage rows]

    def search(self, query: str, filters: dict | None = None) -> list[dict]:
        if not self.enabled or self.search_fn is None:
            return []
        return self.search_fn(query, filters or {})

    def validate_source(self, citation_id: str) -> dict:
        """Delegate to the core authority layer (§13: modules do not bypass
        source authority policy)."""
        from agent.evidence.authority import resolve_authority

        a = resolve_authority(citation_id)
        return {
            "authority_level": a.authority_level.value,
            "permitted_claim_types": sorted(a.permitted_claim_types),
            "restrictions": a.restrictions,
        }


@dataclass
class Capability:
    """§14 interface. plan() is structured; execute() returns structured
    results to the core — capabilities never emit user-facing text
    directly nor bypass validation."""

    name: str
    version: str = "1.0"
    kind: ModuleKind = ModuleKind.CAPABILITY
    enabled: bool = True

    def can_handle(self, context: dict) -> bool:
        raise NotImplementedError

    def plan(self, context: dict) -> dict:
        raise NotImplementedError

    def execute(self, plan: dict) -> dict:
        raise NotImplementedError


@dataclass
class ModuleRegistry:
    """Enabled modules registry; independently enableable (§12)."""

    knowledge: dict[str, KnowledgeProvider] = field(default_factory=dict)
    capabilities: dict[str, Capability] = field(default_factory=dict)

    def enabled_knowledge(self) -> list[KnowledgeProvider]:
        return [m for m in self.knowledge.values() if m.enabled]

    def enabled_capabilities(self) -> list[Capability]:
        return [m for m in self.capabilities.values() if m.enabled]

    def to_dict(self) -> dict:
        return {
            "knowledge": {k: {"enabled": m.enabled, "version": m.version}
                          for k, m in self.knowledge.items()},
            "capabilities": {k: {"enabled": m.enabled, "version": m.version}
                             for k, m in self.capabilities.items()},
        }


# ------------------------------------------------------------- routing
class ModuleRouter:
    """§16: intent → minimum module set. Deterministic."""

    def __init__(self, registry: ModuleRegistry):
        self.registry = registry

    def select(self, message: str, companion_intent=None) -> dict:
        """Returns {knowledge: [names], capabilities: [names], rationale}.

        §17 preserved: emotional companion statements route to NO knowledge
        modules; explicit Islamic questions route to the relevant knowledge
        modules; dua requests route to the dua capability + hadith knowledge.
        """
        ci = companion_intent or classify_companion(message)
        plan = plan_query(message, ci.intent)
        knowledge: list[str] = []
        capabilities: list[str] = []
        rationale: list[str] = []

        needs_rag = ci.needs_islamic_guidance
        if not needs_rag:
            if ci.emotion or ci.intent == "emotional_support":
                rationale.append("companion emotional query: no knowledge modules (§17)")
                capabilities.append("companion")
            else:
                rationale.append("normal chat: no knowledge modules")
                capabilities.append("companion")
            return {"knowledge": [], "capabilities": capabilities, "rationale": rationale}

        # requested-object-driven minimum set (§16)
        obj = plan.requested_object
        prefs = plan.source_preference
        if obj == "specific_dua" or ci.intent == "dua_request":
            capabilities.append("dua")
            if "hadith" in prefs:
                knowledge.append("hadith")
            if "quran" in prefs:
                knowledge.append("quran")
            rationale.append("dua request: dua capability + hadith/quran knowledge")
        elif obj == "verse" or ci.intent in ("quran_question", "quran_request"):
            knowledge.append("quran")
            if "tafsir" in prefs:
                knowledge.append("tafsir")
            rationale.append("quran question: quran (+tafsir when asked)")
        elif obj == "hadith" or ci.intent == "hadith_question":
            knowledge.append("hadith")
            rationale.append("hadith question: hadith module")
        elif obj == "explanation":
            knowledge.extend(["tafsir", "quran"])
            rationale.append("explanation request: tafsir + quran")
        else:  # generic islamic_question
            knowledge.extend(["quran", "hadith"])
            rationale.append("islamic question: quran + hadith")

        # filter to enabled modules only (fiqh/seerah disabled until ingested)
        knowledge = [k for k in knowledge
                      if k in self.registry.knowledge and self.registry.knowledge[k].enabled]
        return {"knowledge": knowledge, "capabilities": capabilities,
                "rationale": rationale}


def build_default_registry(stores: dict | None = None) -> ModuleRegistry:
    """Wire adapters over the existing corpus stores (§3.1)."""

    def _quran_search(query: str, filters: dict) -> list[dict]:
        from ingestion.quran_ingest import QuranStore

        store = (stores or {}).get("quran") or QuranStore()
        lang = "en" if not any("\u0600" <= ch <= "\u06FF" for ch in query) else None
        if lang:
            return [
                {**r, "translation": r.get("translation", "")}
                for r in store.search_translation_fts(query, lang=lang,
                                                      limit=filters.get("limit", 6))
            ]
        return store.search_fts(query, limit=filters.get("limit", 6))

    def _hadith_search(query: str, filters: dict) -> list[dict]:
        from ingestion.hadith_ingest import HadithStore

        store = (stores or {}).get("hadith") or HadithStore()
        return store.search_fts(query, source_id=filters.get("collection"),
                                limit=filters.get("limit", 6))

    def _tafsir_search(query: str, filters: dict) -> list[dict]:
        from ingestion.tafsir_en_ingest import TafsirEnStore

        store = (stores or {}).get("tafsir_en") or TafsirEnStore()
        return store.search_fts(query, limit=filters.get("limit", 6))

    registry = ModuleRegistry(
        knowledge={
            "quran": KnowledgeProvider(name="quran", search_fn=_quran_search),
            "hadith": KnowledgeProvider(name="hadith", search_fn=_hadith_search),
            "tafsir": KnowledgeProvider(name="tafsir", search_fn=_tafsir_search),
            # registered but disabled until corpora are ingested (§12)
            "fiqh": KnowledgeProvider(name="fiqh", enabled=False, search_fn=None),
            "seerah": KnowledgeProvider(name="seerah", enabled=False, search_fn=None),
        },
        capabilities={
            "companion": Capability(name="companion"),
            "dua": Capability(name="dua"),
            "study": Capability(name="study"),
            "reflection": Capability(name="reflection"),
        },
    )
    return registry
