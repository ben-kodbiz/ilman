"""HTTP API for the study assistant (agentodo.md §23 Phase A, §26 Phase 8 seed).

Serves ONLY registry-backed endpoints: search, deterministic ayah lookup,
source metadata, and the grounded answer pipeline. Every response carries
provenance (citation_id + source_id + tier). The server never forwards raw
model output without citation validation.

Local-first: binds 127.0.0.1 by default, no telemetry, no cloud calls.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.companion.engine import CompanionEngine
from agent.companion.memory import CompanionMemory
from agent.context.builder import ContextBuilder
from agent.core.agent import AgentOrchestrator
from agent.core.config import load_config
from agent.core.harness import CompanionHarness
from agent.core.model import ModelRouter
from agent.memory.router import MemoryRouter
from agent.policy.companion_policy import CompanionPolicyEngine
from agent.policy.source_policy import SourcePolicy, SourceRegistry
from agent.state.manager import StateManager
from agent.tools.layer import TOOL_SCHEMAS, ToolLayer, execute_tool
from agent.validators.companion_validator import ResponseValidator
from agent.validators.pipeline import (
    UNVERIFIABLE_NOTICE,
    CitationValidator,
    ResponsePipeline,
)
from ingestion.hadith_ingest import KUTUB_AL_SITTAH, HadithStore
from ingestion.quran_ingest import DEFAULT_DB, QuranStore
from ingestion.tafsir_en_ingest import TafsirEnStore
from ingestion.tafsir_ingest import TafsirStore
from retrieval.hybrid import RetrievalOrchestrator
from retrieval.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "apps" / "web"

app = FastAPI(title="Huurs study API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORE = QuranStore()
if not Path(DEFAULT_DB).exists() or STORE.ayah_count_total() == 0:
    raise RuntimeError(
        "knowledge DB not ingested — run: uv run python -c "
        "'from ingestion.quran_ingest import QuranIngestor; QuranIngestor().ingest()'"
    )
POLICY = SourcePolicy(SourceRegistry.load())
HADITH_STORE = HadithStore()
try:
    HADITH_COUNT = HADITH_STORE.hadith_count()
except Exception:
    HADITH_COUNT = 0
TAFSIR_STORE = TafsirStore()
try:
    TAFSIR_COUNT = TAFSIR_STORE.tafsir_count()
except Exception:
    TAFSIR_COUNT = 0
TAFSIR_EN_STORE = TafsirEnStore()
try:
    TAFSIR_EN_COUNT = TAFSIR_EN_STORE.chunk_count()
except Exception:
    TAFSIR_EN_COUNT = 0
try:
    VECTOR_STORE = VectorStore()
    _ = VECTOR_STORE.size  # loads cache; 0 when missing
except Exception:
    VECTOR_STORE = None
ORCHESTRATOR = RetrievalOrchestrator(
    STORE,
    hadith_store=HADITH_STORE if HADITH_COUNT else None,
    tafsir_store=TAFSIR_STORE if TAFSIR_COUNT else None,
    tafsir_en_store=TAFSIR_EN_STORE if TAFSIR_EN_COUNT else None,
    vector_store=VECTOR_STORE if (VECTOR_STORE and VECTOR_STORE.size) else None,
)
MEMORY = CompanionMemory()
TOOLS = ToolLayer(
    POLICY, store=STORE,
    hadith_store=HADITH_STORE if HADITH_COUNT else None,
    memory=MEMORY,
)
try:
    ROUTER = ModelRouter(load_config())
    PIPELINE = ResponsePipeline(ROUTER)
    AGENT = AgentOrchestrator(ROUTER, ORCHESTRATOR, TOOLS, memory=MEMORY)
except Exception:
    ROUTER = None
    PIPELINE = None  # model backend absent: search still works, answers refuse
    AGENT = None
try:
    COMPANION_V1 = CompanionEngine(ROUTER, ORCHESTRATOR, TOOLS, memory=MEMORY) if ROUTER else None
except Exception:
    COMPANION_V1 = None
try:
    DEV_MEMORY_ROUTER = MemoryRouter(MEMORY)
    COMPANION = (
        CompanionHarness(
            ROUTER, retrieval=ORCHESTRATOR, memory_router=DEV_MEMORY_ROUTER,
            states=StateManager(),
            policy_engine=CompanionPolicyEngine(),
            context_builder=ContextBuilder(),
            validator=ResponseValidator(),
            citation_validator=CitationValidator(),
            model_label="v2",
        )
        if ROUTER
        else None
    )
except Exception:
    COMPANION = None
COMPANION_STATES = StateManager()


class AnswerRequest(BaseModel):
    question: str
    task_class: str = "complex_rag"
    limit: int = 6
    mode: str = "agent"  # "agent" (tool loop) or "pipeline" (single-shot)


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict


@app.get("/api/v1/health")
def health() -> dict:
    return {
        "status": "ok",
        "ayahs": STORE.ayah_count_total(),
        "hadiths": HADITH_COUNT,
        "tafsir_entries": TAFSIR_COUNT,
        "classic_tafsir_chunks": TAFSIR_EN_COUNT,
        "vector_index_size": VECTOR_STORE.size if VECTOR_STORE else 0,
        "answer_backend": "model" if PIPELINE else "unavailable",
    }


@app.get("/api/v1/ayah/{surah}/{ayah}")
def get_ayah(surah: int, ayah: int, lang: str = "en") -> dict:
    row = STORE.get_ayah(surah, ayah, lang=lang if STORE.translation_count(lang) else None)
    if row is None:
        raise HTTPException(404, f"ayah {surah}:{ayah} not found")
    return row


@app.get("/api/v1/ayah")
def get_ayah_by_reference(ref: str) -> dict:
    row = STORE.get_by_reference(ref)
    if row is None:
        raise HTTPException(404, f"could not resolve reference '{ref}'")
    return row


@app.get("/api/v1/search")
def search(q: str, limit: int = 8) -> dict:
    passages = ORCHESTRATOR.search(q, limit=limit)
    return {
        "query": q,
        "results": [
            {
                "citation_id": p.citation_id,
                "surah": p.surah,
                "ayah": p.ayah,
                "arabic": p.arabic,
                "translation": p.translation,
                "source_id": p.source_id,
                "tier": p.tier,
                "collection": p.collection or None,
                "hadithnumber": p.hadithnumber,
                "grades": p.grades if p.citation_id.startswith("hadith:") else None,
            }
            for p in passages
        ],
    }


@app.get("/api/v1/hadith/collections")
def hadith_collections() -> dict:
    return {"collections": HADITH_STORE.collections()}


@app.get("/api/v1/hadith/{collection}/{hadith_number}")
def get_hadith(collection: str, hadith_number: int) -> dict:
    if collection not in KUTUB_AL_SITTAH:
        raise HTTPException(404, f"unknown collection '{collection}'")
    row = HADITH_STORE.get_hadith(collection, hadith_number)
    if row is None:
        raise HTTPException(404, f"hadith {collection}:{hadith_number} not found")
    return row


@app.get("/api/v1/hadith/search")
def hadith_search(q: str, collection: str | None = None, limit: int = 12) -> dict:
    results = HADITH_STORE.search_fts(q, source_id=collection, limit=limit)
    return {"query": q, "results": results}


@app.get("/api/v1/tafsir/{surah}/{ayah}")
def get_tafsir(surah: int, ayah: int) -> dict:
    row = TAFSIR_STORE.get_tafsir(surah, ayah)
    if row is None:
        raise HTTPException(404, f"no tafsir for {surah}:{ayah}")
    return row


@app.get("/api/v1/tafsir/search")
def tafsir_search(q: str, limit: int = 10) -> dict:
    return {"query": q, "results": TAFSIR_STORE.search_fts(q, limit=limit)}


@app.get("/api/v1/tafsir/classic/{surah}/{ayah}")
def classic_tafsir_ayah(surah: int, ayah: int) -> dict:
    """All classic English tafsir commentary on one ayah (Sa'di, Ibn Kathir, Qurtubi)."""
    chunks = TAFSIR_EN_STORE.get_for_ayah(surah, ayah)
    return {"surah": surah, "ayah": ayah, "commentary": chunks}


@app.get("/api/v1/tafsir/classic/search")
def classic_tafsir_search(q: str, source_id: str | None = None, limit: int = 10) -> dict:
    return {"query": q, "results": TAFSIR_EN_STORE.search_fts(q, source_id=source_id, limit=limit)}


@app.get("/api/v1/sources/{source_id}")
def source_metadata(source_id: str) -> dict:
    result = TOOLS.get_source_metadata(source_id)
    if not result.ok:
        raise HTTPException(404, result.error)
    return result.data


@app.get("/api/v1/tools")
def tools() -> list[dict]:
    return TOOL_SCHEMAS


@app.post("/api/v1/tools/call")
def call_tool(req: ToolCallRequest) -> dict:
    result = execute_tool(TOOLS, req.name, req.arguments)
    if not result.ok:
        return {"ok": False, "error": result.error}
    return {"ok": True, "data": result.data}


class CompanionRequest(BaseModel):
    session_id: str = "default"
    message: str


@app.post("/api/v1/companion")
def companion(req: CompanionRequest) -> dict:
    """Context-aware companion mode (fixme_v2 harness): state/policy/context/
    memory/safety/citation-validation. The public response is the §33 shape:
    answer, mode, intent, citations — no internal debug trace."""
    if COMPANION is None:
        raise HTTPException(503, "model backend unavailable")
    response = COMPANION.respond(req.session_id, req.message)
    data = response.to_dict()
    # §33: strip developer trace from the public client payload
    data.pop("debug_trace", None)
    return data


@app.post("/api/v1/companion/v1")
def companion_v1(req: CompanionRequest) -> dict:
    """Legacy v1 companion engine — kept for comparison during the v2 bake."""
    if COMPANION_V1 is None:
        raise HTTPException(503, "model backend unavailable")
    return COMPANION_V1.respond(req.session_id, req.message).to_dict()


@app.get("/api/v1/companion/state/{session_id}")
def companion_state(session_id: str) -> dict:
    machine = COMPANION_STATES.machine(session_id, create=False)
    return {
        "active": machine is not None,
        "state": machine.state.to_dict() if machine else None,
    }


@app.delete("/api/v1/companion/state/{session_id}")
def companion_clear(session_id: str) -> dict:
    """'Clear conversation' control (§25)."""
    COMPANION_STATES.drop(session_id)
    return {"cleared": True}


# ------------------------------------------------------------ chat logs
# owner-approved troubleshooting capture (local-only, gitignored). Sensitive
# (crisis) turns require explicit opt-in.
@app.get("/api/v1/logs/sessions")
def log_sessions() -> dict:
    from agent.companion.logging import CompanionLogger

    sessions = []
    for s in CompanionLogger.read_sessions():
        turns = CompanionLogger.read_turns(s)
        sessions.append({
            "session_id": s,
            "turns": len(turns),
            "sensitive_turns": sum(1 for t in turns if t.get("sensitive")),
        })
    return {"sessions": sessions}


@app.get("/api/v1/logs/stats")
def log_stats() -> dict:
    from agent.companion.logging import CompanionLogger

    return CompanionLogger.stats()


@app.get("/api/v1/logs/{session_id}")
def log_read(session_id: str, include_sensitive: bool = False) -> dict:
    from agent.companion.logging import CompanionLogger

    turns = CompanionLogger.read_turns(
        session_id, include_sensitive=include_sensitive
    )
    return {"session_id": session_id, "turns": turns}


@app.delete("/api/v1/logs/{session_id}")
def log_redact_session(session_id: str) -> dict:
    """§25 control: overwrite this session's log content with [REDACTED],
    preserving the metadata rows (append-only audit trail stays intact)."""
    import glob
    import json as _json
    import re as _re

    from agent.companion.logging import LOG_DIR

    safe = _re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:80]
    rewritten = 0
    for path in glob.glob(str(LOG_DIR / f"{safe}-*.jsonl")):
        p = Path(path)
        lines = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = _json.loads(line)
            if rec.get("session_id") == session_id:
                rec["user"]["text"] = "[REDACTED]"
                rec["companion"]["text"] = "[REDACTED]"
                rec["redacted"] = True
                rewritten += 1
            lines.append(_json.dumps(rec, ensure_ascii=False))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"session_id": session_id, "records_redacted": rewritten}


# ---------------------------------------------------------------- ratings
class RatingRequest(BaseModel):
    session_id: str
    turn: int
    rating: str  # "up" | "down"
    answer_excerpt: str = ""


@app.post("/api/v1/ratings")
def submit_rating(req: RatingRequest) -> dict:
    """Thumbs up/down on a companion answer; stored locally for pipeline
    analysis (down-rated turns are the troubleshooting signal)."""
    from agent.companion.ratings import RatingError, enrich_from_chat_log, record_rating

    try:
        pipeline = enrich_from_chat_log(req.session_id, req.turn)
        path = record_rating(
            req.session_id, req.turn, req.rating,
            answer_excerpt=req.answer_excerpt, pipeline=pipeline,
        )
        return {"ok": True, "stored": str(path)}
    except RatingError as e:
        raise HTTPException(400, str(e))


@app.get("/api/v1/ratings/analysis")
def ratings_analysis_endpoint() -> dict:
    """Aggregate of rated turns — down-rated breakdown for enhancement."""
    from agent.companion.ratings import ratings_analysis

    return ratings_analysis()


@app.get("/api/v1/memories")
def memories_view() -> dict:
    """'View memories' control (§25)."""
    return MEMORY.memory_view()


@app.delete("/api/v1/memories")
def memories_clear() -> dict:
    """'Clear memories' control (§25)."""
    return MEMORY.clear_all()


@app.post("/api/v1/memories/disable")
def memories_disable() -> dict:
    MEMORY.set_memory_enabled(False)
    return {"memory_enabled": False}


@app.post("/api/v1/memories/enable")
def memories_enable() -> dict:
    MEMORY.set_memory_enabled(True)
    return {"memory_enabled": True}


@app.post("/api/v1/answer")
def answer(req: AnswerRequest) -> dict:
    if req.mode == "agent":
        if AGENT is None:
            raise HTTPException(503, "model backend unavailable; search is still usable")
        result = AGENT.answer(req.question, limit=req.limit)
        return result.to_dict()
    if PIPELINE is None:
        raise HTTPException(503, "model backend unavailable; search is still usable")
    result = PIPELINE.answer(req.question, ORCHESTRATOR,
                              task_class=req.task_class, limit=req.limit)
    return result.to_dict()


@app.get("/api/v1/history")
def history(limit: int = 10) -> dict:
    return {"history": MEMORY.history(limit), "notes": MEMORY.notes(limit)}


@app.get("/api/v1/unverifiable-notice")
def unverifiable_notice() -> dict:
    """Exposed so clients render the exact §12 refusal text, not their own."""
    return {"notice": UNVERIFIABLE_NOTICE}


if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="web")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
