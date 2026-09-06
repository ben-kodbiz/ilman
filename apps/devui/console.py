"""Gradio dev console for live chat testing (all components wired).

Launch:
    uv run python -m apps.devui.console

Tabs:
    Companion (v2)  — the full harness: state/policy/context/memory/safety
                      + citation validation; shows mode, policy, citations,
                      memory hits and the developer trace (§32)
    Grounded QA     — the AgentOrchestrator: tools + evidence packs + the
                      deterministic citation validator (religious answers)
    Search          — raw hybrid retrieval (all legs: reference/FTS/translation/
                      hadith/tafsir/vector) for corpus-level debugging

Everything runs live against LM Studio + the ingested corpora. The v2 model
role is a dropdown (runtime config §3); changing it reloads the harness.
"""

# ruff: noqa: E402  (sys.path bootstrap must precede repo imports)
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import gradio as gr  # noqa: E402

from agent.companion.memory import CompanionMemory  # noqa: E402
from agent.context.builder import ContextBuilder  # noqa: E402
from agent.core.agent import AgentOrchestrator  # noqa: E402
from agent.core.config import load_config  # noqa: E402
from agent.core.harness import CompanionHarness  # noqa: E402
from agent.core.model import ModelRouter  # noqa: E402
from agent.memory.router import MemoryRouter  # noqa: E402
from agent.policy.companion_policy import CompanionPolicyEngine  # noqa: E402
from agent.policy.source_policy import SourcePolicy, SourceRegistry  # noqa: E402
from agent.state.manager import StateManager  # noqa: E402
from agent.tools.layer import ToolLayer  # noqa: E402
from agent.validators.companion_validator import ResponseValidator  # noqa: E402
from agent.validators.pipeline import CitationValidator  # noqa: E402
from ingestion.hadith_ingest import HadithStore  # noqa: E402
from ingestion.quran_ingest import QuranStore  # noqa: E402
from ingestion.tafsir_en_ingest import TafsirEnStore  # noqa: E402
from ingestion.tafsir_ingest import TafsirStore  # noqa: E402
from ingestion.web_fatwa_ingest import WebFatwaStore  # noqa: E402
from retrieval.hybrid import RetrievalOrchestrator  # noqa: E402

try:
    from retrieval.vector_store import VectorStore
except Exception:
    VectorStore = None

CFG = load_config()
BACKEND = CFG.defaults.get("backend", "lmstudio")
MODEL_ROLES = list(CFG.backends[BACKEND].models.keys())

STORE = QuranStore()
HADITH = HadithStore()
TAFSIR = TafsirStore()
TAFSIR_EN = TafsirEnStore()
WEB_FATWAS = WebFatwaStore()
POLICY = SourcePolicy(SourceRegistry.load())
MEMORY = MemoryRouter(CompanionMemory(db_path=REPO_ROOT / "knowledge" / "processed" / "devui_memory.db"))
STATES = StateManager()

VECTOR = None
if VectorStore is not None:
    try:
        candidate = VectorStore()
        if candidate.size:
            VECTOR = candidate
    except Exception:
        VECTOR = None

RETRIEVAL = RetrievalOrchestrator(
    STORE, hadith_store=HADITH, tafsir_store=TAFSIR,
    tafsir_en_store=TAFSIR_EN, web_fatwa_store=WEB_FATWAS, vector_store=VECTOR,
)
TOOLS = ToolLayer(POLICY, store=STORE, hadith_store=HADITH, memory=MEMORY.memory)
AGENT: AgentOrchestrator | None = None

_HARNESSES: dict[str, CompanionHarness] = {}


# last exchange bookkeeping for the rating buttons (session -> turn + answer)
_LAST_EXCHANGE: dict[str, dict] = {}


def _harness(role: str) -> CompanionHarness:
    """One harness instance per model role (runtime switchable, §24)."""
    if role not in _HARNESSES:
        cfg = load_config()
        for key in list(cfg.routing):
            cfg.routing[key] = role
        router = ModelRouter(cfg)
        _HARNESSES[role] = CompanionHarness(
            router, retrieval=RETRIEVAL, memory_router=MEMORY,
            states=StateManager(), policy_engine=CompanionPolicyEngine(),
            context_builder=ContextBuilder(), validator=ResponseValidator(),
            citation_validator=CitationValidator(),
            model_label=f"{role}:{cfg.backends[BACKEND].models.get(role, role)}",
        )
    return _HARNESSES[role]


def _agent(role: str) -> AgentOrchestrator:
    global AGENT
    if AGENT is None:
        cfg = load_config()
        for key in list(cfg.routing):
            cfg.routing[key] = role
        AGENT = AgentOrchestrator(ModelRouter(cfg), RETRIEVAL, TOOLS, memory=MEMORY.memory)
    return AGENT


# ---------------------------------------------------------------- companion
def companion_chat(message, history, session, model_role, show_trace):
    if not message.strip():
        yield history, "", _status()
        return
    harness = _harness(model_role)
    history = history + [{"role": "user", "content": message}]
    try:
        result = harness.respond(session or "devui", message)
    except Exception as e:  # backend down: surface, never crash the console
        history = history + [{"role": "assistant", "content": f"[backend error] {e}"}]
        yield history, "", _status()
        return
    meta = []
    if result.citations:
        meta.append("**citations**: " + ", ".join(result.citations))
    if result.unsupported_citations:
        meta.append("**unsupported**: " + ", ".join(result.unsupported_citations))
    cv = result.companion_validation
    if cv:
        meta.append(f"**validation**: {'ok' if cv.get('ok') else cv}")
    answer = result.answer + ("\n\n" + "\n\n".join(meta) if meta else "")
    history = history + [{"role": "assistant", "content": answer}]
    _LAST_EXCHANGE[session or "devui"] = {
        "turn": result.state.get("turn_count", 0),
        "answer": result.answer,
    }
    trace_text = ""
    if show_trace:
        import json

        trace_text = json.dumps(result.trace, indent=2, ensure_ascii=False)
        trace_text += "\n\nstate:\n" + json.dumps(result.state, indent=2, ensure_ascii=False)
        trace_text += "\n\npolicy:\n" + json.dumps(result.policy, indent=2, ensure_ascii=False)
    yield history, trace_text, _status()


def rate_answer(session, rating):
    """Thumb up/down the most recent answer; stored with the chat-log turn
    link + pipeline snapshot for later down-rating analysis."""
    try:
        from agent.companion.ratings import enrich_from_chat_log, record_rating

        last = _LAST_EXCHANGE.get(session or "devui")
        if not last:
            return "no answer to rate yet."
        pipeline = enrich_from_chat_log(session or "devui", last["turn"])
        record_rating(session or "devui", last["turn"], rating,
                      answer_excerpt=last["answer"][:200], pipeline=pipeline)
        return ("👍 thanks — noted as a good answer." if rating == "up"
                else "👎 noted — flagged for pipeline review.")
    except Exception as e:
        return f"rating failed: {e}"


def companion_clear(session):
    STATES.drop(session or "devui")
    return [], "", _status()


def memory_view():
    import json

    return json.dumps(MEMORY.memory.memory_view(), indent=2, ensure_ascii=False)


def memory_clear():
    MEMORY.clear_all()
    return "memories cleared."


# ----------------------------------------------------------------------- qa
def qa_chat(message, history, model_role):
    if not message.strip():
        yield history, _status()
        return
    history = history + [{"role": "user", "content": message}]
    try:
        agent = _agent(model_role)
        result = agent.answer(message)
    except Exception as e:
        history = history + [{"role": "assistant", "content": f"[backend error] {e}"}]
        yield history, _status()
        return
    meta = []
    if result.citations:
        meta.append("**citations**: " + ", ".join(result.citations))
    if result.unsupported_citations:
        meta.append("**unsupported (stripped)**: " + ", ".join(result.unsupported_citations))
    if result.trace.tool_calls:
        calls = ", ".join(t["name"] for t in result.trace.tool_calls)
        meta.append(f"**tools used**: {calls}")
    answer = result.answer + ("\n\n" + "\n\n".join(meta) if meta else "")
    history = history + [{"role": "assistant", "content": answer}]
    yield history, _status()


# ------------------------------------------------------------------- search
def raw_search(query, limit):
    if not query.strip():
        yield ""
        return
    out = []
    for p in RETRIEVAL.search(query, limit=int(limit)):
        out.append(
            f"**{p.citation_id}** (tier {p.tier}, leg {p.leg}, scholar: {p.scholar or '—'})\n"
            + (p.translation[:300] + (" [...]" if len(p.translation) > 300 else "")
               if p.translation else p.arabic[:300])
        )
    yield ("\n\n---\n\n".join(out) or "no results")


def _status() -> str:
    try:
        import requests

        r = requests.get("http://127.0.0.1:1234/api/v0/models", timeout=3)
        loaded = [m["id"] for m in r.json()["data"] if m.get("state") == "loaded"]
        llms = [m for m in loaded if m != "text-embedding-nomic-embed-text-v1.5"]
        return (
            f"corpus: {STORE.ayah_count_total()} ayahs · {HADITH.hadith_count()} hadiths · "
            f"{TAFSIR.tafsir_count()} tafsir · {TAFSIR_EN.chunk_count()} classic chunks · "
            f"vector: {VECTOR.size if VECTOR else 0} | LM Studio LLM: {llms or 'none loaded'}"
        )
    except Exception:
        return "LM Studio unreachable — search still works, chat will error."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Ilman Dev Console") as app:
        gr.Markdown("## Ilman — dev console\nLive testing: v2 companion harness, grounded QA, raw retrieval.")
        status = gr.Markdown(_status())

        with gr.Tabs():
            # ---------------------------------------------------- companion v2
            with gr.Tab("Companion (v2)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chat = gr.Chatbot(height=430)
                        msg = gr.Textbox(placeholder="Talk to Ilman… (Enter to send)", show_label=False)
                        with gr.Row():
                            send = gr.Button("Send", variant="primary")
                            clear = gr.Button("Clear conversation")
                    with gr.Column(scale=2):
                        session = gr.Textbox(label="session id", value="devui")
                        model_role = gr.Dropdown(
                            MODEL_ROLES, value="ling_tiny",
                            label="model role (runtime routing)",
                        )
                        show_trace = gr.Checkbox(label="show developer trace (§32)", value=True)
                        with gr.Row():
                            thumb_up = gr.Button("👍 good answer")
                            thumb_down = gr.Button("👎 bad answer")
                        rate_feedback = gr.Markdown()
                        trace = gr.TextArea(label="trace / state / policy", lines=22, max_lines=30)
                with gr.Accordion("Memory controls (§25)", open=False):
                    with gr.Row():
                        mem_view_btn = gr.Button("View memories")
                        mem_clear_btn = gr.Button("Clear memories")
                    mem_out = gr.TextArea(label="memory", lines=10, max_lines=20)

                send.click(companion_chat, [msg, chat, session, model_role, show_trace],
                           [chat, trace, status])
                msg.submit(companion_chat, [msg, chat, session, model_role, show_trace],
                           [chat, trace, status])
                thumb_up.click(
                    lambda s: rate_answer(s, "up"), [session], rate_feedback
                )
                thumb_down.click(
                    lambda s: rate_answer(s, "down"), [session], rate_feedback
                )
                clear.click(companion_clear, [session], [chat, trace, status])
                mem_view_btn.click(memory_view, None, mem_out)
                mem_clear_btn.click(memory_clear, None, mem_out)
                msg.submit(lambda: "", None, msg)
                send.click(lambda: "", None, msg)

            # --------------------------------------------------------- grounded QA
            with gr.Tab("Grounded QA"):
                qa = gr.Chatbot(height=470)
                qmsg = gr.Textbox(placeholder="Ask an Islamic question…", show_label=False)
                with gr.Row():
                    qsend = gr.Button("Send", variant="primary")
                qa_model = gr.Dropdown(MODEL_ROLES, value="ling_tiny", label="model role")
                qsend.click(qa_chat, [qmsg, qa, qa_model], [qa, status])
                qmsg.submit(qa_chat, [qmsg, qa, qa_model], [qa, status])
                qmsg.submit(lambda: "", None, qmsg)
                qsend.click(lambda: "", None, qmsg)

            # ---------------------------------------------------------- raw search
            with gr.Tab("Search (all legs)"):
                sq = gr.Textbox(placeholder="Search the corpus (Arabic or English)…", show_label=False)
                slimit = gr.Slider(3, 20, value=8, step=1, label="results")
                sbutton = gr.Button("Search", variant="primary")
                sout = gr.Markdown()
                sbutton.click(raw_search, [sq, slimit], sout)
                sq.submit(raw_search, [sq, slimit], sout)
    return app


def main() -> None:
    import os

    app = build_app()
    host = os.environ.get("ILMAN_HOST", "127.0.0.1")
    app.launch(server_name=host, server_port=7860, show_error=True,
               inbrowser=(host == "127.0.0.1"), css=".footer {display:none}")


if __name__ == "__main__":
    main()
