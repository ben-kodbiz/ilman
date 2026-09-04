from __future__ import annotations

import pytest

from agent.core.config import load_config
from agent.policy.source_policy import SourcePolicy, SourceRegistry
from agent.tools.layer import ToolLayer, execute_tool
from ingestion.quran_ingest import QuranStore


@pytest.fixture()
def layer() -> ToolLayer:
    store = QuranStore() if QuranStore().ayah_count_total() else None
    return ToolLayer(SourcePolicy(SourceRegistry.load()), store=store)


class TestGetAyah:
    def test_known_ayah(self, layer):
        result = layer.get_ayah(2, 255)
        assert result.ok
        assert result.data["citation_id"] == "quran:2:255"
        assert result.data["arabic"].strip()

    def test_out_of_range_ayah(self, layer):
        result = layer.get_ayah(3, 300)
        assert not result.ok  # never invents text (§14)

    def test_rejects_non_integers(self, layer):
        assert not layer.get_ayah("2", 255).ok
        assert not layer.get_ayah(2, "255").ok


class TestVerifyQuranReference:
    def test_alias_resolves(self, layer):
        result = layer.verify_quran_reference("Ayat al-Kursi")
        assert result.ok
        assert result.data["surah"] == 2 and result.data["ayah"] == 255

    def test_garbage_rejected(self, layer):
        assert not layer.verify_quran_reference("definitely not a reference").ok


class TestSourceMetadataTool:
    def test_approved(self, layer):
        result = layer.get_source_metadata("quran-uthmani")
        assert result.ok
        assert result.data["tradition"] == "SUNNI"

    def test_excluded_never_retrievable(self, layer):
        result = layer.get_source_metadata("web-uncategorized-fatwa-compilation")
        assert not result.ok
        assert "excluded" in result.error

    def test_unknown_id(self, layer):
        assert not layer.get_source_metadata("no-such-source").ok

    def test_pending_source_blocked_by_filter(self, layer):
        # pending verification -> passes §8 retrieval filter (only 'rejected'
        # blocks retrieval), but must NEVER be ingestable without review
        result = layer.get_source_metadata("tafsir-ibn-kathir")
        assert result.ok  # retrieval shows metadata with pending status
        # ingestion, however, requires human review:
        from agent.policy.source_policy import Decision, SourcePolicy, SourceRegistry
        pol = SourcePolicy(SourceRegistry.load())
        assert pol.ingestion_gate(pol.registry.get("tafsir-ibn-kathir")).decision is Decision.MANUAL_REVIEW


class TestExecuteTool:
    def test_dispatch(self, layer):
        result = execute_tool(layer, "get_ayah", {"surah": 1, "ayah": 1})
        assert result.ok

    def test_unknown_tool(self, layer):
        assert not execute_tool(layer, "delete_everything", {}).ok

    def test_bad_args(self, layer):
        assert not execute_tool(layer, "get_ayah", {"surah": "one"}).ok


class TestRoutingConfig:
    def test_runtime_routing_resolves(self):
        cfg = load_config()
        backend, model = cfg.backend_for("complex_rag")
        assert backend.name == "lmstudio"
        assert model == "ling-3.0-tiny"

    def test_routing_not_hardcoded(self):
        cfg = load_config()
        backend, model = cfg.backend_for("simple_chat")
        assert model != cfg.backends["lmstudio"].models["ling_tiny"]  # gemma for simple chat
        assert model == cfg.backends["lmstudio"].models["gemma_qat"]
