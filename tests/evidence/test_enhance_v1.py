"""enhance_v1 §38 test suites — evidence lifecycle, immutable pack,
authority matrix, claim graph, module router, memory provenance, model
roles, retention."""

from __future__ import annotations

import time

import pytest

from agent.evidence.authority import (
    AuthorityResult,
    check_authority,
    resolve_authority,
)
from agent.evidence.claim_graph import (
    ClaimStatus,
    Severity,
    build_claim_graph,
    invalidation_closure,
    propagate_invalidation,
)
from agent.evidence.lifecycle import (
    EvidenceItem,
    EvidenceState,
    InvalidEvidenceTransition,
    admit_all,
    lifecycle_from_retrieval,
    mark_final,
    mark_used_validated,
    quarantine_filter,
)
from agent.evidence.pack import (
    EvidenceFishingError,
    PackFrozenError,
    assert_new_retrieval,
    freeze_pack,
)
from agent.validators.pipeline import EvidencePack
from retrieval.hybrid import RetrievedPassage


def _hadith(cid, en):
    return RetrievedPassage(
        citation_id=cid, surah=0, ayah=0, arabic="", translation=en,
        source_id=cid.split(":")[1], tier=1, leg="hadith", score=-1.0,
        collection=cid.split(":")[1], hadithnumber=int(cid.split(":")[2]),
    )


def _quran(cid, s, a, t):
    return RetrievedPassage(
        citation_id=cid, surah=s, ayah=a, arabic="", translation=t,
        source_id="quran-uthmani-json", tier=0, leg="reference", score=1.0,
    )


# ---------------------------------------------------------- §4 lifecycle
class TestEvidenceLifecycle:
    def test_happy_path(self):
        item = EvidenceItem(citation_id="quran:1:1")
        for state in (EvidenceState.RETRIEVED, EvidenceState.FILTERED,
                      EvidenceState.QUARANTINED, EvidenceState.ADMITTED,
                      EvidenceState.USED, EvidenceState.VALIDATED,
                      EvidenceState.FINAL):
            item.transition(state)
        assert item.state is EvidenceState.FINAL
        assert item.is_terminal

    def test_invalid_transition_raises(self):
        item = EvidenceItem(citation_id="x:1", state=EvidenceState.DISCOVERED)
        with pytest.raises(InvalidEvidenceTransition):
            item.transition(EvidenceState.ADMITTED)  # skips RETRIEVED/FILTERED

    def test_rejected_is_terminal(self):
        item = EvidenceItem(citation_id="x:1", state=EvidenceState.FILTERED)
        item.transition(EvidenceState.REJECTED)
        for target in EvidenceState:
            with pytest.raises(InvalidEvidenceTransition):
                item.transition(target)

    def test_generation_eligible_states(self):
        q = EvidenceItem(citation_id="q", state=EvidenceState.QUARANTINED)
        r = EvidenceItem(citation_id="r", state=EvidenceState.REJECTED)
        a = EvidenceItem(citation_id="a", state=EvidenceState.ADMITTED)
        assert not q.generation_eligible and not r.generation_eligible
        assert a.generation_eligible

    def test_pipeline_sequence(self):
        """The canonical harness sequence over real RetrievedPassages."""
        passages = [_quran("quran:13:28", 13, 28, "hearts are assured"),
                    _hadith("hadith:sahih-bukhari:999", "irrelevant text")]
        items = lifecycle_from_retrieval(passages)
        assert all(i.state is EvidenceState.FILTERED for i in items)
        kept = quarantine_filter(items, keep_citation_ids={"quran:13:28"})
        assert len(kept) == 1 and kept[0].state is EvidenceState.QUARANTINED
        admit_all(kept)
        assert kept[0].state is EvidenceState.ADMITTED
        mark_used_validated(kept, used_ids={"quran:13:28"},
                            verified_ids={"quran:13:28"})
        mark_final(kept)
        assert kept[0].state is EvidenceState.FINAL


# ------------------------------------------------------ §5 frozen pack
class TestFrozenEvidencePack:
    def test_freeze_and_checksum(self):
        pack = EvidencePack(query="q", passages=[_quran("quran:13:28", 13, 28, "text")])
        frozen = freeze_pack(pack, query_id="q-1")
        assert frozen.frozen
        assert frozen.citation_ids == {"quran:13:28"}

    def test_content_mutation_detected(self):
        pack = EvidencePack(query="q", passages=[_quran("quran:13:28", 13, 28, "original")])
        frozen = freeze_pack(pack, query_id="q-1")
        # tamper with the underlying passage text AFTER freeze
        pack.passages[0].translation = "tampered"
        with pytest.raises(PackFrozenError):
            _ = frozen.citation_ids  # checksum verification on access

    def test_fishing_prevention(self):
        with pytest.raises(EvidenceFishingError):
            assert_new_retrieval("pack-1", "pack-1", "q-1", "q-2")
        with pytest.raises(EvidenceFishingError):
            assert_new_retrieval("pack-1", "pack-2", "q-1", "q-1")
        assert_new_retrieval("pack-1", "pack-2", "q-1", "q-2")  # clean swap

    def test_prompt_block_uses_core_renderer(self):
        pack = EvidencePack(query="q", passages=[
            _hadith("hadith:sahih-bukhari:6369", "O Allah! I seek refuge...")
        ])
        frozen = freeze_pack(pack, query_id="q-1")
        block = frozen.to_prompt_block()
        assert "hadith:sahih-bukhari:6369" in block


# ---------------------------------------------------- §9-11 authority
class TestAuthorityMatrix:
    def test_source_class_resolution(self):
        assert resolve_authority("quran:2:255").authority_level.value == "quran"
        assert resolve_authority("hadith:sahih-bukhari:1").authority_level.value == "sahih_hadith"
        assert resolve_authority("tafsir-en:some_chunk").authority_level.value == "tafsir"
        assert resolve_authority("lecture:some-id").authority_level.value == "external_untrusted"

    def test_approved_collections_only(self):
        ok = resolve_authority("hadith:sahih-muslim:112")
        bad = resolve_authority("hadith:sunan-al-farabi:42")
        assert ok.authority_level.value == "sahih_hadith"
        assert bad.authority_level.value == "external_untrusted"

    def test_authority_vs_entailment_separation(self):
        # §24 'tafsir treated as Quran': right content, wrong class
        assert check_authority("direct_fact", "tafsir-en:x") is AuthorityResult.AUTHORITY_FAIL
        assert check_authority("direct_fact", "quran:13:28") is AuthorityResult.SUPPORTED
        # §24 'lecture treated as revelation'
        assert check_authority("attribution", "lecture:x") is AuthorityResult.AUTHORITY_FAIL
        assert check_authority("attribution", "hadith:sahih-bukhari:1") is AuthorityResult.SUPPORTED

    def test_never_allowed_claim_types(self):
        for ct in ("guarantee", "diagnosis", "causal_claim", "prediction"):
            assert check_authority(ct, "quran:13:28") is AuthorityResult.AUTHORITY_FAIL
            assert check_authority(ct, "hadith:sahih-bukhari:1") is AuthorityResult.AUTHORITY_FAIL

    def test_case_insensitive_claim_types(self):
        assert check_authority("DIRECT_FACT", "quran:1:1") is AuthorityResult.SUPPORTED
        assert check_authority("direct_fact", "quran:1:1") is AuthorityResult.SUPPORTED

    def test_no_evidence(self):
        assert check_authority("attribution", "") is AuthorityResult.NO_EVIDENCE


# -------------------------------------------------------- §6-8 graph
class TestClaimGraph:
    def test_stable_ids_and_severity(self):
        g = build_claim_graph(
            "The Prophet ﷺ said X. This is haram. You have depression."
        )
        ids = [c.id for c in g]
        assert ids == ["c1", "c2", "c3"]
        assert g[0].severity is Severity.CRITICAL  # attribution
        assert g[1].severity is Severity.CRITICAL  # ruling
        assert g[2].severity is Severity.CRITICAL  # diagnosis

    def test_dependency_invalidation_transitive(self):
        text = ("The Quran describes patience in hardship. "
                "Therefore, patience cures every difficulty. "
                "Thus, all hardships are easily overcome.")
        g = build_claim_graph(text)
        assert g[1].dependencies == ["c1"]
        assert g[2].dependencies == ["c2"]
        g[1].status = ClaimStatus.UNSUPPORTED
        propagate_invalidation(g)
        assert g[2].status is ClaimStatus.INVALIDATED
        assert invalidation_closure(g) >= {"c2", "c3"}

    def test_supported_parent_does_not_invalidate(self):
        g = build_claim_graph("The Quran mentions mercy repeatedly. "
                              "Therefore, mercy is central to Islam.")
        g[0].status = ClaimStatus.SUPPORTED
        g[1].status = ClaimStatus.PENDING
        propagate_invalidation(g)
        assert g[1].status is ClaimStatus.PENDING


# ------------------------------------------------------- §12-16 modules
class TestModuleArchitecture:
    def test_minimal_module_selection(self):
        from agent.modules.router import ModuleRouter, build_default_registry

        router = ModuleRouter(build_default_registry())
        emotional = router.select("I feel lonely.")
        assert emotional["knowledge"] == []  # §17: no knowledge modules
        dua = router.select("Is there any dua for depression?")
        assert "hadith" in dua["knowledge"] and "dua" in dua["capabilities"]
        quran_q = router.select("What does 2:255 mean?")
        assert "quran" in quran_q["knowledge"]

    def test_disabled_modules_not_routed(self):
        from agent.modules.router import ModuleRouter, build_default_registry

        registry = build_default_registry()
        registry.knowledge["fiqh"].enabled = False
        router = ModuleRouter(registry)
        sel = router.select("Is riba halal?")
        assert "fiqh" not in sel["knowledge"]

    def test_knowledge_module_provides_evidence_not_answers(self):
        from agent.modules.router import build_default_registry

        registry = build_default_registry()
        quran = registry.knowledge["quran"]
        results = quran.search("mercy", {"limit": 3})
        assert isinstance(results, list) and results  # evidence rows
        # validate_source delegates to the authority layer
        meta = quran.validate_source("quran:1:1")
        assert meta["authority_level"] == "quran"


# -------------------------------------------------- §18-20 memory provenance
class TestMemoryProvenance:
    def test_source_classes_and_confidence(self, tmp_path):
        from agent.companion.memory import CompanionMemory

        mem = CompanionMemory(db_path=tmp_path / "m.db")
        mem.save_fact("User's name is Adam", source_class="explicit_user")
        mem.save_fact("User studies Al-Kahf", source_class="inferred", confidence=0.9)
        mem.save_fact("Session fact", source_class="temporary")
        facts = {f["fact"]: f for f in mem.facts()}
        explicit = facts["User's name is Adam"]
        inferred = facts["User studies Al-Kahf"]
        temporary = facts["Session fact"]
        assert explicit["source_class"] == "explicit_user"
        assert explicit["confidence"] == 1.0
        # §19: INFERRED ≠ FACT — capped confidence + auto-expiry
        assert inferred["confidence"] <= 0.7
        assert inferred["expires_at"] is not None
        assert temporary["expires_at"] is not None

    def test_inferred_never_silent_permanent(self, tmp_path):
        from agent.companion.memory import CompanionMemory

        mem = CompanionMemory(db_path=tmp_path / "m.db")
        mem.save_fact("User likes X", source_class="inferred")
        f = mem.facts()[0]
        assert f["source_class"] == "inferred"
        assert f["confidence"] < 1.0
        assert f["expires_at"] is not None  # will expire, never permanent

    def test_expiry_actually_deletes(self, tmp_path):
        from agent.companion.memory import CompanionMemory

        mem = CompanionMemory(db_path=tmp_path / "m.db")
        mem.save_fact("short-lived", source_class="temporary",
                      expires_at=time.time() - 1)  # already expired
        assert mem.facts() == []  # sweep removed it

    def test_unknown_source_class_rejected(self, tmp_path):
        from agent.companion.memory import CompanionMemory, FactRejected

        mem = CompanionMemory(db_path=tmp_path / "m.db")
        with pytest.raises(FactRejected):
            mem.save_fact("x", source_class="hallucinated_class")


# ---------------------------------------------------- §25-26 model roles
class TestModelRoles:
    def test_roles_configurable(self):
        from agent.core.model_roles import ModelRole, ModelRoleTable

        table = ModelRoleTable.load()
        assert table.model_for(ModelRole.GENERATOR) == "ling_tiny"
        assert table.model_for(ModelRole.EMBEDDER)  # embedder configured
        # §26: swap = config change
        table.replace(ModelRole.VALIDATOR, "gemma_qat")
        assert table.model_for(ModelRole.VALIDATOR) == "gemma_qat"

    def test_classifier_is_deterministic(self):
        from agent.core.model_roles import ModelRole, ModelRoleTable

        table = ModelRoleTable.load()
        assert table.model_for(ModelRole.CLASSIFIER) == "deterministic"


# ------------------------------------------------------- §30-31 privacy
class TestRetentionAndPrivacy:
    def test_retention_config(self):
        from agent.core.model_roles import RetentionPolicy

        policy = RetentionPolicy.load()
        assert policy.log_retention_days > 0
        assert policy.session_retention_days > 0
        assert policy.memory_retention_days > 0
        assert policy.enable_content_logging is True

    def test_restricted_conversation_never_logs_content(self):
        from agent.core.model_roles import ConversationClass

        assert not ConversationClass.logging_allows(
            ConversationClass.RESTRICTED, enable_content_logging=True
        )
        assert ConversationClass.logging_allows(
            ConversationClass.NORMAL, enable_content_logging=True
        )
        assert not ConversationClass.logging_allows(
            ConversationClass.NORMAL, enable_content_logging=False
        )

    def test_no_training_path_from_logs(self):
        """§29: there is no code path from chat logs to any training
        pipeline — verify the log module imports no training/finetune
        references."""
        import inspect

        import agent.companion.logging as logmod

        source = inspect.getsource(logmod)
        for banned in ("train", "finetune", "sft", "qlora", "dataset"):
            assert banned not in source.lower(), (
                f"chat-log module references {banned!r} — training-data isolation violated"
            )


# ------------------------------------------- §40 failure injection (subset)
class TestFailureInjection:
    def test_authority_registry_rejects_source(self):
        # a fake hadith collection cannot establish attribution
        assert check_authority("attribution", "hadith:invented-book:1") \
            is AuthorityResult.AUTHORITY_FAIL

    def test_claim_depends_on_failed_claim(self):
        g = build_claim_graph("The hadith mentions night prayer. "
                              "Therefore, night prayer cures sadness.")
        g[0].status = ClaimStatus.UNSUPPORTED
        propagate_invalidation(g)
        assert g[1].status is ClaimStatus.INVALIDATED
        assert invalidation_closure(g) == {"c1", "c2"}

    def test_rejected_evidence_cannot_return(self):
        item = EvidenceItem(citation_id="bad", state=EvidenceState.FILTERED)
        quarantine_filter([item], keep_citation_ids=set())
        assert item.state is EvidenceState.REJECTED
        with pytest.raises(InvalidEvidenceTransition):
            item.transition(EvidenceState.QUARANTINED)
