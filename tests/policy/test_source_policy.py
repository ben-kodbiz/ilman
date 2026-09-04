"""Source-policy tests (agentodo.md §26 Phase 0 exit condition).

These are the gates the whole project depends on: nothing outside the
approved Sunni registry may ever be ingested or retrieved.
"""

from __future__ import annotations

import pytest

from agent.policy.source_policy import (
    Decision,
    SourcePolicy,
    SourceRecord,
    SourceRegistry,
)


@pytest.fixture()
def policy() -> SourcePolicy:
    return SourcePolicy(SourceRegistry.load())


@pytest.fixture()
def record():
    def _make(**over):
        base = dict(
            id="quran-uthmani", title="The Qur'an", author="Revelation", type="quran",
            language="ar", tradition="SUNNI", allowed=True,
            verification_status="verified", license="public terms",
        )
        base.update(over)
        return SourceRecord.from_dict(base)
    return _make


class TestIngestionGate:
    def test_verified_sunni_source_allowed(self, policy, record):
        result = policy.ingestion_gate(record())
        assert result.decision is Decision.ALLOW
        assert result.allowed

    def test_not_on_registry_rejected(self, policy, record):
        result = policy.ingestion_gate(record(id="random-internet-book"))
        assert result.decision is Decision.REJECT
        assert any("approved-source registry" in r for r in result.reasons)

    def test_excluded_source_rejected_even_if_looks_good(self, policy, record):
        # The exclusion list must win over every other field (§5.3).
        result = policy.ingestion_gate(record(id="web-uncategorized-fatwa-compilation"))
        assert result.decision is Decision.REJECT
        assert any("excluded" in r for r in result.reasons)

    def test_non_sunni_tradition_rejected(self, policy, record):
        result = policy.ingestion_gate(record(tradition="OTHER"))
        assert result.decision is Decision.REJECT
        assert any("not SUNNI" in r for r in result.reasons)

    def test_rejected_verification_rejected(self, policy, record):
        result = policy.ingestion_gate(record(verification_status="rejected"))
        assert result.decision is Decision.REJECT

    def test_pending_verification_goes_to_manual_review(self, policy):
        registry = SourceRegistry.load()
        # tafsir-ibn-kathir is seeded pending -> must block automated ingestion.
        result = SourcePolicy(registry).ingestion_gate(registry.get("tafsir-ibn-kathir"))
        assert result.decision is Decision.MANUAL_REVIEW

    def test_unconfirmed_license_manual_review(self, policy, record):
        result = policy.ingestion_gate(record(license="review_required"))
        assert result.decision is Decision.MANUAL_REVIEW

    def test_missing_metadata_manual_review(self, policy, record):
        result = policy.ingestion_gate(record(author=""))
        assert result.decision is Decision.MANUAL_REVIEW

    def test_registry_rejects_ambiguous_entry(self, policy, record):
        # A record that claims allowed=true but the registry says allowed=false.
        from agent.policy.source_policy import SourceRegistry as R
        merged = R({"x": record(id="x", allowed=False)}, {})
        result = SourcePolicy(merged).ingestion_gate(record(id="x"))
        assert result.decision is Decision.REJECT

    def test_assert_ingestible_blocks_manual_review(self, policy):
        registry = SourceRegistry.load()
        with pytest.raises(Exception):
            SourcePolicy(registry).assert_ingestible(registry.get("tafsir-ibn-kathir"))

    def test_gate_order_ignores_record_claim_of_allowed(self, policy, record):
        """The gate must consult the registry, not trust the incoming record."""
        result = policy.ingestion_gate(record(id="tafsir-ibn-kathir"))
        # tafsir-ibn-kathir is on the registry as pending -> MANUAL_REVIEW, not ALLOW,
        # even though the caller's record says verified/allowed.
        assert result.decision is Decision.MANUAL_REVIEW


class TestRetrievalFilter:
    def test_mandatory_filters_pass(self, policy, record):
        assert policy.retrieval_filter(record())

    def test_disallowed_blocked(self, policy, record):
        assert not policy.retrieval_filter(record(allowed=False))

    def test_non_sunni_blocked(self, policy, record):
        assert not policy.retrieval_filter(record(tradition="SHIA"))

    def test_rejected_blocked(self, policy, record):
        assert not policy.retrieval_filter(record(verification_status="rejected"))

    def test_must_not_retrieve(self, policy):
        assert policy.must_not_retrieve("web-uncategorized-fatwa-compilation")
        assert not policy.must_not_retrieve("quran-uthmani")


class TestTiers:
    def test_quran_is_tier0(self, policy, record):
        assert policy.tier(record()) == 0

    def test_hadith_is_tier1(self, policy, record):
        assert policy.tier(record(type="sunnah")) == 1

    def test_tafsir_is_tier2(self, policy, record):
        assert policy.tier(record(type="tafsir")) == 2

    def test_educational_is_tier5(self, policy, record):
        assert policy.tier(record(type="educational")) == 5


class TestRegistryIntegrity:
    def test_no_overlap_between_registries(self):
        SourceRegistry.load()  # raises ValueError in load() on overlap

    def test_all_approved_have_required_fields(self):
        for rec in SourceRegistry.load().all_approved():
            assert rec.id and rec.title and rec.type and rec.tradition
