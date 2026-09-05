from __future__ import annotations

import pytest

from agent.companion.memory import CompanionMemory
from agent.memory.router import MemoryRouter
from agent.policy.companion_policy import ResponsePolicy
from agent.safety.router import safety_route
from agent.state.models import Risk
from agent.validators.companion_validator import (
    ResponseValidator,
)


@pytest.fixture()
def router(tmp_path):
    return MemoryRouter(CompanionMemory(db_path=tmp_path / "m.db"))


class TestMemoryRouter:
    """§10-§12 categories + lifecycle + relevance."""

    def test_categories_saved(self, router):
        out = router.route_incoming(
            "I prefer Malay explanations. My name is Adam. "
            "I'm studying Surah Al-Kahf."
        )
        cats = {s["category"] for s in out["saved"]}
        assert {"preference", "profile", "study"} <= cats

    def test_transient_rejected(self, router):
        out = router.route_incoming("I feel sad today")
        assert out["saved"] == []
        assert any("transient" in r["reason"] for r in out["rejected"])

    def test_privacy_rejected(self, router):
        out = router.route_incoming("remember that I am addicted to pills")
        assert out["saved"] == []

    def test_explicit_remember_saved(self, router):
        out = router.route_incoming("Remember that my exam is on Friday")
        assert any("exam" in s["fact"] for s in out["saved"])

    def test_dedup_on_repeat(self, router):
        router.route_incoming("My name is Adam")
        router.route_incoming("My name is Adam")
        facts = [f["fact"] for f in router.memory.facts()]
        assert sum("Adam" in f for f in facts) == 1

    def test_relevance_not_dump(self, router):
        router.route_incoming("I'm studying Surah Al-Kahf")
        router.route_incoming("My name is Adam")
        hits = router.relevant("Let's continue studying Al-Kahf")
        assert hits and any("Al-Kahf" in h["fact"] for h in hits)
        assert all("Adam" not in h["fact"] for h in hits)

    def test_forget(self, router):
        router.route_incoming("My name is Adam")
        fid = router.memory.facts()[0]["id"]
        assert router.forget(fid)
        assert router.memory.facts() == []


class TestSafetyRouter:
    def test_levels(self):
        assert safety_route("I want to die").risk is Risk.HIGH
        assert safety_route("I feel worthless and hopeless").risk is Risk.ELEVATED
        assert safety_route("Good morning").risk is Risk.LOW

    def test_high_blocks_model(self):
        assert safety_route("kill myself").model_allowed is False

    def test_low_allows_model(self):
        assert safety_route("I feel lonely").model_allowed is True


class TestResponseValidator:
    def test_factually_fine_but_policy_fail(self):
        """§25: religious opener on a lonely message — factually OK, policy FAIL."""
        v = ResponseValidator()
        policy = ResponsePolicy(mode=__import__(
            "agent.state.models", fromlist=["Mode"]).Mode.COMPANION,
            preach=False, max_followups=1, word_budget=90,
        )
        validation = v.validate(
            "Allah says in the Quran that He is with you. Why are you lonely? "
            "What happened?", policy,
        )
        assert not validation.ok
        assert any("preachy" in p for p in validation.policy_problems)
        assert any("questions" in p for p in validation.policy_problems)

    def test_dependency_language_flagged(self):
        v = ResponseValidator()
        policy = ResponsePolicy()
        validation = v.validate("I am all you need. You only need me.", policy)
        assert not validation.ok
        assert "dependency-forming language" in validation.companion_problems

    def test_human_pretense_flagged(self):
        v = ResponseValidator()
        validation = v.validate("When I was a child, I cried too.", ResponsePolicy())
        assert "pretends to be human" in validation.companion_problems

    def test_diagnosis_flagged(self):
        v = ResponseValidator()
        validation = v.validate("You have depression.", ResponsePolicy())
        assert "diagnosis-like language" in validation.companion_problems

    def test_clean_empathy_passes(self):
        v = ResponseValidator()
        policy = ResponsePolicy()
        validation = v.validate(
            "I'm sorry you're feeling that way. Want to tell me more?", policy
        )
        assert validation.ok

    def test_safety_guilt_flagged(self):
        v = ResponseValidator()
        policy = ResponsePolicy(safety_override=True)
        validation = v.validate("Do not despair, this would be sinful.", policy)
        assert not validation.ok
