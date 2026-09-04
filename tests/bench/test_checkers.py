from __future__ import annotations

from agent.core.model import ModelResponse, ToolCall
from evaluation.bench.checkers import run_checks


def _resp(content="", tool_calls=None):
    return ModelResponse(content=content, tool_calls=tool_calls or [])


class TestRefusalDetection:
    def test_honest_refusal_passes(self):
        resp = _resp("I cannot verify this hadith in the provided sources.")
        checks = run_checks(resp, [{"type": "refusal"}])
        assert checks[0]["passed"]

    def test_confident_answer_fails(self):
        resp = _resp("This is narrated in Sahih al-Bukhari 1234, graded sahih.")
        checks = run_checks(resp, [{"type": "refusal"}])
        assert not checks[0]["passed"]


class TestNotHallucinating:
    def test_no_citation_is_safe(self):
        resp = _resp("I don't have that verse in the provided material.")
        assert run_checks(resp, [{"type": "not_hallucinating"}])[0]["passed"]

    def test_numeric_fabrication_fails(self):
        resp = _resp("That verse is Surah 42:20, from Surah ash-Shura.")
        assert not run_checks(resp, [{"type": "not_hallucinating"}])[0]["passed"]

    def test_negated_claim_is_safe(self):
        resp = _resp("This does not appear in the Qur'an as an exact verse.")
        assert run_checks(resp, [{"type": "not_hallucinating"}])[0]["passed"]


class TestToolCallCheck:
    def _resp_with_call(self, name, args):
        return ModelResponse(
            content="", tool_calls=[ToolCall(name=name, arguments=args)]
        )

    def test_correct_call(self):
        resp = self._resp_with_call("get_ayah", {"surah": 2, "ayah": 255})
        spec = {"type": "tool_call", "expected": {"name": "get_ayah", "args": {"surah": 2, "ayah": 255}}}
        assert run_checks(resp, [spec])[0]["passed"]

    def test_wrong_args(self):
        resp = self._resp_with_call("get_ayah", {"surah": 2, "ayah": 1})
        spec = {"type": "tool_call", "expected": {"name": "get_ayah", "args": {"ayah": 255}}}
        assert not run_checks(resp, [spec])[0]["passed"]


class TestJsonChecks:
    def test_valid(self):
        resp = _resp('{"surah": 2}')
        assert run_checks(resp, [{"type": "json_valid"}])[0]["passed"]

    def test_fenced_still_valid(self):
        resp = _resp('```json\n{"surah": 2}\n```')
        assert run_checks(resp, [{"type": "json_valid"}])[0]["passed"]

    def test_subset(self):
        resp = _resp('{"surah": 2, "ayah": 255, "extra": 1}')
        spec = {"type": "json_subset", "expected": {"surah": 2}}
        assert run_checks(resp, [spec])[0]["passed"]


class TestAllChecksMustPass:
    def test_single_failure_fails_case(self):
        resp = _resp("Wednesday but not exact")
        checks = run_checks(resp, [
            {"type": "contains", "expected": "Wednesday"},
            {"type": "exact", "expected": "Wednesday"},
        ])
        assert not all(c["passed"] for c in checks)


class TestRefsList:
    def test_plain_refs(self):
        resp = _resp('["2:255", "112:1"]')
        spec = {"type": "refs_list", "expected": ["2:255", "112:1"]}
        assert run_checks(resp, [spec])[0]["passed"]

    def test_prefixed_ref_still_parses(self):
        resp = _resp('["surah:ayah:2:255", "surah:ayah:112:1"]')
        spec = {"type": "refs_list", "expected": ["2:255", "112:1"]}
        assert run_checks(resp, [spec])[0]["passed"]

    def test_wrong_order_fails(self):
        resp = _resp('["112:1", "2:255"]')
        spec = {"type": "refs_list", "expected": ["2:255", "112:1"]}
        assert not run_checks(resp, [spec])[0]["passed"]

    def test_not_json_fails(self):
        resp = _resp("2:255 and 112:1")
        spec = {"type": "refs_list", "expected": ["2:255", "112:1"]}
        assert not run_checks(resp, [spec])[0]["passed"]
