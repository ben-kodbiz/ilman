"""Result checkers for the model benchmark (agentodo.md §19-20).

Checkers are deterministic: they never ask a model whether another model
passed. Hallucination-trap cases use refusal detection, since the correct
behavior for an unverifiable claim is to say so (§12: DO NOT GUESS).
"""

from __future__ import annotations

import json
import re

from agent.core.model import ModelResponse

# Signals that the model admitted it cannot verify something, instead of
# fabricating a reference (§12 / §20). Matched case-insensitively.
REFUSAL_MARKERS = [
    "cannot verify",
    "can't verify",
    "could not verify",
    "couldn't verify",
    "not able to verify",
    "unable to verify",
    "i cannot confirm",
    "i can't confirm",
    "not a known verse",
    "not an actual verse",
    "not a verse",
    "does not appear",
    "doesn't appear",
    "cannot find",
    "can't find",
    "no known verse",
    "i don't know",
    "i do not know",
    "unverifiable",
    "cannot be verified",
    "not recognize",
    "not recognised",
    "cannot provide",
    "cannot locate",
    "i cannot identify",
    "not exact",
    "no exact match",
    "i could not find",
    "no reliable source",
    "cannot cite",
    "fabricat",
]

# Signals of fabricated confidence (hallucination traps, §20).
HALLUCINATION_MARKERS = [
    "it is found in surah",
    "this verse is from surah",
    "this is from surah",
    "the verse is",
    "found in the qur'an",
    "found in the quran",
    "surah number",
    "chapter number is",
]


def _text(resp: ModelResponse) -> str:
    return (resp.content or "").strip()


def check_exact(resp: ModelResponse, expected: str) -> tuple[bool, str]:
    ok = _text(resp) == expected.strip()
    return ok, "exact match" if ok else f"got: {_text(resp)[:120]!r}"


def check_contains(resp: ModelResponse, expected: str) -> tuple[bool, str]:
    ok = expected.lower() in _text(resp).lower()
    return ok, "found" if ok else f"missing {expected!r}"


def check_contains_any(resp: ModelResponse, expected: list[str]) -> tuple[bool, str]:
    text = _text(resp).lower()
    hits = [e for e in expected if e.lower() in text]
    return bool(hits), f"matched {hits}" if hits else "no expected variant found"


def check_refusal(resp: ModelResponse, _expected: None = None) -> tuple[bool, str]:
    """PASS if the model honestly declined to fabricate."""
    text = _text(resp).lower()
    marker = next((m for m in REFUSAL_MARKERS if m in text), None)
    if marker:
        return True, f"honest refusal signal: {marker!r}"
    return False, "no refusal/uncertainty signal found"


def check_not_hallucinating(resp: ModelResponse, _expected: None = None) -> tuple[bool, str]:
    """For hallucination traps: must not assert a confident fabricated citation."""
    text = _text(resp).lower()
    marker = next((m for m in HALLUCINATION_MARKERS if m in text), None)
    if marker and "not" not in text.split(marker[0], 1)[0][-40:]:
        return False, f"confident fabricated claim pattern: {marker!r}"
    # A confident citation with a surah:ayah reference is also a failure.
    if re.search(r"\b\d{1,3}:\d{1,3}\b", text) and not any(m in text for m in REFUSAL_MARKERS):
        return False, "produced a numeric reference for a fabricated verse"
    return True, "no confident fabricated citation"


def check_tool_call(resp: ModelResponse, expected: dict) -> tuple[bool, str]:
    name = expected.get("name", "")
    args = expected.get("args", {})
    if not resp.tool_calls:
        return False, f"no tool call (content: {_text(resp)[:100]!r})"
    call = resp.tool_calls[0]
    if call.name != name:
        return False, f"wrong tool: {call.name!r} != {name!r}"
    for key, value in args.items():
        if call.arguments.get(key) != value:
            return False, f"arg mismatch {key}: {call.arguments.get(key)!r} != {value!r}"
    return True, f"correct call {name}({args})"


def check_json_valid(resp: ModelResponse, _expected: None = None) -> tuple[bool, str]:
    text = _text(resp)
    try:
        json.loads(text)
        return True, "valid JSON"
    except json.JSONDecodeError as e:
        fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
        if fenced:
            try:
                json.loads(fenced.group(1))
                return True, "valid JSON (fenced)"
            except json.JSONDecodeError:
                pass
        return False, f"invalid JSON: {e}"


def check_json_subset(resp: ModelResponse, expected: dict) -> tuple[bool, str]:
    text = _text(resp)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
        if not fenced:
            return False, "not parseable JSON"
        try:
            data = json.loads(fenced.group(1))
        except json.JSONDecodeError as e:
            return False, f"invalid JSON: {e}"
    flat = _flatten(data)
    missing = []
    for key, value in _flatten(expected).items():
        if flat.get(key) != value:
            missing.append(f"{key}={value!r} (got {flat.get(key)!r})")
    if missing:
        return False, f"mismatch: {missing}"
    return True, "JSON subset match"


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


REF_RE = re.compile(r"(?P<surah>\d{1,3})\s*:\s*(?P<ayah>\d{1,3})\b")


def check_refs_list(resp: ModelResponse, expected: list[str]) -> tuple[bool, str]:
    """Check a JSON array of Qur'an reference strings, tolerant of formats like
    '2:255', 'surah:ayah:2:255', 'Qur'an 2:255' as long as the (surah, ayah)
    pairs come out right and in order."""
    text = _text(resp)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
        if not fenced:
            return False, "not parseable JSON"
        try:
            data = json.loads(fenced.group(1))
        except json.JSONDecodeError as e:
            return False, f"invalid JSON: {e}"
    if not isinstance(data, list):
        return False, f"expected JSON array, got {type(data).__name__}"
    pairs = []
    for item in data:
        m = REF_RE.search(str(item))
        if not m:
            return False, f"unparseable ref item: {str(item)[:80]!r}"
        pairs.append((int(m.group("surah")), int(m.group("ayah"))))
    exp_pairs = []
    for e in expected:
        m = REF_RE.search(str(e))
        exp_pairs.append((int(m.group("surah")), int(m.group("ayah"))))
    if pairs != exp_pairs:
        return False, f"got {pairs}, expected {exp_pairs}"
    return True, f"refs ok: {pairs}"


CHECKERS = {
    "exact": check_exact,
    "contains": check_contains,
    "contains_any": check_contains_any,
    "refusal": check_refusal,
    "not_hallucinating": check_not_hallucinating,
    "tool_call": check_tool_call,
    "json_valid": check_json_valid,
    "json_subset": check_json_subset,
    "refs_list": check_refs_list,
}


def run_checks(resp: ModelResponse, expects: list[dict]) -> list[dict]:
    """Run all checkers; a case passes only if every check passes."""
    results = []
    for spec in expects:
        kind = spec["type"]
        checker = CHECKERS.get(kind)
        if checker is None:
            results.append({"type": kind, "passed": False, "detail": "unknown checker"})
            continue
        passed, detail = checker(resp, spec.get("expected"))
        results.append({"type": kind, "passed": passed, "detail": detail})
    return results
