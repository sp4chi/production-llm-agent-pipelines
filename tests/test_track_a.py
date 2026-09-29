import asyncio
import json

import pandas as pd
import pytest

from track_a import harness


def _verdict(**overrides):
    v = {"is_incident": True, "category": "capacity", "root_cause": "rate_limit_breach",
         "remediation": "add_backpressure_and_request_quota_increase",
         "confidence": 0.9, "reasoning": "r"}
    v.update(overrides)
    return json.dumps(v)


def _parse(raw, severity="WARN"):
    r = harness.CallResult(event_id="E1", message="m")
    harness._parse_verdict(raw, r, severity)
    return r


def test_valid_verdict_passes():
    r = _parse(_verdict())
    assert r.schema_valid and not r.needs_human_review
    assert r.root_cause == "rate_limit_breach"


def test_category_derived_from_root_cause():
    r = _parse(_verdict(category="performance"))
    assert r.category == "capacity"
    assert r.category_overridden


def test_free_form_remediation_fails_validation():
    r = _parse(_verdict(remediation="just restart it"))
    assert not r.schema_valid
    assert r.needs_human_review


def test_severity_gate_blocks_error_as_noise():
    raw = _verdict(is_incident=False, category=None, root_cause=None, remediation=None)
    r = _parse(raw, severity="ERROR")
    assert r.is_incident is True
    assert r.needs_human_review


def test_string_boolean_is_coerced():
    assert _parse(_verdict(is_incident="true")).is_incident is True
    assert _parse(_verdict(is_incident="false")).is_incident is False


def test_low_confidence_goes_to_review():
    assert _parse(_verdict(confidence=0.3)).needs_human_review


def test_unparseable_output_is_flagged():
    r = _parse("I think this is a rate limit problem.")
    assert not r.schema_valid
    assert r.error == "could not parse JSON from model output"


def test_every_root_cause_maps_to_allowed_values():
    for root, cat in harness.ROOT_CAUSE_TO_CATEGORY.items():
        assert root in harness.ALLOWED_ROOT_CAUSES
        assert cat in harness.ALLOWED_CATEGORIES
        assert harness.ROOT_CAUSE_TO_REMEDIATION[root] in harness.ALLOWED_REMEDIATIONS


def test_optimized_calls_once_per_unique_message(monkeypatch):
    calls = []

    async def fake_call(service, severity, message, session_id, client, retries=3):
        calls.append(message)
        return harness.CallResult(event_id=session_id, message=message, is_incident=False,
                                  latency_s=1.0, prompt_tokens=100, cost_usd=0.01)

    monkeypatch.setattr(harness, "call_lyzr_agent", fake_call)
    df = pd.DataFrame({
        "event_id": ["E1", "E2", "E3", "E4"],
        "service": ["api"] * 4,
        "severity": ["INFO"] * 4,
        "message": ["heartbeat ok", "heartbeat ok", "cache warm", "heartbeat ok"],
    })
    results, stats = asyncio.run(harness.run_optimized(df))

    assert sorted(calls) == ["cache warm", "heartbeat ok"]
    assert [r.event_id for r in results] == ["E1", "E2", "E3", "E4"]
    assert [r.is_actual_call for r in results] == [True, False, True, False]
    assert sum(r.cost_usd for r in results) == pytest.approx(0.02)
    assert stats["calls_saved_by_dedup"] == 2
