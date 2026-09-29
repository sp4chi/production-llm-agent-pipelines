import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from track_b import harness

DATA = Path(harness.__file__).parent


def _accounts():
    labels = pd.read_excel(DATA / "track_b_labels_and_kb.xlsx")
    df = pd.read_excel(DATA / "track_b_accounts.xlsx").merge(
        labels[["account_id", "verified_fact"]], on="account_id", how="left")
    df["verified_fact"] = df["verified_fact"].fillna("")
    return df, labels


@pytest.mark.parametrize("value, expected", [
    (None, False), (float("nan"), False), ("", False), ("   ", False), ("uses AWS", True),
])
def test_has_signal(value, expected):
    assert harness._has_signal(value) is expected


def test_tier_threshold_boundary():
    assert harness.deterministic_tier(100) == "A"
    assert harness.deterministic_tier(99) == "B"


def test_tier_rule_matches_all_labeled_accounts():
    df, labels = _accounts()
    merged = labels.merge(df[["account_id", "employees"]], on="account_id")
    predicted = merged["employees"].apply(harness.deterministic_tier)
    assert (predicted == merged["gt_tier"]).all()


def test_ground_truth_columns_never_reach_agent_input():
    df, _ = _accounts()
    assert not [c for c in df.columns if c.startswith("gt_")]
    assert (df["verified_fact"] != "").sum() == 25


CTX = "X Co fintech EMEA"


@pytest.mark.parametrize("opener, source, context, expected", [
    ("Congrats to Quarry Robotics on the $30M Series B in 2026 — would a short call help?",
     "uses Snowflake | raised Series B $30M in 2026", "Quarry Robotics devtools EMEA", []),
    ("Saw Fernhill Freight is hiring 2 platform engineers — happy to share how teams on GCP do it.",
     "hiring 2 platform engineers | uses GCP", "Fernhill Freight logistics APAC", []),
    ("I noticed X Co is hiring 3 ML engineers — I'd love to share insights.",
     "hiring 3 ML engineers", CTX, []),
    ("Saw X Co just launched an EU region.", "launched an EU region recently", CTX, []),
    ("Congrats on the $22M Series B, X Co! Would a call help?",
     "raised Series B $22M in 2026", CTX, []),
    ("Congrats on the $22M Series B! Saw you use Snowflake too.",
     "raised Series B $22M in 2026", CTX, ["Snowflake"]),
    ("Congrats on the $50M Series B!", "raised Series B $22M in 2026", CTX, ["$50M"]),
    ("Congrats on the $22M Series C!", "raised Series B $22M in 2026", CTX, ["C"]),
    ("Saw you moved to Snowflake.", "uses Databricks", CTX, ["Snowflake"]),
    ("Congrats on expanding into Asia.", "uses AWS", CTX, ["Asia"]),
    ("Saw you're hiring 5 ML engineers.", "hiring 3 ML engineers", CTX, ["5"]),
])
def test_ungrounded_claims(opener, source, context, expected):
    assert harness.ungrounded_claims(opener, source, context) == expected


def _verdict(**overrides):
    v = {"fit_tier": "B", "target_decision": "target", "next_action": "request_warm_intro",
         "opener": "Saw Vertex Logistics is hiring 3 ML engineers.",
         "confidence": 0.85, "reasoning": "r"}
    v.update(overrides)
    return json.dumps(v)


def _parse(raw, employees=12):
    r = harness.CallResult(account_id="A002", company="Vertex Logistics", employees=employees,
                           signals="hiring 3 ML engineers | uses AWS",
                           verified_fact="hiring 3 ML engineers",
                           industry="logistics", region="North America")
    harness._parse_verdict(raw, r)
    return r


def test_valid_verdict_passes():
    r = _parse(_verdict())
    assert r.schema_valid and r.opener_grounded and not r.needs_human_review


def test_wrong_tier_is_corrected():
    r = _parse(_verdict(fit_tier="A"))
    assert r.fit_tier == "B"
    assert r.tier_overridden


def test_tier_c_is_rejected():
    assert not _parse(_verdict(fit_tier="C")).schema_valid


@pytest.mark.parametrize("signals, decision, expected", [
    ("uses AWS | raised Series A $8M in 2026 | hiring 3 ML engineers", "target", "schedule_intro_call"),
    ("uses AWS | hiring a Head of AI", "target", "request_warm_intro"),
    ("launched a new API recently | uses GCP", "target", "send_case_study"),
    ("raised Series B $22M in 2026", "review", "add_to_nurture_sequence"),
    (None, "insufficient_data", "add_to_nurture_sequence"),
])
def test_expected_next_action(signals, decision, expected):
    assert harness.expected_next_action(signals, decision) == expected


def test_next_action_off_rule_is_corrected():
    r = _parse(_verdict(next_action="send_case_study"))
    assert r.next_action == "request_warm_intro"
    assert r.next_action_overridden
    assert r.schema_valid


def test_next_action_outside_closed_set_fails():
    assert not _parse(_verdict(next_action="send_linkedin_message")).schema_valid


def test_wrong_agent_schema_fails_loudly():
    track_a_style = json.dumps({"is_incident": False, "category": None, "root_cause": None,
                                "remediation": None, "confidence": 0.99, "reasoning": ""})
    r = _parse(track_a_style)
    assert not r.schema_valid
    assert "missing required keys" in r.error


def test_fabricated_opener_goes_to_review():
    r = _parse(_verdict(opener="Congrats on the $40M Series C!"))
    assert r.opener_grounded is False
    assert r.needs_human_review


def test_optimized_gate_skips_zero_signal_accounts(monkeypatch):
    called = []

    async def fake_call(row, session_id, client, retries=3):
        called.append(row["account_id"])
        return harness.CallResult(account_id=row["account_id"], company=row["company"],
                                  employees=int(row["employees"]), signals=row["signals"],
                                  latency_s=1.0, cost_usd=0.01)

    monkeypatch.setattr(harness, "call_lyzr_agent", fake_call)
    df, _ = _accounts()
    results, stats = asyncio.run(harness.run_optimized(df))

    assert len(called) == 95
    assert stats["calls_saved_by_gate"] == 105
    assert [r.account_id for r in results] == list(df["account_id"])
    gated = [r for r in results if not r.is_actual_call]
    assert len(gated) == 105
    assert all(r.target_decision == "insufficient_data" and r.opener is None for r in gated)
    assert all(r.cost_usd == 0 for r in gated)
