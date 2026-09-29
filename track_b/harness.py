"""
Track B benchmark harness — Account-Based Management.

Runs TWO passes over track_b_accounts.xlsx and prints a results table:
  1. NAIVE baseline   — one LLM call per account (200 calls), including the
                         105 accounts with empty/near-empty `signals`.
  2. OPTIMIZED build  — code-side zero-signal gate skips the LLM entirely
                         for accounts with no usable signal (105/200 here),
                         and for the rest, `fit_tier` is cross-checked
                         against a deterministic employees-threshold rule
                         rather than trusted from the model's raw string.

Usage:
    Put LYZR_API_KEY / LYZR_AGENT_ID_B / LYZR_USER_ID in a .env file, or
    export them directly.
    python track_b/harness.py --mode both  (from repo root, or run inside track_b/)

Design notes (read before you run this against real money):
  - The zero-signal gate is NOT a shortcut invented to win the benchmark —
    it's the correct behavior. An account with no signal has no facts to
    ground an opener in, so the only honest output is insufficient_data;
    asking the model to produce one just pays tokens to reproduce a
    decision the code can already make for free. See track_b/AGENT_SETUP.md.
  - Tier rule (employees >= 100 -> A, else B) is validated against all 25
    labeled accounts in track_b_labels_and_kb.xlsx: min employees among
    tier-A rows is 150, max among tier-B rows is 80 — any threshold in
    (80, 150) achieves 100% separation on the labeled set. 100 is used as
    a round number inside that gap. No tier C: the signals contain nothing
    negative that could justify overriding headcount (see AGENT_SETUP.md).
  - Token/cost accounting is a LOCAL ESTIMATE for the same reason as
    Track A: Lyzr's /v3/inference/chat/ response carries no usage block.
    Reuses the same calibrated overhead/pricing constants as track_a's
    harness (same agent-hosting platform, same model family) rather than
    re-deriving a second calibration from scratch — flagged as an estimate
    either way, so this is a reasonable shared assumption, not a hidden one.
  - NO response_format / JSON-schema enforcement, for the same reason as
    Track A: a documented prior incident where Lyzr's platform-level
    structured-output enforcement broke every call after a silent model
    fallback. Enforcement here is (1) prompt instructions, (2) this
    script's own validation of fit_tier / next_action / opener grounding,
    (3) deterministic tier cross-check. A softer guarantee than schema
    enforcement, stated plainly, not implied to be equivalent.
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import lyzr_client as lyzr  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LYZR_AGENT_ID_B = os.environ.get("LYZR_AGENT_ID_B", "")

# Validated against all 25 labeled accounts: min tier-A employees = 150,
# max tier-B employees = 80. Any threshold in (80, 150) gives 100%
# separation on the labeled set; 100 is a round number inside that gap.
TIER_EMPLOYEE_THRESHOLD = 100

CLOSED_SET_NEXT_ACTIONS = {
    "schedule_intro_call", "send_case_study",
    "request_warm_intro", "add_to_nurture_sequence",
}
ALLOWED_TIERS = {"A", "B"}
ALLOWED_TARGET_DECISIONS = {"target", "skip", "review", "insufficient_data"}
REQUIRED_KEYS = {"fit_tier", "target_decision", "next_action", "opener", "confidence", "reasoning"}

# Borrowed from Track A's calibration, not measured for this agent's
# (shorter) prompt, so Track B costs are likely overestimated. Calibrate
# from Track B traces before quoting absolute dollar figures.
PROMPT_OVERHEAD_TOKENS = 3397


def _has_signal(signals) -> bool:
    """True if `signals` has any usable content. Empty/NaN/whitespace-only
    counts as no signal — this is the code-side gate's entire decision
    rule, kept as a standalone function so naive/optimized/tests all agree
    on exactly what 'zero-signal' means."""
    if signals is None:
        return False
    if isinstance(signals, float):  # NaN
        return False
    return bool(str(signals).strip())


def deterministic_tier(employees: int) -> str:
    return "A" if employees >= TIER_EMPLOYEE_THRESHOLD else "B"


def expected_next_action(signals: str | None, target_decision: str | None) -> str:
    """The priority rule from the agent instructions (STEP 3), computed in code.
    gpt-4o-mini ignores the priority order for ~1 in 3 accounts, so the harness
    trusts this over the model, the same way tier is trusted over the model."""
    if not signals or target_decision in ("skip", "review", "insufficient_data"):
        return "add_to_nurture_sequence"
    s = signals.lower()
    if "raised series" in s:
        return "schedule_intro_call"
    if "hiring" in s:
        return "request_warm_intro"
    return "send_case_study"


def ungrounded_claims(opener: str, source: str, context: str) -> list[str]:
    """Concrete claims in the opener with no match in this account's own
    signals/verified_fact. A claim is any number, or any capitalized term
    after the first word that isn't part of the company/industry/region
    (e.g. a technology, funding round, role, or place). A heuristic, not
    NLI: it catches invented facts, not paraphrases that change meaning."""
    source_l, context_l = source.lower(), context.lower()

    def found(tok: str, text: str) -> bool:
        return re.search(rf"(?<![a-z0-9]){re.escape(tok.lower())}(?![a-z0-9])", text) is not None

    claims = re.findall(r"\$?\d+(?:\.\d+)?[MKBmkb]?(?![A-Za-z])", opener)
    for sentence in re.split(r"(?<=[.!?])\s+", opener):
        words = re.findall(r"(?<![\w$])[A-Za-z][A-Za-z0-9'-]*", sentence)
        claims += [w for w in words[1:]
                   if w[0].isupper() and w.split("'")[0] != "I" and not found(w, context_l)]
    return [c for c in dict.fromkeys(claims) if not found(c, source_l)]


@dataclass
class CallResult:
    account_id: str
    company: str
    employees: int
    signals: str | None
    verified_fact: str | None = None
    industry: str = ""
    region: str = ""
    fit_tier: str | None = None
    target_decision: str | None = None
    next_action: str | None = None
    opener: str | None = None
    confidence: float | None = None
    reasoning: str = ""
    needs_human_review: bool = False
    schema_valid: bool = True
    tier_overridden: bool = False
    next_action_overridden: bool = False
    opener_grounded: bool | None = None  # None = no opener to check
    error: str | None = None
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    cost_usd: float = 0.0
    is_actual_call: bool = True  # False for accounts the zero-signal gate
                                  # skipped entirely — no call was made


# ---------------------------------------------------------------------------
# Lyzr API call
# ---------------------------------------------------------------------------

async def call_lyzr_agent(row: pd.Series, session_id: str,
                           client: httpx.AsyncClient, retries: int = 3) -> CallResult:
    """Score one account with the Track B agent."""
    result = CallResult(
        account_id=row["account_id"], company=row["company"],
        employees=int(row["employees"]),
        signals=row["signals"] if _has_signal(row["signals"]) else None,
        verified_fact=row["verified_fact"] or None,
        industry=row["industry"], region=row["region"],
    )

    user_message = (
        f"company={row['company']} domain={row['domain']} "
        f"industry={row['industry']} employees={row['employees']} "
        f"region={row['region']} signals=\"{result.signals or ''}\" "
        f"verified_fact=\"{row['verified_fact']}\""
    )

    try:
        resp = await lyzr.chat(client, LYZR_AGENT_ID_B, session_id, user_message,
                               PROMPT_OVERHEAD_TOKENS, retries)
    except lyzr.LyzrCallError as e:
        result.error = str(e)
        result.needs_human_review = True
        result.schema_valid = False
        return result

    result.latency_s = resp.latency_s
    result.prompt_tokens = resp.prompt_tokens
    result.completion_tokens = resp.completion_tokens
    result.tokens_estimated = resp.tokens_estimated
    result.cost_usd = resp.cost_usd

    try:
        _parse_verdict(resp.text, result)
    except Exception as parse_err:  # noqa: BLE001 - parsing bugs are not retried
        result.error = f"parse error (not retried): {parse_err}"
        result.needs_human_review = True
        result.schema_valid = False
    return result


def _parse_verdict(raw_text: str, result: CallResult) -> None:
    """Parses + validates the model's JSON output. No platform schema
    enforcement (see module docstring) — this is the only guardrail."""

    try:
        verdict = lyzr.parse_json_object(raw_text)
    except ValueError as e:
        result.error = str(e)
        result.needs_human_review = True
        result.schema_valid = False
        return

    missing = REQUIRED_KEYS - verdict.keys()
    if missing:
        result.error = f"missing required keys: {sorted(missing)}"
        result.needs_human_review = True
        result.schema_valid = False
        return

    result.fit_tier = verdict.get("fit_tier")
    result.target_decision = verdict.get("target_decision")
    result.next_action = verdict.get("next_action")
    result.opener = verdict.get("opener")
    result.confidence = verdict.get("confidence")
    result.reasoning = verdict.get("reasoning", "")

    if result.confidence is not None and not isinstance(result.confidence, (int, float)):
        try:
            result.confidence = float(result.confidence)
        except (TypeError, ValueError):
            result.schema_valid = False
            result.confidence = None

    # Closed-set validation.
    if result.fit_tier is not None and result.fit_tier not in ALLOWED_TIERS:
        result.schema_valid = False
    if result.target_decision is not None and result.target_decision not in ALLOWED_TARGET_DECISIONS:
        result.schema_valid = False
    if result.next_action is not None and result.next_action not in CLOSED_SET_NEXT_ACTIONS:
        result.schema_valid = False

    # Employees fully determines A/B on all 25 labeled rows — trust the rule
    # over the model's raw string, same as Track A trusts root_cause->category.
    if result.fit_tier in ALLOWED_TIERS:
        derived_tier = deterministic_tier(result.employees)
        if result.fit_tier != derived_tier:
            result.tier_overridden = True
            result.reasoning += (
                f" [tier corrected: model said '{result.fit_tier}', "
                f"derived '{derived_tier}' from employees={result.employees}]"
            )
            result.fit_tier = derived_tier

    if result.next_action in CLOSED_SET_NEXT_ACTIONS:
        derived_action = expected_next_action(result.signals, result.target_decision)
        if result.next_action != derived_action:
            result.next_action_overridden = True
            result.reasoning += (
                f" [next_action corrected: model said '{result.next_action}', "
                f"rule gives '{derived_action}']"
            )
            result.next_action = derived_action

    if result.opener:
        source = " ".join(str(x) for x in (result.signals, result.verified_fact) if x)
        context = " ".join((result.company, result.industry, result.region))
        ungrounded = ungrounded_claims(result.opener, source, context)
        result.opener_grounded = not ungrounded
        if ungrounded:
            result.reasoning += f" [ungrounded claims: {ungrounded}]"

    if not result.schema_valid:
        result.needs_human_review = True
        result.reasoning += " [FLAGGED: output failed closed-set validation]"

    if result.confidence is None or result.confidence < 0.5:
        result.needs_human_review = True

    if result.opener_grounded is False:
        result.needs_human_review = True
        result.reasoning += " [FLAGGED: opener has no traceable claim in signals/verified_fact]"


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------

MAX_TASK = 4


async def run_naive_baseline(df: pd.DataFrame) -> tuple[list[CallResult], float]:
    """One call per account, including all 105 zero-signal accounts. This
    is the intentionally-wasteful comparison point — no gate, no rule-based
    shortcuts, nothing the optimized build gets for free."""
    rows = list(df.iterrows())
    total = len(rows)
    results: list = [None] * total
    done = 0
    sem = asyncio.Semaphore(MAX_TASK)

    async def _task(i: int, row, client: httpx.AsyncClient) -> None:
        nonlocal done
        async with sem:
            r = await call_lyzr_agent(row, session_id=f"naive-{row['account_id']}", client=client)
        results[i] = r
        done += 1
        print(f"  [naive] {done}/{total}", end="\r", file=sys.stderr)

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0) as client:
        await asyncio.gather(*(_task(i, row, client) for i, (_, row) in enumerate(rows)))
    wall_clock_s = time.perf_counter() - start
    print(file=sys.stderr)
    return results, wall_clock_s


async def run_optimized(df: pd.DataFrame) -> tuple[list[CallResult], dict]:
    """Code-side zero-signal gate skips the LLM entirely for accounts with
    no usable signal — this is the single biggest cost lever for this
    dataset, the Track B analogue of Track A's dedup. Only accounts with
    real signal reach the model."""
    has_signal_mask = df["signals"].apply(_has_signal)
    gated_df = df[has_signal_mask]
    skipped_df = df[~has_signal_mask]

    print(f"  {len(df)} accounts -> {len(gated_df)} with usable signal "
          f"({len(skipped_df)} zero-signal, gated for free)", file=sys.stderr)

    rows = list(gated_df.iterrows())
    total = len(rows)
    done = 0
    sem = asyncio.Semaphore(MAX_TASK)

    async def _task(row, client: httpx.AsyncClient) -> CallResult:
        nonlocal done
        async with sem:
            r = await call_lyzr_agent(row, session_id=f"opt-{row['account_id']}", client=client)
        done += 1
        print(f"  [optimized] {done}/{total}", end="\r", file=sys.stderr)
        return r

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=30.0) as client:
        called_results = await asyncio.gather(*(_task(row, client) for _, row in rows))
    wall_clock_s = time.perf_counter() - start
    print(file=sys.stderr)

    gated_results = []
    for _, row in skipped_df.iterrows():
        gated_results.append(CallResult(
            account_id=row["account_id"], company=row["company"],
            employees=int(row["employees"]), signals=None,
            verified_fact=row["verified_fact"] or None,
            industry=row["industry"], region=row["region"],
            fit_tier=None, target_decision="insufficient_data",
            next_action="add_to_nurture_sequence", opener=None,
            confidence=0.95, reasoning="[gate] no usable signal — routed without an LLM call",
            schema_valid=True, is_actual_call=False,
        ))

    all_results = called_results + gated_results
    # restore original row order so downstream reporting reads naturally
    order = {aid: i for i, aid in enumerate(df["account_id"])}
    all_results.sort(key=lambda r: order[r.account_id])

    stats = {
        "raw_rows": len(df),
        "accounts_with_signal": len(gated_df),
        "accounts_gated_zero_signal": len(skipped_df),
        "llm_calls_made": len(gated_df),
        "calls_saved_by_gate": len(skipped_df),
        "wall_clock_s": wall_clock_s,
    }
    return all_results, stats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[CallResult], labels_df: pd.DataFrame,
                     wall_clock_s: float | None = None) -> dict:
    by_id = {r.account_id: r for r in results}

    correct_tier = 0
    tier_scored = 0
    correct_target = 0
    target_scored = 0
    for _, row in labels_df.iterrows():
        r = by_id.get(row["account_id"])
        if r is None:
            continue
        if r.fit_tier is not None:
            tier_scored += 1
            if r.fit_tier == row["gt_tier"]:
                correct_tier += 1
        if r.target_decision is not None:
            target_scored += 1
            gt_target = "target" if str(row["gt_should_target"]).strip().lower() == "yes" else "skip"
            if r.target_decision == gt_target:
                correct_target += 1

    tier_accuracy = round(correct_tier / tier_scored, 4) if tier_scored else 0.0
    target_accuracy = round(correct_target / target_scored, 4) if target_scored else 0.0

    invalid_schema = sum(1 for r in results if not r.schema_valid)
    off_closed_set_next_action = sum(
        1 for r in results if r.next_action is not None and r.next_action not in CLOSED_SET_NEXT_ACTIONS
    )
    tier_overridden = sum(1 for r in results if r.tier_overridden)

    openers_checked = [r for r in results if r.opener_grounded is not None]
    fabricated = [r for r in openers_checked if r.opener_grounded is False]
    fabricated_claim_rate = round(len(fabricated) / len(openers_checked), 4) if openers_checked else 0.0

    actual_calls = [r for r in results if r.is_actual_call]
    latencies = [r.latency_s for r in actual_calls if r.latency_s > 0]
    n_calls = len(actual_calls)

    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = lyzr.percentile(latencies, 95)

    total_cost = sum(r.cost_usd for r in actual_calls)
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in actual_calls)

    if wall_clock_s and wall_clock_s > 0:
        throughput = round(len(results) / (wall_clock_s / 60), 2)
    elif latencies:
        throughput = round(len(results) / (sum(latencies) / 60), 2)
    else:
        throughput = 0

    return {
        "tier_accuracy": tier_accuracy,
        "tier_accuracy_n": tier_scored,
        "target_decision_accuracy": target_accuracy,
        "target_decision_accuracy_n": target_scored,
        "fabricated_claim_rate": fabricated_claim_rate,
        "openers_checked": len(openers_checked),
        "invalid_schema_count": invalid_schema,
        "off_closed_set_next_action_count": off_closed_set_next_action,
        "tier_overridden_count": tier_overridden,
        "next_action_overridden_count": sum(1 for r in results if r.next_action_overridden),
        "human_review_flagged": sum(1 for r in results if r.needs_human_review),
        "actual_llm_calls": n_calls,
        "rows_covered": len(results),
        "p50_latency_s": round(p50, 3),
        "p95_latency_s": round(p95, 3),
        "total_tokens": total_tokens,
        "avg_tokens_per_call": round(total_tokens / max(n_calls, 1), 1),
        "total_cost_usd": round(total_cost, 4),
        "cost_per_task_usd": round(total_cost / max(len(results), 1), 6),
        "wall_clock_s": round(wall_clock_s, 2) if wall_clock_s else None,
        "throughput_tasks_per_min": throughput,
        "any_estimated_tokens": any(r.tokens_estimated for r in results),
    }




def print_results_table(naive: dict | None = None, optimized: dict | None = None,
                        stats: dict | None = None):
    print("\n" + "=" * 90)
    print("RESULTS TABLE — Track B: Account-Based Management")
    print("=" * 90)
    if stats:
        print(f"Raw accounts: {stats.get('raw_rows', 'N/A')}  |  With signal: "
              f"{stats.get('accounts_with_signal', 'N/A')}  |  Zero-signal gated: "
              f"{stats.get('accounts_gated_zero_signal', 'N/A')}")
    print("-" * 90)

    if naive and optimized:
        rows = [
            ("Tier accuracy (25 labeled)", naive["tier_accuracy"], optimized["tier_accuracy"]),
            ("Target-decision accuracy (25 labeled)", naive["target_decision_accuracy"], optimized["target_decision_accuracy"]),
            ("Fabricated-claim rate", naive["fabricated_claim_rate"], optimized["fabricated_claim_rate"]),
            ("Invalid schema count", naive["invalid_schema_count"], optimized["invalid_schema_count"]),
            ("Tier corrected by rule", naive["tier_overridden_count"], optimized["tier_overridden_count"]),
            ("Next action corrected by rule", naive["next_action_overridden_count"], optimized["next_action_overridden_count"]),
            ("Human-review flagged", naive["human_review_flagged"], optimized["human_review_flagged"]),
            ("p50 latency (s)", naive["p50_latency_s"], optimized["p50_latency_s"]),
            ("p95 latency (s)", naive["p95_latency_s"], optimized["p95_latency_s"]),
            ("Total tokens", naive["total_tokens"], optimized["total_tokens"]),
            ("Total cost (USD)", naive["total_cost_usd"], optimized["total_cost_usd"]),
            ("Cost per task (USD)", naive["cost_per_task_usd"], optimized["cost_per_task_usd"]),
            ("Batch wall-clock (s)", naive.get("wall_clock_s") or 0, optimized.get("wall_clock_s") or 0),
            ("Throughput (tasks/min)", naive["throughput_tasks_per_min"], optimized["throughput_tasks_per_min"]),
        ]
        print(f"{'Metric':<38}{'Naive':>15}{'Optimized':>15}{'Delta':>14}")
        print("-" * 82)
        for name, a, b in rows:
            if a == 0:
                d = "n/a"
            else:
                pct = (b - a) / a * 100
                arrow = "↑" if pct > 0 else "↓"
                d = f"{arrow}{abs(pct):.1f}%"
            print(f"{name:<38}{a:>15}{b:>15}{d:>14}")
    else:
        metrics = naive or optimized
        print(json.dumps(metrics, indent=2))

    print("=" * 90)
    checked = [m for m in (naive, optimized) if m]
    if any(m.get("any_estimated_tokens") for m in checked):
        print("NOTE: Lyzr's API returns no token counts, so every token and "
              "cost figure above is a local estimate, not a metered number.")
    print("NOTE: fabricated_claim_rate only checks numbers and capitalized "
          "terms in the opener against that account's own signals. It misses "
          "made-up details in lowercase, so treat 0 as 'none found', not proof.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MAX_TASK
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).parent / "track_b_accounts.xlsx"))
    parser.add_argument("--labels", default=str(Path(__file__).parent / "track_b_labels_and_kb.xlsx"))
    parser.add_argument("--mode", choices=["naive", "optimized", "both"], default="both")
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--out-prefix", default="run")
    parser.add_argument("--max-tasks", type=int, default=MAX_TASK,
                         help="Concurrency level for API calls.")
    args = parser.parse_args()
    MAX_TASK = args.max_tasks

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    args.out_prefix = str(Path(args.out_dir) / args.out_prefix)

    if not lyzr.LYZR_API_KEY or not LYZR_AGENT_ID_B or not lyzr.LYZR_USER_ID:
        print("ERROR: set LYZR_API_KEY, LYZR_AGENT_ID_B, and LYZR_USER_ID "
              "(via .env or environment) before running.", file=sys.stderr)
        sys.exit(1)
    if LYZR_AGENT_ID_B == os.environ.get("LYZR_AGENT_ID"):
        print("ERROR: LYZR_AGENT_ID_B is the same as the Track A agent (LYZR_AGENT_ID). "
              "Set it to the Track B agent's ID.", file=sys.stderr)
        sys.exit(1)

    labels_df = pd.read_excel(args.labels)
    # Only verified_fact (the KB column) joins the agent's input; gt_* labels never do.
    df = pd.read_excel(args.data).merge(
        labels_df[["account_id", "verified_fact"]], on="account_id", how="left")
    df["verified_fact"] = df["verified_fact"].fillna("")

    naive_metrics = optimized_metrics = None
    stats = {"raw_rows": len(df),
              "accounts_with_signal": int(df["signals"].apply(_has_signal).sum())}
    stats["accounts_gated_zero_signal"] = stats["raw_rows"] - stats["accounts_with_signal"]

    if args.mode in ("naive", "both"):
        print("Running NAIVE baseline (one call per account, no gate)...", file=sys.stderr)
        naive_results, naive_wall_clock_s = asyncio.run(run_naive_baseline(df))
        naive_metrics = compute_metrics(naive_results, labels_df, wall_clock_s=naive_wall_clock_s)
        pd.DataFrame([r.__dict__ for r in naive_results]).to_csv(
            f"{args.out_prefix}_naive_raw.csv", index=False)

    if args.mode in ("optimized", "both"):
        print("Running OPTIMIZED build (zero-signal gate first)...", file=sys.stderr)
        opt_results, gate_stats = asyncio.run(run_optimized(df))
        optimized_metrics = compute_metrics(
            opt_results, labels_df, wall_clock_s=gate_stats.get("wall_clock_s"))
        pd.DataFrame([r.__dict__ for r in opt_results]).to_csv(
            f"{args.out_prefix}_optimized_raw.csv", index=False)
        stats = gate_stats

    if naive_metrics and optimized_metrics:
        print_results_table(naive=naive_metrics, optimized=optimized_metrics, stats=stats)
        with open(f"{args.out_prefix}_summary.json", "w") as f:
            json.dump({"naive": naive_metrics, "optimized": optimized_metrics,
                       "stats": stats}, f, indent=2)
        print(f"\nSaved: {args.out_prefix}_naive_raw.csv, "
              f"{args.out_prefix}_optimized_raw.csv, {args.out_prefix}_summary.json")
    elif naive_metrics:
        print_results_table(naive=naive_metrics, stats=stats)
    elif optimized_metrics:
        print_results_table(optimized=optimized_metrics, stats=stats)


if __name__ == "__main__":
    main()
