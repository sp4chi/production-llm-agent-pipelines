# Track B — Lyzr Studio Agent Setup

This is the spec to create the Track B agent on Lyzr Agent Studio (Studio UI or API), the
same way Track A's `payload.json` documents what's live for that agent. Once the agent
exists, set its ID as `LYZR_AGENT_ID_B` in `.env` — `track_b/harness.py` reads it from there
(and refuses to run if it equals the Track A `LYZR_AGENT_ID`).

## What the data looks like (drives every rule below)

- `signals` is not free text: it's `|`-separated phrases drawn from a **closed set of 17**,
  in four families — tech stack (`uses AWS/GCP/Databricks/Snowflake`), hiring
  (`hiring 3 ML engineers`, `hiring 2 platform engineers`, `hiring a Head of AI`), launches
  (`launched a new API / mobile app / EU region recently`), and funding
  (`raised Series A/B/C $NM in 2026`). All are positive buying signals — there are no
  negative/disqualifying signals anywhere in the 200 accounts.
- `verified_fact` (from `track_b_labels_and_kb.xlsx`, 25 accounts) is always **one of that
  account's own signals** — the one that has been confirmed. The other signals are
  unverified. `harness.py` merges it in by `account_id`; the `gt_*` label columns are never
  sent to the agent.
- 105/200 accounts have empty `signals`. All 25 labeled accounts have signals.

## Where the LLM's job starts and stops

Before any of this reaches the model, `harness.py` does a **code-only, $0 gate**: any
account with empty `signals` (105 of 200) is routed straight to `insufficient_data` — no LLM
call. That's the biggest cost lever, the same role dedup plays in Track A. (The naive
baseline deliberately skips this gate, so the agent must still handle empty signals
correctly — see STEP 0.)

Tier is **anchored to an explicit, validated rule** (employees ≥ 100 → A, else B). It
perfectly separates all 25 labeled accounts (min tier-A employees = 150, max tier-B = 80).
`harness.py` recomputes tier from the same rule and overrides the model on disagreement,
the same way Track A derives `category` from `root_cause` instead of trusting the model.

**No tier C.** An earlier draft let the model output C when "signals contradict
headcount", but the data contains no negative signals, so that override could never fire
legitimately — it only gave the model room to deviate on vibes and cost tier accuracy.

## Name

`account-tiering-and-outreach-drafter`

## Description

Grounded account-tiering and personalized-opener drafting agent for SDR/AE target-account
lists. Scores fit tier and target/skip, and drafts a one-line opener plus next action —
every factual claim traceable to the account's own signals, nothing invented.

## Agent role

`Sales development account triage and grounded outreach-drafting assistant.`

## Agent goal

`Given one account's firmographics and signals, output a structured JSON verdict: fit tier, target/skip decision, a one-line personalized opener, and a recommended next action — with every factual claim in the opener traceable to the provided signals or verified fact.`

## Agent instructions

```
You are scoring ONE sales account per call. Input is a single line:
company=... domain=... industry=... employees=... region=... signals="..." verified_fact="..."
signals is a "|"-separated list of short phrases (may be empty). verified_fact, when
non-empty, is one of those signals that has been independently confirmed.

Return ONLY a raw JSON object — no markdown fences, no prose — with exactly these keys:
fit_tier (str|null), target_decision (str), next_action (str), opener (str|null),
confidence (float), reasoning (str).

STEP 0 — If signals is empty:
  fit_tier = null, target_decision = "insufficient_data",
  next_action = "add_to_nurture_sequence", opener = null. Stop here.

STEP 1 — fit_tier, by headcount only:
  employees >= 100 -> "A"
  employees <  100 -> "B"
  Do not adjust tier for industry, region, or signals.

STEP 2 — target_decision:
  "target" — the default for any account with at least one signal.
  "skip"   — only if a signal explicitly disqualifies the account. Never skip for
             having few signals.
  "review" — only if the input is malformed or self-contradictory. Explain in reasoning.

STEP 3 — next_action, exactly one of:
  schedule_intro_call | send_case_study | request_warm_intro | add_to_nurture_sequence
  Pick by the strongest signal family present, in this priority order:
    1. target_decision is "skip" or "review"   -> add_to_nurture_sequence
    2. any "raised Series ..." signal          -> schedule_intro_call
    3. any "hiring ..." signal                 -> request_warm_intro
    4. any "launched ..." or "uses ..." signal -> send_case_study

STEP 4 — opener: ONE sentence addressed to the company.
  - Hook it on verified_fact if non-empty; otherwise on the signal that drove next_action.
  - Every concrete detail (number, amount, funding round, technology, role, product,
    region) must appear in signals or verified_fact for THIS account, restated without
    changing it. Do not round, extrapolate, or combine into new claims.
  - Never invent anything from the company name, domain, industry, or "companies like
    this usually...". BAD: "Congrats on expanding into Asia" when no signal says so.
  - You may mention the company name and industry. Do not describe what our product
    does beyond a generic offer to talk / share a relevant example.

confidence: 0-1, your confidence in fit_tier + target_decision (not opener fluency).
reasoning: <= 20 words, naming the exact signal(s) the opener uses.
```

## Few-shot examples

All fictitious — none of these companies are in the dataset, so the few-shots can't leak
into the 25-account evaluation. Input lines use the exact format `harness.py` sends.

```
Input: company=Quarry Robotics domain=quarryrobotics.com industry=devtools employees=900 region=EMEA signals="uses Snowflake | raised Series B $30M in 2026" verified_fact="raised Series B $30M in 2026"
Output: {"fit_tier":"A","target_decision":"target","next_action":"schedule_intro_call","opener":"Congrats to Quarry Robotics on the $30M Series B in 2026 — would a short call on what comes next be useful?","confidence":0.9,"reasoning":"employees>=100 -> A; funding signal; opener uses verified_fact 'raised Series B $30M in 2026'."}

Input: company=Fernhill Freight domain=fernhillfreight.io industry=logistics employees=40 region=APAC signals="hiring 2 platform engineers | uses GCP" verified_fact=""
Output: {"fit_tier":"B","target_decision":"target","next_action":"request_warm_intro","opener":"Saw Fernhill Freight is hiring 2 platform engineers — happy to share how similar teams on GCP approach that build-out.","confidence":0.88,"reasoning":"employees<100 -> B; hiring signal; opener uses 'hiring 2 platform engineers' and 'uses GCP'."}

Input: company=Tidewater Games domain=tidewatergames.com industry=gaming employees=220 region=North America signals="uses Databricks" verified_fact=""
Output: {"fit_tier":"A","target_decision":"target","next_action":"send_case_study","opener":"Noticed Tidewater Games runs on Databricks — we have a case study from a team on the same stack if useful.","confidence":0.85,"reasoning":"employees>=100 -> A; stack-only signal; opener uses 'uses Databricks'."}

Input: company=Orchard Pay domain=orchardpay.com industry=fintech employees=300 region=LATAM signals="" verified_fact=""
Output: {"fit_tier":null,"target_decision":"insufficient_data","next_action":"add_to_nurture_sequence","opener":null,"confidence":0.95,"reasoning":"No signals; nothing to ground an opener in."}
```

## Model config

| Setting | Value | Why |
|---|---|---|
| Provider / model | `openai` / `gpt-4o-mini` | Same as Track A — cheap, fast, enough for a rule-anchored classification + one-sentence draft. |
| Temperature | `0` | Deterministic tiering/grounding is the point; no creative variance wanted. |
| `max_iterations` | `1` | Single-shot verdict per account, no tool loop. |

## Studio features — on/off, and why

| Feature | On? | Why |
|---|---|---|
| Conversational memory (`store_messages`) | Off | Each account is scored independently — memory adds tokens/latency for no benefit, same as Track A. |
| Knowledge Base | **Off** | The KB here is one `verified_fact` string per account, keyed by `account_id`. That's an exact-key lookup, not a retrieval problem: `harness.py` joins it into the prompt for free. A vector KB would add latency and a chance of retrieving *another* account's fact — a grounding failure. Revisit if a real per-account document corpus (call notes, website scrapes) is added. |
| Platform JSON-schema / `response_format` | Off, deliberately | Track A hit a real incident where Studio's structured-output enforcement 400'd every call after a silent model fallback. Same mitigation: JSON via instructions + code-side validation in `harness.py`. |
| Reflection / multi-step orchestration | Off | Single-shot, rule-anchored task; an extra pass adds latency/cost without a clear accuracy win. |
| Multi-provider fallback | On (Studio config) | Same outage protection as Track A. |

## What `harness.py` validates on the way out (not trusted from the model raw)

- All six keys must be present; a response missing any (e.g. the wrong agent answering)
  is a validation failure, not a silently-empty verdict.
- `fit_tier` recomputed from the employees rule; mismatches are corrected and counted.
- `next_action` must be one of the four closed-set strings, and is recomputed from the
  STEP 3 priority rule; mismatches are corrected and counted. (gpt-4o-mini ignored the
  priority order for ~1 in 3 accounts, at both temperature 0 and 0.7.)
- `opener` grounding: every number and every capitalized term (after the first word,
  excluding the company name, industry, and region) must appear in that account's
  `signals` + `verified_fact`. Failures count toward the reported fabricated-claim rate.
- Zero-signal accounts never reach the model in the optimized build (code-side gate).

## Known design choices without ground truth

The labeled set has only tier A/B and `should_target = yes`, and no labels for
`next_action` or opener quality. So the `next_action` priority order and the `skip` rule
are documented design choices, not measured ones — tier and target accuracy are the only
numbers the labels can actually score.
