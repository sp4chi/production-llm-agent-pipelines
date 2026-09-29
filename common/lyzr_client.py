"""Shared Lyzr Studio client for both tracks: config, the chat API call with
retries, JSON extraction from model output, token/cost estimation, percentiles."""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

LYZR_BASE_URL = os.environ.get("LYZR_BASE_URL", "https://agent-prod.studio.lyzr.ai")
LYZR_API_KEY = os.environ.get("LYZR_API_KEY", "")
LYZR_USER_ID = os.environ.get("LYZR_USER_ID", "")

# Pricing calibrated from 16 real traces on Lyzr's traces-v2 dashboard
# (gpt-4o-mini, Track A agent). Each trace carries llm_input_tokens,
# llm_output_tokens and action_cost in credits; a least-squares fit of
# credits = a*input + b*output (no intercept) matches every point within
# 0.08%. Re-calibrate if the model changes.
CREDITS_PER_INPUT_TOKEN = 0.00015006
CREDITS_PER_OUTPUT_TOKEN = 0.00059287
# Lyzr's published self-serve rate (~$10 per 1,000 credits). Not confirmed
# against this account's billing tier.
USD_PER_CREDIT = 0.01

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except ImportError:
    def estimate_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * CREDITS_PER_INPUT_TOKEN
            + completion_tokens * CREDITS_PER_OUTPUT_TOKEN) * USD_PER_CREDIT


@dataclass
class ChatResponse:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    tokens_estimated: bool
    cost_usd: float


class LyzrCallError(Exception):
    pass


async def chat(client: httpx.AsyncClient, agent_id: str, session_id: str, message: str,
               prompt_overhead_tokens: int, retries: int = 3) -> ChatResponse:
    """POST one message to /v3/inference/chat/, retrying transient failures with
    backoff. session_id is deterministic per item so a retry doesn't create
    duplicate state on Lyzr's side. Raises LyzrCallError once retries run out.

    The endpoint returns {"response": "...", "module_outputs": {}} with no usage
    block, so tokens are estimated: message tokens + the agent's fixed prompt
    overhead (calibrated per agent by the caller)."""
    payload = {"user_id": LYZR_USER_ID, "agent_id": agent_id,
               "session_id": session_id, "message": message}
    headers = {"x-api-key": LYZR_API_KEY, "Content-Type": "application/json"}

    last_err = None
    for attempt in range(retries):
        start = time.perf_counter()
        try:
            resp = await client.post(f"{LYZR_BASE_URL}/v3/inference/chat/",
                                     headers=headers, json=payload)
            latency = time.perf_counter() - start
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001 - retry on anything transient
            last_err = str(e)
            if attempt < retries - 1:
                await asyncio.sleep(min(2 ** attempt, 8))
            continue

        text = str(data.get("response") or data.get("agent_response") or "")
        usage = data.get("usage") or {}
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens = estimate_tokens(message) + prompt_overhead_tokens
            completion_tokens = estimate_tokens(text)
        return ChatResponse(
            text=text, latency_s=latency,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            tokens_estimated=not usage,
            cost_usd=estimate_cost_usd(prompt_tokens, completion_tokens),
        )

    raise LyzrCallError(f"failed after {retries} attempts: {last_err}")


def parse_json_object(raw: str) -> dict:
    """Extract one JSON object from model output. Without platform schema
    enforcement the model may wrap JSON in markdown fences or add stray prose,
    so try a fenced block first, then the outermost {...} span. Raises
    ValueError if nothing parses to a dict."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        braces = re.search(r"\{.*\}", text, re.DOTALL)
        if braces:
            text = braces.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError("could not parse JSON from model output") from e
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON was not an object")
    return obj


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)
