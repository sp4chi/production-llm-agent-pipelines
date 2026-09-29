import asyncio

import httpx
import pytest

from common import lyzr_client as lyzr


@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    '```\n{"a": 1}\n```',
    'Sure, here you go:\n{"a": 1}\nHope that helps.',
])
def test_parse_json_object_extracts_object(raw):
    assert lyzr.parse_json_object(raw) == {"a": 1}


@pytest.mark.parametrize("raw, message", [
    ("not json at all", "could not parse"),
    ("{broken", "could not parse"),
    ("[1, 2]", "not an object"),
    ('"just a string"', "not an object"),
])
def test_parse_json_object_rejects_non_objects(raw, message):
    with pytest.raises(ValueError, match=message):
        lyzr.parse_json_object(raw)


def test_percentile():
    assert lyzr.percentile([], 95) == 0.0
    assert lyzr.percentile([3.0], 95) == 3.0
    assert lyzr.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert lyzr.percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_estimate_cost_usd():
    expected = (1000 * lyzr.CREDITS_PER_INPUT_TOKEN
                + 100 * lyzr.CREDITS_PER_OUTPUT_TOKEN) * lyzr.USD_PER_CREDIT
    assert lyzr.estimate_cost_usd(1000, 100) == pytest.approx(expected)


def _run_chat(handler, retries=3):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await lyzr.chat(client, "agent", "sess", "hello", 100, retries)
    return asyncio.run(go())


def test_chat_estimates_tokens_when_no_usage_block():
    resp = _run_chat(lambda req: httpx.Response(200, json={"response": '{"x": 1}'}))
    assert resp.text == '{"x": 1}'
    assert resp.tokens_estimated
    assert resp.prompt_tokens == lyzr.estimate_tokens("hello") + 100
    assert resp.cost_usd > 0


def test_chat_uses_usage_block_when_present():
    body = {"response": "ok", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    resp = _run_chat(lambda req: httpx.Response(200, json=body))
    assert (resp.prompt_tokens, resp.completion_tokens, resp.tokens_estimated) == (10, 5, False)


def test_chat_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(lyzr.asyncio, "sleep", _no_sleep)
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(500) if len(calls) < 3 else httpx.Response(200, json={"response": "ok"})

    assert _run_chat(handler).text == "ok"
    assert len(calls) == 3


def test_chat_raises_after_retries(monkeypatch):
    monkeypatch.setattr(lyzr.asyncio, "sleep", _no_sleep)
    with pytest.raises(lyzr.LyzrCallError, match="failed after 2 attempts"):
        _run_chat(lambda req: httpx.Response(503), retries=2)


async def _no_sleep(_):
    return None
