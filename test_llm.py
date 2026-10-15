import asyncio
import json
import types

import httpx
import openai
import pytest

from llm import (Config, ConfigError, DailyBudgetExceeded, LLMClient, LLMError, LLMReply,
                 ToolCall, assistant_message)


def cfg(**kw):
    base = dict(provider="groq", base_url="http://x", api_key="k", model="m",
                rpm=None, rpd=None, tpm=None, tpd=None)
    base.update(kw)
    return Config(**base)


def make(tmp_path, **kw):
    use_cache = kw.pop("use_cache", True)
    c = LLMClient(cfg(**kw), use_cache=use_cache, cache_dir=tmp_path / "cache",
                  ledger_path=tmp_path / "usage.json")

    async def no_sleep(s):
        c.stats["waited_s"] += s
    c._sleep = no_sleep
    return c


def rate_limit(msg="slow down", retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after else {}
    resp = httpx.Response(429, headers=headers, request=httpx.Request("POST", "http://x"))
    return openai.RateLimitError(msg, response=resp, body=None)


def test_cache_hit_does_not_call_network(tmp_path):
    c = make(tmp_path)
    calls = []

    async def fake(messages, tools, max_tokens):
        calls.append(1)
        return LLMReply("привіт", [], 50)
    c._request = fake
    r1 = asyncio.run(c.chat("sys", [{"role": "user", "content": "q"}], []))
    r2 = asyncio.run(c.chat("sys", [{"role": "user", "content": "q"}], []))
    assert len(calls) == 1 and r2.cached and r2.content == "привіт"
    assert c.stats["requests"] == 1 and c.stats["cache_hits"] == 1


def test_cache_roundtrips_tool_calls(tmp_path):
    c = make(tmp_path)

    async def fake(messages, tools, max_tokens):
        return LLMReply(None, [ToolCall("id1", "get_policy", '{"category": "meals"}')], 10)
    c._request = fake
    asyncio.run(c.chat("s", [{"role": "user", "content": "q"}], []))
    r = asyncio.run(c.chat("s", [{"role": "user", "content": "q"}], []))
    assert r.cached and r.tool_calls[0].name == "get_policy"
    assert assistant_message(r)["tool_calls"][0]["function"]["arguments"] == '{"category": "meals"}'


def test_no_cache_flag_always_hits_network(tmp_path):
    c = make(tmp_path, use_cache=False)
    n = []

    async def fake(messages, tools, max_tokens):
        n.append(1)
        return LLMReply("x", [], 5)
    c._request = fake
    for _ in range(2):
        asyncio.run(c.chat("s", [{"role": "user", "content": "q"}], []))
    assert len(n) == 2


def test_daily_request_budget_stops_before_provider_does(tmp_path):
    c = make(tmp_path, rpd=3)   # 95% від 3 = 2.85 -> дозволено 2 запити

    async def fake(messages, tools, max_tokens):
        return LLMReply("ok", [], 5)
    c._request = fake
    for i in range(2):
        asyncio.run(c.chat("s", [{"role": "user", "content": f"q{i}"}], []))
    with pytest.raises(DailyBudgetExceeded):
        asyncio.run(c.chat("s", [{"role": "user", "content": "q-third"}], []))
    assert json.loads((tmp_path / "usage.json").read_text())["groq:m"]["requests"] == 2


def test_daily_budget_survives_new_client_instance(tmp_path):
    a = make(tmp_path, rpd=2)

    async def fake(messages, tools, max_tokens):
        return LLMReply("ok", [], 5)
    a._request = fake
    asyncio.run(a.chat("s", [{"role": "user", "content": "q1"}], []))
    b = make(tmp_path, rpd=2) 
    b._request = fake
    with pytest.raises(DailyBudgetExceeded):
        asyncio.run(b.chat("s", [{"role": "user", "content": "q2"}], []))


def test_429_is_retried_and_every_attempt_counts(tmp_path):
    c = make(tmp_path)
    seq = [rate_limit(retry_after=7), LLMReply("ok", [], 5)]

    async def fake(messages, tools, max_tokens):
        x = seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    c._request = fake
    r = asyncio.run(c.chat("s", [{"role": "user", "content": "q"}], []))
    assert r.content == "ok"
    assert c.stats["requests"] == 2 and c.stats["retries"] == 1
    assert c.stats["waited_s"] == 7


def test_daily_limit_message_from_provider_stops_run(tmp_path):
    c = make(tmp_path)

    async def fake(messages, tools, max_tokens):
        raise rate_limit("Rate limit reached ... on tokens per day (TPD)")
    c._request = fake
    with pytest.raises(DailyBudgetExceeded):
        asyncio.run(c.chat("s", [{"role": "user", "content": "q"}], []))


def test_gives_up_after_max_attempts_with_llm_error(tmp_path):
    c = make(tmp_path)

    async def fake(messages, tools, max_tokens):
        raise rate_limit()
    c._request = fake
    with pytest.raises(LLMError):
        asyncio.run(c.chat("s", [{"role": "user", "content": "q"}], []))
    assert c.stats["requests"] == 4


def test_throttle_waits_when_rpm_window_is_full(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("llm.time", types.SimpleNamespace(time=lambda: clock[0]))
    c = make(tmp_path, rpm=10) 

    async def advancing_sleep(s):
        c.stats["waited_s"] += s
        clock[0] += s
    c._sleep = advancing_sleep

    async def fake(messages, tools, max_tokens):
        return LLMReply("ok", [], 5)
    c._request = fake
    for i in range(9):
        asyncio.run(c.chat("s", [{"role": "user", "content": f"q{i}"}], []))
    assert c.stats["waited_s"] == 0 
    asyncio.run(c.chat("s", [{"role": "user", "content": "q-tenth"}], []))
    assert 59 < c.stats["waited_s"] < 62 


def test_throttle_respects_tokens_per_minute(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("llm.time", types.SimpleNamespace(time=lambda: clock[0]))
    c = make(tmp_path, tpm=1000)

    async def advancing_sleep(s):
        c.stats["waited_s"] += s
        clock[0] += s
    c._sleep = advancing_sleep

    async def fake(messages, tools, max_tokens):
        return LLMReply("ok", [], 600) 
    c._request = fake
    asyncio.run(c.chat("s", [{"role": "user", "content": "a"}], []))
    asyncio.run(c.chat("s", [{"role": "user", "content": "b"}], []))
    assert c.stats["waited_s"] > 0


def test_openrouter_requires_free_model():
    env = {"PROVIDER": "openrouter", "OPENROUTER_API_KEY": "k", "MODEL": "some/paid-model"}
    with pytest.raises(ConfigError):
        Config.from_env(env)
    ok = Config.from_env({**env, "MODEL": "some/model:free"})
    assert ok.rpd == 50


def test_config_errors_are_clear():
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        Config.from_env({"PROVIDER": "groq"})
    with pytest.raises(ConfigError, match="MODEL"):
        Config.from_env({"PROVIDER": "ollama"})
    c = Config.from_env({"PROVIDER": "groq", "GROQ_API_KEY": "k", "LIMIT_RPD": "0"})
    assert c.rpd is None and c.model == "openai/gpt-oss-120b"
