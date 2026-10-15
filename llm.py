import asyncio
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

SAFETY_RATE = 0.9
SAFETY_DAY = 0.95
MAX_ATTEMPTS = 4
EST_OUTPUT_TOKENS = 400
PRESETS = {
    "groq": dict(base_url="https://api.groq.com/openai/v1", key_env="GROQ_API_KEY",
                 model="openai/gpt-oss-120b", rpm=30, rpd=1000, tpm=8000, tpd=200_000),
    "openrouter": dict(base_url="https://openrouter.ai/api/v1", key_env="OPENROUTER_API_KEY",
                       model="", rpm=20, rpd=50, tpm=None, tpd=None),
    "ollama": dict(base_url="http://localhost:11434/v1", key_env="",
                   model="", rpm=None, rpd=None, tpm=None, tpd=None),
    "custom": dict(base_url="", key_env="API_KEY",
                   model="", rpm=None, rpd=None, tpm=None, tpd=None),
}


class ConfigError(Exception):
    pass


class LLMError(Exception):
    """Збій одного виклику моделі (агент зупиняється, прогін триває)."""


class DailyBudgetExceeded(Exception):
    """Добовий безкоштовний ліміт вичерпано: прогін треба зупинити."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: int = 0
    cached: bool = False


def assistant_message(reply: LLMReply) -> dict:
    msg = {"role": "assistant", "content": reply.content}
    if reply.tool_calls:
        msg["tool_calls"] = [{"id": t.id, "type": "function",
                              "function": {"name": t.name, "arguments": t.arguments}}
                             for t in reply.tool_calls]
    return msg


@dataclass
class Config:
    provider: str
    base_url: str
    api_key: str
    model: str
    rpm: int | None
    rpd: int | None
    tpm: int | None
    tpd: int | None

    @classmethod
    def from_env(cls, env=None) -> "Config":
        env = os.environ if env is None else env
        provider = env.get("PROVIDER", "groq").lower()
        if provider not in PRESETS:
            raise ConfigError("PROVIDER має бути одним із: " + ", ".join(PRESETS))
        p = PRESETS[provider]
        base_url = env.get("BASE_URL") or p["base_url"]
        if not base_url:
            raise ConfigError("Для PROVIDER=custom задайте BASE_URL.")
        key_env = p["key_env"]
        api_key = env.get(key_env, "") if key_env else "ollama"
        if key_env and not api_key:
            raise ConfigError(f"Не задано {key_env} (див. .env.example).")
        model = env.get("MODEL") or p["model"]
        if not model:
            raise ConfigError("Не задано MODEL (див. .env.example).")
        if provider == "openrouter" and not model.endswith(":free"):
            raise ConfigError("Для безкоштовного режиму OpenRouter модель має закінчуватись на ':free'.")

        def lim(name):
            v = env.get(f"LIMIT_{name.upper()}")
            if v is None or v == "":
                return p[name]
            return int(v) or None  
        return cls(provider, base_url, api_key, model,
                   lim("rpm"), lim("rpd"), lim("tpm"), lim("tpd"))


class LLMClient:
    def __init__(self, cfg: Config, use_cache: bool = True,
                 cache_dir: Path = HERE / ".llm_cache", ledger_path: Path = HERE / ".usage.json"):
        self.cfg, self.use_cache = cfg, use_cache
        self.cache_dir, self.ledger_path = Path(cache_dir), Path(ledger_path)
        self._sdk = openai.AsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                                       max_retries=0, timeout=90)
        self._window: deque = deque()
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0, "tokens": 0, "waited_s": 0.0}

    async def chat(self, system: str, messages: list[dict], tools: list[dict],
                   max_tokens: int = 1200) -> LLMReply:
        payload = [{"role": "system", "content": system}] + messages
        key = hashlib.sha256(json.dumps(
            {"m": self.cfg.model, "msgs": payload, "tools": tools, "mt": max_tokens},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cached = self._cache_get(key)
        if cached:
            self.stats["cache_hits"] += 1
            return cached

        est = self._estimate(payload, tools)
        last_err = ""
        for attempt in range(MAX_ATTEMPTS):
            self._check_daily(est)
            await self._throttle(est)
            self._count_request(est)
            try:
                reply = await self._request(payload, tools, max_tokens)
                break
            except openai.RateLimitError as e:
                text = str(e).lower()
                if any(s in text for s in ("per day", "(tpd)", "(rpd)")):
                    raise DailyBudgetExceeded(f"Провайдер: добовий ліміт вичерпано. {str(e)[:160]}")
                last_err = f"429: {str(e)[:120]}"
                self.stats["retries"] += 1
                await self._sleep(self._retry_after(e) or 20)
            except openai.BadRequestError as e:
                last_err = f"400: {str(e)[:160]}"
                if "tool_use_failed" in str(e) or "Failed to call a function" in str(e):
                    self.stats["retries"] += 1
                    continue
                raise LLMError(last_err)
            except openai.AuthenticationError:
                raise LLMError("Ключ API не підходить (401). Перевірте .env.")
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                last_err = f"мережа/5xx: {str(e)[:120]}"
                self.stats["retries"] += 1
                await self._sleep(3 * (attempt + 1))
        else:
            raise LLMError(f"Не вдалося після {MAX_ATTEMPTS} спроб. Остання помилка: {last_err}")

        self._settle(est, reply.tokens)
        self._cache_put(key, reply)
        return reply

    async def _request(self, messages, tools, max_tokens) -> LLMReply:
        kw = {"tools": tools} if tools else {}
        resp = await self._sdk.chat.completions.create(
            model=self.cfg.model, messages=messages, temperature=0, max_tokens=max_tokens, **kw)
        m = resp.choices[0].message
        calls = [ToolCall(t.id, t.function.name, t.function.arguments or "{}")
                 for t in (m.tool_calls or [])]
        tokens = getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
        return LLMReply(m.content, calls, tokens)

    async def _sleep(self, s: float):
        self.stats["waited_s"] += s
        await asyncio.sleep(s)

    @staticmethod
    def _retry_after(e) -> float | None:
        try:
            h = e.response.headers.get("retry-after")
            return min(float(h), 90) if h else None
        except Exception:
            return None

    @staticmethod
    def _estimate(payload, tools) -> int:
        chars = (len(json.dumps(payload, ensure_ascii=False))
                 + len(json.dumps(tools, ensure_ascii=False)))
        return int(chars / 2.5) + EST_OUTPUT_TOKENS

    async def _throttle(self, est: int):
        c = self.cfg
        while True:
            now = time.time()
            while self._window and now - self._window[0][0] >= 60:
                self._window.popleft()
            wait = 0.0
            if self._window:
                oldest_left = 60 - (now - self._window[0][0])
                if c.rpm and len(self._window) >= max(1, int(c.rpm * SAFETY_RATE)):
                    wait = max(wait, oldest_left)
                if c.tpm and sum(t for _, t in self._window) + est > c.tpm * SAFETY_RATE:
                    wait = max(wait, oldest_left)
            if wait <= 0:
                return
            await self._sleep(wait + 0.2)

    def _count_request(self, est: int):
        self._window.append([time.time(), est])
        self.stats["requests"] += 1
        self._ledger_add(requests=1, tokens=est)

    def _settle(self, est: int, actual: int):
        if actual:
            if self._window:
                self._window[-1][1] = actual
            self._ledger_add(tokens=actual - est)
            self.stats["tokens"] += actual

    def _ledger(self):
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        key = f"{self.cfg.provider}:{self.cfg.model}"
        today = datetime.now(timezone.utc).date().isoformat()
        e = data.get(key)
        if not e or e.get("date") != today:
            e = {"date": today, "requests": 0, "tokens": 0}
        return data, key, e

    def _ledger_add(self, requests=0, tokens=0):
        data, key, e = self._ledger()
        e["requests"] += requests
        e["tokens"] = max(0, e["tokens"] + tokens)
        data[key] = e
        self.ledger_path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def usage_today(self) -> dict:
        return self._ledger()[2]

    def _check_daily(self, est: int):
        c, e = self.cfg, self._ledger()[2]
        if c.rpd and e["requests"] + 1 > c.rpd * SAFETY_DAY:
            raise DailyBudgetExceeded(
                f"Добовий ліміт запитів майже вичерпано ({e['requests']} з ~{c.rpd}). "
                "Продовжіть завтра (кеш збережено) або змініть провайдера.")
        if c.tpd and e["tokens"] + est > c.tpd * SAFETY_DAY:
            raise DailyBudgetExceeded(
                f"Добовий ліміт токенів майже вичерпано ({e['tokens']} з ~{c.tpd}). "
                "Продовжіть завтра (кеш збережено) або змініть провайдера.")

    def _cache_get(self, key: str) -> LLMReply | None:
        if not self.use_cache:
            return None
        f = self.cache_dir / f"{key}.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text(encoding="utf-8"))
        return LLMReply(d["content"], [ToolCall(**t) for t in d["tool_calls"]], d["tokens"], True)

    def _cache_put(self, key: str, r: LLMReply):
        if not self.use_cache:
            return
        self.cache_dir.mkdir(exist_ok=True)
        d = {"content": r.content, "tokens": r.tokens,
             "tool_calls": [t.__dict__ for t in r.tool_calls]}
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(d, ensure_ascii=False), encoding="utf-8")


def make_client(use_cache: bool = True) -> LLMClient:
    return LLMClient(Config.from_env(), use_cache=use_cache)
