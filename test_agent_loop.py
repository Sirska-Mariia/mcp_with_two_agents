import asyncio
import json
import shutil
from pathlib import Path

from agents import PREPARER_TOOLS, run_agent
from llm import LLMError, LLMReply, ToolCall
from mcp_client import Tracer, mcp_session
import run_tests as rt

DATA = Path(__file__).parent / "data"
EMP = "Іван Шевченко"


class FakeClient:
    """Підміняє LLMClient: віддає заздалегідь написаний сценарій відповідей."""

    def __init__(self, script):
        self.script = list(script)
        self.stats = {"requests": 0, "cache_hits": 0, "retries": 0, "tokens": 0, "waited_s": 0.0}
        self.seen_tools = []

    async def chat(self, system, messages, tools, max_tokens=1200):
        self.seen_tools = [t["function"]["name"] for t in tools]
        self.stats["requests"] += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(name, **args):
    return LLMReply(None, [ToolCall(f"id_{name}", name, json.dumps(args, ensure_ascii=False))])


def say(text):
    return LLMReply(text)


def test_preparer_cannot_write_and_trace_is_logged(tmp_path):
    shutil.copy(DATA / "policy.json", tmp_path / "policy.json")
    shutil.copy(DATA / "expenses.seed.csv", tmp_path / "expenses.csv")
    trace = tmp_path / "trace.jsonl"
    client = FakeClient([
        call("get_policy", category="meals"),
        call("submit_expense", employee="x", category="meals", amount_uah=1,
             date="2026-01-01", description="", has_receipt=True),  
        say("готово"),
    ])

    async def go():
        async with mcp_session(tmp_path) as mcp:
            defs = await mcp.tool_defs()
            return await run_agent(client, "preparer", "sys", "q", mcp, defs,
                                   PREPARER_TOOLS, Tracer(trace, echo=False), "t")
    out = asyncio.run(go())
    assert out.text == "готово" and out.tool_calls == 2
    assert "submit_expense" not in client.seen_tools         
    recs = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
    assert recs[1]["result"]["error"] == "permission_denied" 
    assert len((tmp_path / "expenses.csv").read_text(encoding="utf-8").splitlines()) == 4


def test_bad_json_arguments_become_tool_error_not_crash(tmp_path):
    shutil.copy(DATA / "policy.json", tmp_path / "policy.json")
    shutil.copy(DATA / "expenses.seed.csv", tmp_path / "expenses.csv")
    client = FakeClient([LLMReply(None, [ToolCall("i1", "get_policy", "{не json")]), say("ок")])

    async def go():
        async with mcp_session(tmp_path) as mcp:
            defs = await mcp.tool_defs()
            return await run_agent(client, "preparer", "s", "q", mcp, defs, PREPARER_TOOLS,
                                   Tracer(tmp_path / "t.jsonl", echo=False), "t")
    out = asyncio.run(go())
    assert out.text == "ок" and not out.halted


def test_llm_failure_halts_agent_gracefully(tmp_path):
    client = FakeClient([LLMError("429 після повторів")])

    async def go():
        async with mcp_session(DATA) as mcp:
            defs = await mcp.tool_defs()
            return await run_agent(client, "single", "s", "q", mcp, defs, PREPARER_TOOLS,
                                   Tracer(tmp_path / "t.jsonl", echo=False), "t")
    out = asyncio.run(go())
    assert out.halted and "429" in out.error


def test_end_to_end_two_agents_matches_expected_state(tmp_path):
    """Наскрізно: run_one -> два агенти -> MCP -> CSV -> порівняння з очікуваним."""
    test = rt.TESTS[0]
    _, iso = rt.ago(3)
    draft = json.dumps([{"category": "meals", "amount_uah": 420, "date": iso, "description": "Обід",
                         "has_receipt": True, "verdict": "ok", "reason": ""}], ensure_ascii=False)
    client = FakeClient([
        call("get_policy", category="meals"), call("list_expenses", employee=EMP), say(draft),
        call("get_policy", category="meals"),
        call("submit_expense", employee=EMP, category="meals", amount_uah=420, date=iso,
             description="Обід", has_receipt=True),
        say("Записано."),
    ])
    r = asyncio.run(rt.run_one("multi", test, client, Tracer(tmp_path / "t.jsonl", echo=False)))
    assert r["passed"]
    assert r["model_calls"] == 6  
    assert r["tool_calls"] == 4  
    assert r["api_requests"] == 6


def test_summary_marks_missing_results():
    rows = [{"system": "multi", "test": 1, "passed": True, "tool_calls": 3, "model_calls": 4, "seconds": 1.0}]
    table = rt.summarize(rows + [{"system": "single", "test": 2, "passed": False,
                                  "tool_calls": 1, "model_calls": 2, "seconds": 1.0}])
    assert "..." in table and "-" in table
