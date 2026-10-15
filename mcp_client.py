import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).parent


class Tracer:

    def __init__(self, path: str | Path = HERE / "trace.jsonl", echo: bool = True,
                 reset: bool = False):
        self.path, self.echo = Path(path), echo
        if reset:
            self.path.write_text("", encoding="utf-8")

    def log(self, **rec):
        rec["ts"] = time.strftime("%H:%M:%S")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self.echo:
            status = "OK " if rec.get("ok") else "ERR"
            res = json.dumps(rec.get("result"), ensure_ascii=False)
            print(f"  [{rec['trace_id']}] {rec['agent']:<9} {status} "
                  f"{rec['tool']}({json.dumps(rec['args'], ensure_ascii=False)}) "
                  f"{rec['latency_ms']}ms -> {res[:160]}")


class McpHandle:
    def __init__(self, session: ClientSession):
        self.session = session

    async def tool_defs(self) -> list[dict]:
        tools = (await self.session.list_tools()).tools
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description or "",
                              "parameters": t.inputSchema}} for t in tools]

    async def call(self, name: str, args: dict) -> dict:
        res = await self.session.call_tool(name, args)
        text = "".join(c.text for c in res.content if getattr(c, "type", "") == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"error": "tool_failure", "message": text[:300],
                    "hint": "Перевір назви й типи аргументів за схемою", "retryable": True}


@asynccontextmanager
async def mcp_session(data_dir: str | Path):
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(HERE / "server.py")],
        env={**os.environ, "EXPENSES_DATA_DIR": str(data_dir)},
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            yield McpHandle(session)
