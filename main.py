import argparse
import asyncio
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

from agents import run_single_agent, run_two_agents
from llm import ConfigError, DailyBudgetExceeded, make_client
from mcp_client import Tracer, mcp_session

DATA = Path(__file__).parent / "data"
EMPLOYEE = "Іван Шевченко"


def demos() -> dict[str, str]:
    d = (date.today() - timedelta(days=1)).strftime("%d.%m.%Y")
    return {
        "unknown_category": f"Подай витрату: їжа, 300 грн, {d}, чек є",
        "duplicate": "Подай обід 250 грн за 15.01.2026, чек є",
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?")
    p.add_argument("--single", action="store_true")
    p.add_argument("--reset", action="store_true", help="повернути expenses.csv до seed")
    p.add_argument("--demo", choices=list(demos()))
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    if args.reset:
        shutil.copy(DATA / "expenses.seed.csv", DATA / "expenses.csv")
        print("expenses.csv скинуто до початкового стану")
        if not (args.query or args.demo):
            return
    query = demos()[args.demo] if args.demo else args.query
    if not query:
        p.error("потрібен запит або --demo")

    try:
        client = make_client(use_cache=not args.no_cache)
    except ConfigError as e:
        sys.exit(f"Налаштування: {e}")
    print(f"Запит: {query}\nСистема: {'один агент' if args.single else 'два агенти'}"
          f" | провайдер: {client.cfg.provider}, модель: {client.cfg.model}\n")
    tracer = Tracer()
    try:
        async with mcp_session(DATA) as mcp:
            run = run_single_agent if args.single else run_two_agents
            res = await run(client, mcp, query, EMPLOYEE, tracer, "demo")
    except DailyBudgetExceeded as e:
        sys.exit(f"\nСтоп: {e}")
    print(f"\nВідповідь:\n{res.answer}\n")
    s, u = client.stats, client.usage_today()
    print("\nexpenses.csv (останні 4 рядки):")
    print("".join((DATA / "expenses.csv").read_text(encoding="utf-8").splitlines(True)[-4:]))


if __name__ == "__main__":
    asyncio.run(main())
