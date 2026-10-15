import argparse
import asyncio
import csv
import json
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

from agents import run_single_agent, run_two_agents
from llm import ConfigError, DailyBudgetExceeded, make_client
from mcp_client import Tracer, mcp_session

DATA = Path(__file__).parent / "data"
EMPLOYEE = "Іван Шевченко"
TODAY = date.today()


def days_ago(n: int) -> date:
    return TODAY - timedelta(days=n)


def q(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def ago(n: int) -> tuple[str, str]:
    """Залишено для сумісності зі старими тестами: (дата для запиту, ISO)."""
    d = days_ago(n)
    return q(d), d.isoformat()


D1, D3, D4, D5, D6 = days_ago(3), days_ago(4), days_ago(5), days_ago(2), days_ago(3)
TYPO_YEAR = D6.replace(year=D6.year + 1)


def rows_eq(*expected):
    want = set(expected)
    return lambda rows: rows == want

TESTS = [
    {"id": 1, "why": "щасливий шлях (meals 420, ліміт 500, чек є)",
     "query": f"Подай обід 420 грн, {q(D1)}, чек є",
     "state": rows_eq(("meals", 420.0, D1.isoformat())),
     "expected": f"рівно один рядок meals 420 {D1.isoformat()}",
     "submit_check": lambda subs: len(subs) == 1 and subs[0].get("has_receipt") is True},
    {"id": 2, "why": "помилка сервера і вихід із неї: описка в році (future_date)",
     "query": f"Подай обід 420 грн, {q(TYPO_YEAR)}, чек є",
     "state": rows_eq(("meals", 420.0, D6.isoformat())),
     "expected": f"один рядок meals 420 {D6.isoformat()} (змінено лише рік, день і місяць збережені)",
     "note_server": "future_date",
     "note_answer": (("рік", "дат", "2026", "виправ"),)},
    {"id": 3, "why": "пастка: чека немає, але для office він не потрібен",
     "query": f"Канцтовари 300 грн, {q(D3)}, чека немає",
     "state": rows_eq(("office", 300.0, D3.isoformat())),
     "expected": f"один рядок office 300 {D3.isoformat()}, has_receipt=false",
     "submit_check": lambda subs: len(subs) == 1 and subs[0].get("has_receipt") is False},
    {"id": 4, "why": "пастка: чека немає, а для training він обов'язковий",
     "query": f"Курс 2500 грн, {q(D4)}, чека немає",
     "state": lambda rows: not rows,
     "expected": "нічого не записано; у відповіді названо причину (чек)",
     "answer_all": (("чек",),)},
    {"id": 5, "why": "пакет із трьох: одна проходить, дві порушують правила різними способами",
     "query": (f"Подай три витрати за {q(D5)}: обід 300 грн з чеком, "
               f"таксі 1800 грн з чеком, курс 2500 грн без чека"),
     "state": rows_eq(("meals", 300.0, D5.isoformat())),
     "expected": f"записано лише meals 300; відхилено таксі (ліміт 1000) і курс (потрібен чек)",
     "answer_all": (("таксі", "transport", "транспорт"), ("курс", "training", "навчан"))},
]


def new_rows(workdir: Path) -> set:
    with (workdir / "expenses.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with (DATA / "expenses.seed.csv").open(encoding="utf-8", newline="") as f:
        seed_ids = {r["id"] for r in csv.DictReader(f)}
    return {(r["category"], float(r["amount_uah"]), r["date"])
            for r in rows if r["id"] not in seed_ids}


def read_trace(tracer, trace_id: str) -> list[dict]:
    path = Path(getattr(tracer, "path", "trace.jsonl"))
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("trace_id") == trace_id:
            out.append(r)
    return out


def has_all_groups(text: str, groups) -> bool:
    text = text.lower()
    return all(any(w in text for w in g) for g in groups)


def evaluate(system: str, test: dict, got: set, answer: str, trace: list[dict]):
    notes = []
    submits = [r for r in trace if r.get("tool") == "submit_expense"]
    ok_submits = [r["args"] for r in submits if r.get("ok") and (r.get("result") or {}).get("ok", True)]
    server_errors = [(r.get("result") or {}).get("error") for r in submits if not r.get("ok")]

    if not test["state"](got):
        return "стан CSV не збігається з очікуваним", notes
    if system == "multi":
        bad = [r["agent"] for r in submits if r.get("agent") != "verifier"]
        if bad:
            return f"порушено розподіл ролей: submit_expense викликав {bad[0]}", notes
    if "submit_check" in test and not test["submit_check"](ok_submits):
        return "аргументи submit_expense не відповідають запиту (напр. вигаданий has_receipt)", notes
    if "answer_all" in test and not has_all_groups(answer or "", test["answer_all"]):
        return "у відповіді не названо причину відмови", notes

    if test.get("note_server") and test["note_server"] not in server_errors:
        notes.append(f"сервер не повертав {test['note_server']}: модель виправила запит ще до виклику "
                     f"(стан правильний, але шлях помилки не перевірено)")
    if test.get("note_answer") and not has_all_groups(answer or "", test["note_answer"]):
        notes.append("відповідь не повідомляє користувача про виправлення")
    return None, notes


async def run_one(system: str, test: dict, client, tracer) -> dict:
    s0 = dict(client.stats)
    trace_id = f"{system}-t{test['id']}"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copy(DATA / "policy.json", work / "policy.json")
        shutil.copy(DATA / "expenses.seed.csv", work / "expenses.csv")
        print(f"\n--- {trace_id}: {test['query']}")
        try:
            async with mcp_session(work) as mcp:
                run = run_two_agents if system == "multi" else run_single_agent
                res = await run(client, mcp, test["query"], EMPLOYEE, tracer, trace_id)
            got = new_rows(work)
            err = next((p.error for p in res.parts if p.error), None)
        except DailyBudgetExceeded:
            raise
        except Exception as e:
            res, got, err = None, set(), repr(e)

    notes = []
    reason = err
    if reason is None:
        reason, notes = evaluate(system, test, got, res.answer or "", read_trace(tracer, trace_id))
    passed = reason is None

    print(f"    очікувано: {test['expected']}\n    фактично:  {sorted(got)}  "
          f"=> {'PASS' if passed else 'FAIL'}" + (f"  ({reason})" if reason else ""))
    for n in notes:
        print(f"    ⚠ {n}")
    s1 = client.stats
    return {"system": system, "test": test["id"], "passed": passed,
            "got": sorted(got), "expected": test["expected"],
            "answer": res.answer if res else err, "error": err, "reason": reason, "notes": notes,
            "tool_calls": res.tool_calls if res else 0,
            "model_calls": res.model_calls if res else 0,
            "api_requests": s1["requests"] - s0["requests"],
            "cache_hits": s1["cache_hits"] - s0["cache_hits"],
            "seconds": round(res.seconds, 1) if res else 0}


def summarize(results: list[dict]) -> str:
    by = {(r["system"], r["test"]): r for r in results}
    systems = sorted({r["system"] for r in results})
    names = {"multi": "Два агенти", "single": "Один агент"}
    ids = sorted({r["test"] for r in results})
    lines = ["| # | Що перевіряє | " + " | ".join(names[s] for s in systems) +
             " | Де помилка (заповніть вручну) |",
             "|---|---|" + "---|" * len(systems) + "---|"]
    for t in (t for t in TESTS if t["id"] in ids):
        cells = []
        for s in systems:
            r = by.get((s, t["id"]))
            cells.append("..." if r is None else ("OK" if r["passed"] else "FAIL") +
                         (" ⚠" if r and r.get("notes") else ""))
        lines.append(f"| {t['id']} | {t['why']} | " + " | ".join(cells) + " | |")
    totals, cost = [], []
    for s in systems:
        rs = [r for r in results if r["system"] == s]
        totals.append(f"**{sum(r['passed'] for r in rs)} з {len(rs)}**")
        cost.append(f"{sum(r['tool_calls'] for r in rs)} викл. / "
                    f"{sum(r['model_calls'] for r in rs)} крок. / "
                    f"{sum(r['seconds'] for r in rs):.0f} c")
    lines.append("| | **Разом** | " + " | ".join(totals) + " | |")
    lines.append("| | Вартість | " + " | ".join(cost) + " | |")
    return "\n".join(lines)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["multi", "single"])
    p.add_argument("--tests", nargs="+", type=int, help="номери тестів, напр. 1 2")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()
    systems = [args.only] if args.only else ["multi", "single"]
    tests = [t for t in TESTS if not args.tests or t["id"] in args.tests]
    try:
        client = make_client(use_cache=not args.no_cache)
    except ConfigError as e:
        sys.exit(f"Налаштування: {e}")
    print(f"Провайдер: {client.cfg.provider}, модель: {client.cfg.model}; "
          f"сьогодні вже використано: {client.usage_today()}")
    tracer = Tracer(reset=True)
    results, stopped = [], None
    try:
        for s in systems:
            for t in tests:
                results.append(await run_one(s, t, client, tracer))
    except DailyBudgetExceeded as e:
        stopped = str(e)
    if results:
        suffix = "_".join(sorted({r["system"] for r in results}))
        Path(f"results_{suffix}.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        table = summarize(results)
        Path(f"results_{suffix}.md").write_text(table + "\n", encoding="utf-8")
        if suffix == "multi_single":
            Path("results.md").write_text(table + "\n", encoding="utf-8")
            shutil.copy(f"results_{suffix}.json", "results.json")
        print("\n" + table)
    st, u = client.stats, client.usage_today()
    print(f"\nЗапитів до API за цей запуск: {st['requests']} (з кешу: {st['cache_hits']}, "
          f"повторів: {st['retries']}, очікування: {st['waited_s']:.0f} c). "
          f"Сьогодні всього: {u['requests']} запитів, ~{u['tokens']} токенів.")
    if stopped:
        print(f"\nЗУПИНЕНО: {stopped}\nЗапустіть команду ще раз пізніше: уже зроблені кроки беруться з кешу.")
    else:
        print(f"\nЗбережено: results_{suffix}.json / .md (файли окремі для multi і single)")


if __name__ == "__main__":
    asyncio.run(main())