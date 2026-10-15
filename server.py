import csv
import json
import os
from datetime import date, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DATA_DIR = Path(os.environ.get("EXPENSES_DATA_DIR", Path(__file__).parent / "data"))
CSV_PATH = DATA_DIR / "expenses.csv"
POLICY_PATH = DATA_DIR / "policy.json"
FIELDS = ["id", "employee", "category", "amount_uah", "date",
          "description", "has_receipt", "status"]

mcp = FastMCP("expenses")


def _err(error: str, message: str, hint: str, retryable: bool) -> dict:
    return {"error": error, "message": message, "hint": hint, "retryable": retryable}


def _policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _rows() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _unknown_category(category: str, policy: dict) -> dict:
    return _err("unknown_category",
                f"Категорії '{category}' не існує.",
                "Допустимі: " + ", ".join(policy),
                True)


@mcp.tool()

def get_policy(category: str) -> dict:
    policy = _policy()
    key = category.strip().lower()
    if key not in policy:
        return _unknown_category(category, policy)
    return {"category": key, **policy[key]}


@mcp.tool()
def list_expenses(employee: str, month: str = "") -> dict:
    if month:
        try:
            datetime.strptime(month, "%Y-%m")
        except ValueError:
            return _err("invalid_month", f"Невірний формат місяця '{month}'.",
                        "Використай YYYY-MM, наприклад 2026-10", True)
    rows = [r for r in _rows()
            if r["employee"] == employee and (not month or r["date"].startswith(month))]
    rows.sort(key=lambda r: r["date"], reverse=True)
    return {"employee": employee, "count": len(rows), "expenses": rows[:50]}


@mcp.tool()
def submit_expense(employee: str, category: str, amount_uah: float, date: str,
                   description: str, has_receipt: bool) -> dict:
    policy = _policy()
    cat = category.strip().lower()

    if not employee.strip():
        return _err("missing_employee", "Не вказано співробітника.",
                    "Передай повне імʼя співробітника", True)
    if cat not in policy:
        return _unknown_category(category, policy)
    if amount_uah <= 0:
        return _err("invalid_amount", f"Сума {amount_uah} має бути більшою за 0.",
                    "Передай додатне число в гривнях", True)
    try:
        d = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        return _err("invalid_date", f"Невірний формат дати '{date}'.",
                    "Використай YYYY-MM-DD", True)
    if d > _today():
        return _err("future_date", f"Дата {date} у майбутньому.",
                    f"Сьогодні {_today().isoformat()}; витрати подаються постфактум", True)
    rule = policy[cat]
    if rule["receipt_required"] and not has_receipt:
        return _err("missing_receipt", f"Для '{cat}' потрібен чек, а його немає.",
                    "Попроси користувача додати чек і подати знову", False)
    if amount_uah > rule["limit_uah"]:
        return _err("exceeds_limit",
                    f"{amount_uah} грн перевищує ліміт {rule['limit_uah']} грн для '{cat}'.",
                    "Потрібне погодження менеджера, автоматично подати не можна", False)
    for r in _rows():
        if (r["employee"], r["category"], float(r["amount_uah"]), r["date"]) == \
                (employee, cat, float(amount_uah), date):
            return _err("duplicate_expense",
                        f"Така витрата вже подана ({r['id']}).",
                        "Не подавай повторно", False)

    rows = _rows()
    new_id = f"E-{len(rows) + 1:04d}"
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({"id": new_id, "employee": employee, "category": cat,
                    "amount_uah": float(amount_uah), "date": date,
                    "description": description,
                    "has_receipt": str(bool(has_receipt)).lower(),
                    "status": "pending"})
    return {"ok": True, "expense_id": new_id, "status": "pending"}


def _today() -> date:
    return date.today()


if __name__ == "__main__":
    mcp.run()
