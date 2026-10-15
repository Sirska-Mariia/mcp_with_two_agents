import asyncio
import csv
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest

from mcp_client import mcp_session

DATA = Path(__file__).parent / "data"
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()
BASE = dict(employee="Іван Шевченко", category="meals", amount_uah=420,
            date=YESTERDAY, description="Обід", has_receipt=True)


@pytest.fixture
def work(tmp_path):
    shutil.copy(DATA / "policy.json", tmp_path / "policy.json")
    shutil.copy(DATA / "expenses.seed.csv", tmp_path / "expenses.csv")
    return tmp_path


def call(work, name, args):
    async def go():
        async with mcp_session(work) as mcp:
            return await mcp.call(name, args)
    return asyncio.run(go())


def rows(work):
    with (work / "expenses.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_submit_ok_writes_row(work):
    r = call(work, "submit_expense", BASE)
    assert r["ok"] and r["status"] == "pending"
    assert len(rows(work)) == 4


@pytest.mark.parametrize("patch,code,retryable", [
    ({"category": "їжа"}, "unknown_category", True),
    ({"amount_uah": -5}, "invalid_amount", True),
    ({"date": "вчора"}, "invalid_date", True),
    ({"date": (date.today() + timedelta(days=3)).isoformat()}, "future_date", True),
    ({"has_receipt": False}, "missing_receipt", False),
    ({"amount_uah": 900}, "exceeds_limit", False),
    ({"employee": " "}, "missing_employee", True),
])
def test_bad_input_returns_clear_error_and_writes_nothing(work, patch, code, retryable):
    r = call(work, "submit_expense", {**BASE, **patch})
    assert r["error"] == code and r["retryable"] is retryable
    assert r["message"] and r["hint"]
    assert len(rows(work)) == 3


def test_unknown_category_hint_lists_valid(work):
    r = call(work, "submit_expense", {**BASE, "category": "їжа"})
    assert "meals" in r["hint"]


def test_duplicate_rejected(work):
    r = call(work, "submit_expense", {**BASE, "amount_uah": 250, "date": "2026-01-15"})
    assert r["error"] == "duplicate_expense"


def test_office_needs_no_receipt(work):
    r = call(work, "submit_expense", {**BASE, "category": "office", "amount_uah": 300,
                                      "has_receipt": False})
    assert r["ok"]


def test_read_tools(work):
    assert call(work, "get_policy", {"category": "meals"})["limit_uah"] == 500
    assert call(work, "get_policy", {"category": "x"})["error"] == "unknown_category"
    r = call(work, "list_expenses", {"employee": "Іван Шевченко", "month": "2026-01"})
    assert r["count"] == 2
    assert call(work, "list_expenses", {"employee": "a", "month": "січень"})["error"] == "invalid_month"
