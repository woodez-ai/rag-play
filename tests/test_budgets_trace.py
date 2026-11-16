"""Budgets (computed in code) and the JSONL trace log."""

import json
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from ragplay import agent, budgets, categorize, db, ingest, trace
from ragplay.agent import AgentAnswer, ToolCall
from ragplay.categorize import Classification
from ragplay.text2sql import Answer, Attempt

ROOT = Path(__file__).resolve().parents[1]
D = Decimal


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake")
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_statements.py", "--months", "2",
                    "--end", "2026-01", "--out", out], check=True, capture_output=True)
    truth = [t for js in sorted(out.glob("*.json")) for t in json.loads(js.read_text())["transactions"]]
    cats = {t["description"].split()[0]: t["category"] for t in truth}
    db.migrate(db.TEST_SCHEMA)
    c = db.connect_owner(db.TEST_SCHEMA)
    assert c.execute("SELECT current_schema()").fetchone()[0] == db.TEST_SCHEMA
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories, budgets CASCADE")
    for pdf in sorted(out.glob("*.pdf")):
        assert ingest.ingest_file(c, pdf).status == "loaded"
    mp = pytest.MonkeyPatch()
    mp.setattr(categorize, "llm_classify", lambda key, raw, model=None: Classification(
        key.split()[0].title(), cats.get(raw.split()[0]) or "other", "x", "llm"))
    categorize.categorize(c)
    mp.undo()
    yield c, truth
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories, budgets CASCADE")
    c.close()


@pytest.fixture(autouse=True)
def no_budgets(request):
    """Each budget test starts clean (a failing test must not leave budgets behind)."""
    if "conn" in request.fixturenames:
        request.getfixturevalue("conn")[0].execute("DELETE FROM budgets")


def spent(truth, cat, month):
    return sum((D(t["amount"]) for t in truth if t["txn_date"].startswith(month)
                and t["type"] in ("purchase", "refund") and (cat == "all" or t["category"] == cat)), D(0))


def test_set_list_remove(conn):
    c, _ = conn
    budgets.set_budget(c, "groceries", D("500"))
    budgets.set_budget(c, "groceries", D("650"))  # update, not duplicate
    assert dict(budgets.list_budgets(c))["groceries"] == D("650.00")
    with pytest.raises(ValueError):
        budgets.set_budget(c, "not-a-category", D("10"))
    with pytest.raises(ValueError):
        budgets.set_budget(c, "groceries", D("0"))
    assert budgets.remove_budget(c, "groceries") and not budgets.remove_budget(c, "groceries")


def test_status_matches_ground_truth(conn):
    c, truth = conn
    budgets.set_budget(c, "groceries", D("100"))
    budgets.set_budget(c, "all", D("100000"))
    month, items = budgets.status(c, "2025-12", today=date(2026, 3, 1))
    by = {b.category: b for b in items}
    assert by["groceries"].spent == spent(truth, "groceries", "2025-12")
    assert by["groceries"].status == "over" and by["groceries"].remaining < 0
    assert by["all"].spent == spent(truth, "all", "2025-12") and by["all"].status == "on track"
    assert by["all"].projected is None  # December fully covered by the data
    budgets.remove_budget(c, "groceries"), budgets.remove_budget(c, "all")


def test_projection_when_data_stops_mid_month(conn):
    c, truth = conn
    budgets.set_budget(c, "all", D("100000"))
    # the last statement closes Feb 14, so February is only partly covered (a past month)
    last = date.fromisoformat(max(t["txn_date"] for t in truth))
    assert last.month == 2 and last.day < 28
    _, items = budgets.status(c, "2026-02", today=date(2026, 6, 1))
    b = items[0]
    assert b.data_through == last and b.spent == spent(truth, "all", "2026-02")
    assert b.projected == (b.spent / last.day * 28).quantize(D("0.01"))


def test_budget_questions_are_not_routed_to_sql():
    for q in ("How much of my dining budget is left?", "Am I over budget this month?"):
        assert agent.ANOMALY_RE.search(q)


def test_budget_tool(conn, monkeypatch):
    c, _ = conn
    budgets.set_budget(c, "dining", D("50"))
    replies = iter([[NS(function=NS(name="budget_status", arguments={"month": "2025-12"}))], "Done."])
    monkeypatch.setattr(agent, "_chat", lambda *a, **k: (lambda r: NS(content=r, tool_calls=None) if isinstance(r, str)
                                                         else NS(content="", tool_calls=r))(next(replies)))
    ans = agent.ask(c, "Am I over my dining budget in December 2025?", today=date(2026, 3, 1))
    call = ans.calls[0]
    assert call.name == "budget_status" and call.columns[:2] == ["month", "category"]
    assert call.rows[0][0] == "2025-12" and call.rows[0][1] == "dining"


def test_budget_tool_returns_every_budget_even_if_asked_for_all(conn, monkeypatch):
    """The model passed category='all' meaning 'all my budgets'; the tool must not narrow."""
    c, _ = conn
    budgets.set_budget(c, "dining", D("50"))
    budgets.set_budget(c, "all", D("100000"))
    replies = iter([[NS(function=NS(name="budget_status", arguments={"month": "2025-12", "category": "all"}))], "x"])
    monkeypatch.setattr(agent, "_chat", lambda *a, **k: (lambda r: NS(content=r, tool_calls=None) if isinstance(r, str)
                                                         else NS(content="", tool_calls=r))(next(replies)))
    ans = agent.ask(c, "Which budgets did I go over in December 2025?", today=date(2026, 3, 1))
    assert {r[1] for r in ans.calls[0].rows} == {"dining", "all"}


def test_budget_tool_month_without_data(conn, monkeypatch):
    c, _ = conn
    budgets.set_budget(c, "dining", D("50"))
    replies = iter([[NS(function=NS(name="budget_status", arguments={"month": "2026-11"}))], "x"])
    monkeypatch.setattr(agent, "_chat", lambda *a, **k: (lambda r: NS(content=r, tool_calls=None) if isinstance(r, str)
                                                         else NS(content="", tool_calls=r))(next(replies)))
    ans = agent.ask(c, "Am I on track with my budgets this month?", today=date(2026, 11, 20))
    assert ans.calls[0].error and "no transactions recorded for 2026-11" in ans.calls[0].error


def test_trace_record_and_recent(tmp_path):
    ans = AgentAnswer("and in April?", "Spend in April 2026?", text="$12.50",
                      calls=[ToolCall("query_database", {"routed": "calculation"}, ["total"],
                                      [(D("12.50"),)], "SELECT 1", seconds=1.23)], seconds=2.0)
    e = trace.record(ans, system="agent", model="m", schema="s", session="chat-1", path=tmp_path / "t.jsonl")
    assert e["route"] == "router: sql" and e["steps"][0]["rows"] == [["12.50"]]
    sql = Answer("q2", [Attempt("SELECT 2", columns=["n"], rows=[(1,)])], text="1", seconds=1.0)
    trace.record(sql, system="sql", model="m", schema="s", path=tmp_path / "t.jsonl")
    got = trace.recent(10, tmp_path)
    assert [g["question"] for g in got] == ["and in April?", "q2"]
    assert got[0]["standalone"] == "Spend in April 2026?" and got[1]["route"] == "text2sql"
    assert json.loads((tmp_path / "t.jsonl").read_text().splitlines()[0])["session"] == "chat-1"
