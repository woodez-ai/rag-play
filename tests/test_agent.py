"""Agent loop with a scripted fake LLM (no Ollama): routing, guards, relaxation, faithfulness."""

import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from ragplay import agent, categorize, db, ingest
from ragplay.categorize import Classification

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 11, 14)


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake")
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_statements.py", "--months", "2",
                    "--end", "2026-01", "--out", out], check=True, capture_output=True)
    db.migrate(db.TEST_SCHEMA)
    c = db.connect_owner(db.TEST_SCHEMA)
    c.autocommit = True
    assert c.execute("SELECT current_schema()").fetchone()[0] == db.TEST_SCHEMA
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories CASCADE")
    for pdf in sorted(out.glob("*.pdf")):
        assert ingest.ingest_file(c, pdf).status == "loaded"
    mp = pytest.MonkeyPatch()
    mp.setattr(categorize, "llm_classify", lambda key, raw, model=None: Classification(
        key.split()[0].title(), "home" if "DEPOT" in key else "other", "test shop", "llm"))
    categorize.categorize(c)
    mp.undo()
    yield c
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories CASCADE")
    c.close()


def call(name, **args):
    return NS(function=NS(name=name, arguments=args))


def script(monkeypatch, *replies, sql_replies=()):
    """Fake the agent's LLM turns, and (if given) the text-to-SQL LLM turns."""
    turns = iter(replies)
    seen = []

    def fake_chat(messages, model, tools=None):
        seen.append(messages)
        r = next(turns)
        return NS(content=r, tool_calls=None) if isinstance(r, str) else NS(content="", tool_calls=r)

    monkeypatch.setattr(agent, "_chat", fake_chat)
    sql_turns = iter(sql_replies)
    monkeypatch.setattr(agent.text2sql, "_chat", lambda *a, **k: next(sql_turns))
    return seen


def test_decline_without_tools(conn, monkeypatch):
    script(monkeypatch, "I don't have data about credit scores.")
    ans = agent.ask(conn, "What is my credit score?", today=TODAY)
    assert ans.calls == [] and ans.final is None and ans.unsupported == []


def test_query_database_gets_the_users_question(conn, monkeypatch):
    """The model's own arguments are ignored: paraphrasing changed the meaning before."""
    seen_sql_prompts = []

    def fake_sql_chat(prompt, *a, **k):
        seen_sql_prompts.append(prompt)
        return "```sql\nSELECT count(*) AS n FROM transactions\n```"

    script(monkeypatch, [call("query_database", question="paraphrased!")], "There are N.")
    monkeypatch.setattr(agent.text2sql, "_chat", fake_sql_chat)
    q = "How many transactions are on my December 2025 statement?"
    ans = agent.ask(conn, q, today=TODAY)
    assert "Q: " + q in seen_sql_prompts[0] and "paraphrased" not in seen_sql_prompts[0]
    assert ans.tools == ["query_database"] and ans.final.columns == ["n"]


@pytest.mark.parametrize("q,routed", [
    ("How much did I spend on dining in Q2 2026?", True),
    ("How many coffee purchases did I make?", True),
    ("What was my single largest purchase?", True),
    ("Which month had my highest spending?", True),
    ("How much interest was I charged in 2025?", True),
    ("Show me my coffee purchases in March 2026.", False),
    ("Which subscriptions do I pay for regularly?", False),
    ("Which of my subscriptions went up in price?", False),
    ("Were there any duplicate charges?", False),
    ("What is my credit score?", False),
])
def test_calculation_router(q, routed):
    assert bool(agent.CALC_RE.search(q) and not agent.ANOMALY_RE.search(q)) is routed


def test_calculation_goes_straight_to_sql(conn, monkeypatch):
    """No agent turn at all: the LLM never gets to pick find_transactions for a total."""
    seen = script(monkeypatch, sql_replies=["```sql\nSELECT count(*) AS n FROM transactions\n```",
                                            "There are 3 transactions."])
    ans = agent.ask(conn, "How many transactions do I have?", today=TODAY)
    assert seen == [] and ans.tools == ["query_database"] and ans.final.columns == ["n"]


def test_find_returns_rows_with_ids(conn, monkeypatch):
    script(monkeypatch, [call("find_transactions", merchant="safeway")], "Here they are.")
    ans = agent.ask(conn, "Show my Safeway purchases", today=TODAY)
    rows = ans.calls[0].rows
    assert rows and all(r[2] == "Safeway" for r in rows)
    assert ans.final.columns[0] == "id"


def test_unknown_find_arguments_are_refused(conn, monkeypatch):
    seen = script(monkeypatch, [call("find_transactions", sort="amount", limit=1)], "Use the other tool.")
    ans = agent.ask(conn, "Show my top Safeway purchase", today=TODAY)  # not routed: goes to the agent
    assert ans.calls[0].error and "query_database" in ans.calls[0].error
    assert "ERROR" in seen[1][-1]["content"]  # the error is fed back to the model


def test_redundant_filters_are_relaxed(conn, monkeypatch):
    script(monkeypatch, [call("find_transactions", category="home", merchant="hardware store")], "Done.")
    ans = agent.ask(conn, "Show my hardware store purchases", today=TODAY)
    c = ans.calls[0]
    assert c.rows and c.arguments.get("relaxed") == "dropped merchant"
    assert all(r[4] == "home" for r in c.rows)


def test_unsupported_number_is_flagged(conn, monkeypatch):
    script(monkeypatch, [call("find_transactions", merchant="safeway")], "You spent $999,999.99 there.")
    ans = agent.ask(conn, "Safeway?", today=TODAY)
    assert "999999.99" in ans.unsupported


def test_followup_is_rewritten(conn, monkeypatch):
    seen = script(monkeypatch, "Show me my dining purchases in April 2026.", "No data for April.")
    ans = agent.ask(conn, "and in April?", today=TODAY,
                    history=[("Show me my dining purchases in March 2026.", "Here they are.")])
    assert ans.standalone == "Show me my dining purchases in April 2026."
    assert "and in April?" in seen[0][0]["content"]  # rewrite prompt saw the follow-up


def test_no_history_no_rewrite(conn, monkeypatch):
    seen = script(monkeypatch, "I don't have that data.")
    agent.ask(conn, "What is my credit score?", today=TODAY)
    assert len(seen) == 1  # no rewrite call
