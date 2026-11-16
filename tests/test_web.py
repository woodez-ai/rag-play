"""Web UI (Django) on synthetic data in the test schema. The LLM is mocked."""

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

django = pytest.importorskip("django")
os.environ["DJANGO_SETTINGS_MODULE"] = "ragplay_web.settings"
os.environ["RAGPLAY_WEB_SCHEMA"] = "ragplay_test"
django.setup()

from django.conf import settings  # noqa: E402
from django.test import Client  # noqa: E402
from django.test.utils import setup_test_environment  # noqa: E402

setup_test_environment()  # records response.context

from ragplay import agent, budgets, categorize, db, ingest, stats, text2sql, trace  # noqa: E402
from ragplay.agent import AgentAnswer, ToolCall  # noqa: E402
from ragplay.categorize import Classification  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = Decimal


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake")
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_statements.py", "--months", "3",
                    "--end", "2026-01", "--out", out], check=True, capture_output=True)
    truth = [t for js in sorted(out.glob("*.json")) for t in json.loads(js.read_text())["transactions"]]
    cats = {t["description"].split()[0]: t["category"] for t in truth}
    db.migrate(db.TEST_SCHEMA)
    c = db.connect_owner(db.TEST_SCHEMA)
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories, budgets CASCADE")
    for pdf in sorted(out.glob("*.pdf")):
        assert ingest.ingest_file(c, pdf).status == "loaded"
    mp = pytest.MonkeyPatch()
    mp.setattr(categorize, "llm_classify", lambda key, raw, model=None: Classification(
        key.split()[0].title(), cats.get(raw.split()[0]) or "other", "x", "llm"))
    categorize.categorize(c)
    mp.undo()
    budgets.set_budget(c, "dining", D("1"))
    yield c, truth
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories, budgets CASCADE")
    c.close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    assert settings.RAG_SCHEMA == db.TEST_SCHEMA
    monkeypatch.setattr(settings, "SESSION_FILE_PATH", str(tmp_path))
    monkeypatch.setattr(trace, "TRACES", tmp_path / "traces")
    return Client(HTTP_HOST="127.0.0.1")


def spent(truth, month, cat=None):
    return sum((D(t["amount"]) for t in truth if t["txn_date"].startswith(month)
                and t["type"] in ("purchase", "refund") and (cat is None or t["category"] == cat)), D(0))


def test_dashboard_numbers_match_ground_truth(data, client):
    c, truth = data
    r = client.get("/?month=2025-12")
    assert r.status_code == 200
    ctx = r.context
    assert ctx["cur"].spent == spent(truth, "2025-12") and ctx["cur"].complete
    assert dict((k, a) for k, a, _ in ctx["cats"])["groceries"] == spent(truth, "2025-12", "groceries")
    assert ctx["over"] == 1  # dining budget of $1
    assert ctx["chart"]["labels"][-1] == "2025-12" and ctx["chart"]["labels"][0] == min(stats.months(c))
    assert b'id="trend-data"' in r.content


def test_dashboard_defaults_to_latest_month_and_flags_partial(data, client):
    c, _ = data
    r = client.get("/?month=not-a-month")
    assert r.context["month"] == stats.months(c)[0]
    assert not r.context["cur"].complete and b"isn't complete yet" in r.content


def test_attention_loads(data, client):
    assert client.get("/attention?month=2025-12").status_code == 200


def test_ask_keeps_conversation_and_passes_history(data, client, monkeypatch):
    seen = []

    def fake_ask(conn, question, *, model, history):
        seen.append(list(history))
        standalone = question if not history else "Groceries in January 2026?"
        return AgentAnswer(question, standalone, text=f"answer {len(seen)}",
                           calls=[ToolCall("query_database", {"routed": "calculation"}, ["total"],
                                           [(D("12.50"),)], "SELECT 1")], seconds=1.0)

    monkeypatch.setattr(agent, "ask", fake_ask)
    r = client.post("/ask/send", {"q": "Groceries in December 2025?", "mode": "agent"}, HTTP_HX_REQUEST="true")
    assert r.status_code == 200 and b"answer 1" in r.content and b"query_database" in r.content
    r = client.post("/ask/send", {"q": "and January?", "mode": "agent"}, HTTP_HX_REQUEST="true")
    assert b"Understood as: Groceries in January 2026?" in r.content
    assert seen[1] == [("Groceries in December 2025?", "answer 1")]
    page = client.get("/ask")
    assert page.content.count(b'class="turn"') == 2
    # traced with this schema and a web session id
    entries = trace.recent(10)
    assert [e["schema"] for e in entries] == [db.TEST_SCHEMA] * 2
    assert entries[0]["session"].startswith("web-") and entries[0]["session"] == entries[1]["session"]
    client.post("/ask/new")
    assert b'class="turn"' not in client.get("/ask").content


def test_ask_sql_only_mode(data, client, monkeypatch):
    called = {}

    def fake_sql(conn, question, *, model):
        called["q"] = question
        return text2sql.Answer(question, [text2sql.Attempt("SELECT 2", columns=["n"], rows=[(2,)])],
                               text="2", seconds=0.5)

    monkeypatch.setattr(text2sql, "ask", fake_sql)
    monkeypatch.setattr(agent, "ask", lambda *a, **k: pytest.fail("agent must not run"))
    r = client.post("/ask/send", {"q": "How many?", "mode": "sql"}, HTTP_HX_REQUEST="true")
    assert called["q"] == "How many?" and b"text2sql" in r.content and b"SQL only" in r.content


def test_empty_question_does_nothing(data, client):
    assert client.post("/ask/send", {"q": "  "}).status_code == 204


def test_history_shows_only_this_schema(data, client):
    a = AgentAnswer("synthetic q", "synthetic q", text="x")
    trace.record(a, system="agent", model="m", schema=db.TEST_SCHEMA)
    trace.record(AgentAnswer("real q", "real q", text="y"), system="agent", model="m", schema=db.SCHEMA)
    body = client.get("/history").content
    assert b"synthetic q" in body and b"real q" not in body


def test_csrf_and_host_are_enforced(data, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SESSION_FILE_PATH", str(tmp_path))
    strict = Client(HTTP_HOST="127.0.0.1", enforce_csrf_checks=True)
    assert strict.post("/ask/send", {"q": "x"}).status_code == 403
    assert Client(HTTP_HOST="evil.example").get("/").status_code == 400


def test_pages_set_security_headers(data, client):
    r = client.get("/ask")
    assert "script-src 'self'" in r["Content-Security-Policy"] and r["X-Frame-Options"] == "DENY"
