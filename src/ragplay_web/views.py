"""Pages. Each view calls the same ragplay functions as the CLI; no numbers are computed here."""

import json
from datetime import datetime, timedelta
from decimal import Decimal

import ollama
import psycopg
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST

from ragplay import agent, anomalies, budgets, db, stats, text2sql, trace

NAV = [("dashboard", "Dashboard"), ("ask", "Ask"), ("history", "History")]
STARTERS = [
    "How much did I spend on groceries last month?",
    "Which subscriptions do I pay for?",
    "Show my largest purchases this year",
    "Am I over any budgets this month?",
]
CHAT_TURNS = 30  # kept in the session (the agent itself only uses the last 3 for rewriting)


def nav(request: HttpRequest) -> dict:
    """Context for every page: sidebar links and which schema is shown."""
    return {"nav": NAV, "schema": settings.RAG_SCHEMA, "is_real": settings.RAG_SCHEMA == db.SCHEMA}


def reader() -> psycopg.Connection:
    """Read-only role on the real schema; synthetic schemas have no reader role (as in the CLI)."""
    return db.connect_reader() if settings.RAG_SCHEMA == db.SCHEMA else db.connect_owner(settings.RAG_SCHEMA)


def _month_arg(request: HttpRequest, available: list[str]) -> str | None:
    m = request.GET.get("month")
    return m if m in available else (available[0] if available else None)


# ---------------------------------------------------------------- dashboard

def dashboard(request: HttpRequest) -> HttpResponse:
    with reader() as conn:
        months = stats.months(conn)
        month = _month_arg(request, months)
        if month is None:
            return render(request, "ragplay_web/empty.html")
        cur = stats.summary(conn, month)
        prev_month = stats.previous(month)
        prev = stats.summary(conn, prev_month) if prev_month in months else None
        cats = stats.by_category(conn, month)
        trend = stats.trend(conn, month)
        _, budget_items = budgets.status(conn, month)
    change = None
    if prev and prev.spent and cur.complete:
        change = ((cur.spent - prev.spent) / prev.spent * 100).quantize(Decimal("1"))
    top = cats[0][1] if cats else Decimal(1)
    return render(request, "ragplay_web/dashboard.html", {
        "months": months, "month": month, "cur": cur, "prev": prev, "change": change,
        "change_abs": abs(change) if change is not None else None,
        "cats": [(c, a, float(a / top * 100)) for c, a in cats],
        "budgets": [(b, min(float(b.pct_used), 100.0)) for b in budget_items],
        "over": sum(b.status == "over" for b in budget_items),
        "chart": {"labels": [m for m, _ in trend], "values": [float(v) for _, v in trend], "month": month},
    })


def attention(request: HttpRequest) -> HttpResponse:
    """Loaded after the dashboard renders (recurring detection scans all purchases)."""
    with reader() as conn:
        month = _month_arg(request, stats.months(conn))
        items: list[tuple[str, str]] = []
        if month:
            start, nxt = stats.bounds(month)
            end = nxt - timedelta(days=1)
            for s in anomalies.detect_recurring(conn):
                for ch in s.changes:
                    if start <= ch.on <= end:
                        verb = "went up" if ch.new > ch.old else "went down"
                        items.append(("price", f"{s.merchant} {verb} from ${ch.old:,.2f} to ${ch.new:,.2f} on {ch.on:%b %-d}"))
            for p in anomalies.find_duplicates(conn, date_from=start, date_to=end):
                items.append(("duplicate", f"Possible duplicate: {p[2]} ${p[3]:,.2f} on {p[1]:%b %-d}"))
            for o in anomalies.spend_outliers(conn, date_from=start, date_to=end):
                items.append(("outlier", f"Unusually large: {o.merchant} ${o.amount:,.2f} on {o.txn_date:%b %-d} ({o.reason})"))
    return render(request, "ragplay_web/_attention.html", {"items": items})


# ---------------------------------------------------------------- ask

def _chat(request: HttpRequest) -> list[dict]:
    return request.session.get("chat", [])


def ask(request: HttpRequest) -> HttpResponse:
    return render(request, "ragplay_web/ask.html", {"chat": _chat(request), "starters": STARTERS})


@require_POST
def ask_send(request: HttpRequest) -> HttpResponse:
    question = request.POST.get("q", "").strip()[:500]
    sql_only = request.POST.get("mode") == "sql"
    if not question:
        return HttpResponse(status=204)
    chat = _chat(request)
    session_id = request.session.setdefault("chat_id", datetime.now().strftime("web-%Y%m%d-%H%M%S"))
    with reader() as conn:
        if sql_only:
            ans = text2sql.ask(conn, question, model=text2sql.MODEL)
            entry = trace.record(ans, system="sql", model=text2sql.MODEL, schema=settings.RAG_SCHEMA, session=session_id)
            text = ans.text or "I couldn't produce a working query for that."
        else:
            history = [(t["standalone"], t["answer"]) for t in chat]
            ans = agent.ask(conn, question, model=agent.MODEL, history=history)
            entry = trace.record(ans, system="agent", model=agent.MODEL, schema=settings.RAG_SCHEMA, session=session_id)
            text = ans.text
    turn = json.loads(json.dumps(entry, default=str))  # session storage is JSON
    turn["answer"] = text
    turn["mode"] = "sql" if sql_only else "agent"
    request.session["chat"] = (chat + [turn])[-CHAT_TURNS:]
    if request.headers.get("HX-Request"):
        return render(request, "ragplay_web/_turn.html", {"t": turn})
    return redirect("ask")


@require_POST
def ask_new(request: HttpRequest) -> HttpResponse:
    request.session.pop("chat", None)
    request.session.pop("chat_id", None)
    return redirect("ask")


# ---------------------------------------------------------------- history

def history(request: HttpRequest) -> HttpResponse:
    # only this schema's questions: `rag web --schema ragplay_sample` must not list real ones
    entries = [e for e in reversed(trace.recent(1000)) if e.get("schema") == settings.RAG_SCHEMA][:50]
    for e in entries:
        e["when"] = datetime.fromisoformat(e["ts"])
    return render(request, "ragplay_web/history.html", {"entries": entries})


# ---------------------------------------------------------------- status

def health(request: HttpRequest) -> HttpResponse:
    """Footer status chip: database and Ollama reachable?"""
    ok_db = ok_llm = False
    try:
        with reader() as conn:
            ok_db = conn.execute("SELECT 1").fetchone()[0] == 1
    except Exception:
        pass
    try:
        ok_llm = any(m.model == agent.MODEL for m in ollama.list().models)
    except Exception:
        pass
    return render(request, "ragplay_web/_health.html", {"ok_db": ok_db, "ok_llm": ok_llm})
