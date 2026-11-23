"""Pages. Each view calls the same ragplay functions as the CLI; no numbers are computed here."""

import json
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import ollama
import psycopg
from django.conf import settings
from django.http import Http404, HttpRequest, HttpResponse, HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.utils.text import get_valid_filename
from django.views.decorators.http import require_POST

from ragplay import agent, anomalies, budgets, categorize, db, stats, store, text2sql, trace
from ragplay_web import jobs

NAV = [("dashboard", "Dashboard"), ("ask", "Ask"), ("spending", "Spending"), ("insights", "Insights"),
       ("budgets", "Budgets"), ("categories", "Categories"), ("import", "Import"), ("history", "History")]
STARTERS = [
    "How much did I spend on groceries last month?",
    "Which subscriptions do I pay for?",
    "Show my largest purchases this year",
    "Am I over any budgets this month?",
]
CHAT_TURNS = 30  # kept in the session (the agent itself only uses the last 3 for rewriting)


def nav(request: HttpRequest) -> dict:
    """Context for every page: sidebar links and which schema is shown."""
    match = request.resolver_match
    section = match.url_name.split("_")[0] if match and match.url_name else ""
    return {"nav": NAV, "section": section, "schema": settings.RAG_SCHEMA,
            "is_real": settings.RAG_SCHEMA == db.SCHEMA}


def reader() -> psycopg.Connection:
    """Read-only role on the real schema; synthetic schemas have no reader role (as in the CLI)."""
    return db.connect_reader() if settings.RAG_SCHEMA == db.SCHEMA else db.connect_owner(settings.RAG_SCHEMA)


def owner() -> psycopg.Connection:
    """Owner role, only for the user's own edits (budgets; later categories and import)."""
    return db.connect_owner(settings.RAG_SCHEMA)


def _date(s: str | None) -> date | None:
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


def _money(s: str | None) -> Decimal | None:
    try:
        d = Decimal((s or "").replace(",", "").replace("$", "").strip())
    except InvalidOperation:
        return None
    return d.quantize(Decimal("0.01")) if d.is_finite() else None


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
        "month_start": stats.bounds(month)[0], "month_end": stats.bounds(month)[1] - timedelta(days=1),
        "cats": [(c, a, float(a / top * 100)) for c, a in cats],
        "budgets": [(b, min(float(b.pct_used), 100.0)) for b in budget_items],
        "over": sum(b.status == "over" for b in budget_items),
        "chart": {"labels": [m for m, _ in trend], "values": [float(v) for _, v in trend], "month": month},
    })


def attention(request: HttpRequest) -> HttpResponse:
    """Loaded after the dashboard renders (recurring detection scans all purchases)."""
    with reader() as conn:
        month = _month_arg(request, stats.months(conn))
        items: list[tuple[str, str, str]] = []  # (kind, text, link)
        if month:
            start, nxt = stats.bounds(month)
            end = nxt - timedelta(days=1)
            for s in anomalies.detect_recurring(conn):
                for ch in s.changes:
                    if start <= ch.on <= end:
                        verb = "went up" if ch.new > ch.old else "went down"
                        items.append(("price", f"{s.merchant} {verb} from ${ch.old:,.2f} to ${ch.new:,.2f} on {ch.on:%b %-d}",
                                      "/insights/subscriptions?changed=1"))
            for p in anomalies.find_duplicates(conn, date_from=start, date_to=end):
                items.append(("duplicate", f"Possible duplicate: {p[2]} ${p[3]:,.2f} on {p[1]:%b %-d}",
                              "/insights/duplicates"))
            for o in anomalies.spend_outliers(conn, date_from=start, date_to=end):
                items.append(("outlier", f"Unusually large: {o.merchant} ${o.amount:,.2f} on {o.txn_date:%b %-d} ({o.reason})",
                              f"/insights/unusual?from={start}&to={end}"))
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


# ---------------------------------------------------------------- spending

PAGE_SIZE = 50
FIND_LIMIT = 2000  # filtered lists show at most this many rows (paged); text search shows the top matches
SEARCH_TOP = 50


def spending(request: HttpRequest) -> HttpResponse:
    g = request.GET
    f = {"q": g.get("q", "").strip()[:200], "merchant": g.get("merchant", "").strip()[:100],
         "category": g.get("category", ""), "from": _date(g.get("from")), "to": _date(g.get("to")),
         "min": _money(g.get("min")), "max": _money(g.get("max")), "card": g.get("card", "")}
    with reader() as conn:
        cats = stats.categories(conn)
        if f["category"] not in cats:
            f["category"] = ""
        cards = stats.accounts(conn)
        if f["card"] not in dict(cards):
            f["card"] = ""
        hits = store.find(conn, text=f["q"] or None, merchant=f["merchant"] or None,
                          category=f["category"] or None, date_from=f["from"], date_to=f["to"],
                          min_amount=f["min"], max_amount=f["max"], account=f["card"] or None,
                          k=SEARCH_TOP if f["q"] else FIND_LIMIT)
    try:
        page = max(1, int(g.get("page", 1)))
    except ValueError:
        page = 1
    pages = max(1, -(-len(hits) // PAGE_SIZE))
    page = min(page, pages)
    query = g.copy()
    query.pop("page", None)
    return render(request, "ragplay_web/spending.html", {
        "f": f, "categories": cats, "cards": cards, "rows": hits[(page - 1) * PAGE_SIZE: page * PAGE_SIZE],
        "count": len(hits), "total": sum((h.amount for h in hits), Decimal(0)),
        "ranked": bool(f["q"]), "capped": not f["q"] and len(hits) == FIND_LIMIT,
        "page": page, "pages": pages, "query": query.urlencode(),
        "filtered": any(v for v in f.values()),
    })


# ---------------------------------------------------------------- insights

TABS = [("subscriptions", "Subscriptions"), ("duplicates", "Duplicates"), ("unusual", "Unusual charges"),
        ("review", "Monthly review")]
DAY_OPTIONS = [(0, "Same day"), (1, "1 day"), (3, "3 days"), (7, "7 days")]


def insights(request: HttpRequest, tab: str = "subscriptions") -> HttpResponse:
    if tab not in dict(TABS):
        raise Http404
    g = request.GET
    ctx: dict = {"tabs": TABS, "tab": tab}
    with reader() as conn:
        if tab == "subscriptions":
            series = anomalies.detect_recurring(conn)
            show_all, changed = g.get("all") == "1", g.get("changed") == "1"
            active = [s for s in series if s.active]
            shown = [s for s in series if (show_all or s.active) and (not changed or s.changes)]
            per_month = sum((s.amount if s.cadence == "monthly" else s.amount / 12 for s in active), Decimal(0))
            ctx.update(series=sorted(shown, key=lambda s: (not s.active, -s.amount)), show_all=show_all,
                       changed=changed, n_active=len(active), per_month=per_month.quantize(Decimal("0.01")),
                       n_increased=sum(s.increased for s in active))
        elif tab == "duplicates":
            days = int(g["days"]) if g.get("days", "").isdigit() else 0
            days = days if days in dict(DAY_OPTIONS) else 0
            ctx.update(days=days, day_options=DAY_OPTIONS, pairs=anomalies.find_duplicates(conn, days=days)[::-1])
        elif tab == "unusual":
            f, t = _date(g.get("from")), _date(g.get("to"))
            ctx.update(outliers=anomalies.spend_outliers(conn, date_from=f, date_to=t)[::-1], date_from=f, date_to=t)
        else:
            months = stats.months(conn)
            ctx.update(months=months, month=g.get("month") if g.get("month") in months else None)
    return render(request, "ragplay_web/insights.html", ctx)


def review_run(request: HttpRequest) -> HttpResponse:
    """The LLM-written review (~20 s), requested by the Monthly review tab."""
    with reader() as conn:
        months = stats.months(conn)
        month = request.GET.get("month")
        facts, text, unsupported = agent.review(conn, month if month in months else None, model=agent.MODEL)
    return render(request, "ragplay_web/_review.html", {"facts": facts, "text": text, "unsupported": unsupported})


# ---------------------------------------------------------------- budgets

def _budgets_ctx(conn: psycopg.Connection, month: str | None, error: str = "") -> dict:
    months = stats.months(conn)
    month, items = budgets.status(conn, month if month in months else None)
    have = {b.category for b in items}
    return {"months": months, "month": month, "items": [(b, min(float(b.pct_used), 100.0)) for b in items],
            "addable": [c for c in [budgets.ALL, *stats.categories(conn)] if c not in have and c != "payment"],
            "error": error}


def budgets_page(request: HttpRequest) -> HttpResponse:
    with reader() as conn:
        return render(request, "ragplay_web/budgets.html", _budgets_ctx(conn, request.GET.get("month")))


@require_POST
def budget_set(request: HttpRequest) -> HttpResponse:
    category, amount = request.POST.get("category", ""), _money(request.POST.get("amount"))
    error = ""
    with owner() as conn:
        try:
            if amount is None:
                raise ValueError("enter an amount, e.g. 400")
            budgets.set_budget(conn, category, amount)
        except ValueError as e:
            error = str(e)
        ctx = _budgets_ctx(conn, request.POST.get("month"), error)
    return _budgets_response(request, ctx)


@require_POST
def budget_remove(request: HttpRequest) -> HttpResponse:
    with owner() as conn:
        budgets.remove_budget(conn, request.POST.get("category", ""))
        ctx = _budgets_ctx(conn, request.POST.get("month"))
    return _budgets_response(request, ctx)


def _budgets_response(request: HttpRequest, ctx: dict) -> HttpResponse:
    if request.headers.get("HX-Request"):
        return render(request, "ragplay_web/_budgets.html", ctx)
    return redirect(f"/budgets?month={ctx['month']}")


# ---------------------------------------------------------------- categories

SOURCES = [("", "Any source"), ("llm", "Set by AI"), ("rule", "Set by rule"), ("manual", "Set by you")]


def categories(request: HttpRequest) -> HttpResponse:
    g = request.GET
    cat = g.get("category") if g.get("category") in categorize.CATEGORIES else None
    src = g.get("source") if g.get("source") in ("llm", "rule", "manual") else None
    name = g.get("name", "").strip()[:100] or None
    with reader() as conn:
        summary = categorize.summary(conn)
        rows = categorize.merchants(conn, category=cat, source=src, name=name)
        to_review = len(categorize.merchants(conn, source="llm"))
    try:
        page = max(1, int(g.get("page", 1)))
    except ValueError:
        page = 1
    pages = max(1, -(-len(rows) // PAGE_SIZE))
    page = min(page, pages)
    query = g.copy()
    query.pop("page", None)
    return render(request, "ragplay_web/categories.html", {
        "summary": summary, "rows": rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE], "count": len(rows),
        "choices": categorize.CATEGORIES, "sources": SOURCES, "f": {"category": cat, "source": src, "name": name},
        "page": page, "pages": pages, "query": query.urlencode(), "to_review": to_review,
    })


@require_POST
def category_set(request: HttpRequest) -> HttpResponse:
    """Change one merchant's category (becomes source=manual). Returns the updated row."""
    merchant, category = request.POST.get("merchant", ""), request.POST.get("category", "")
    with owner() as conn:
        try:
            categorize.set_category(conn, merchant, category)
        except ValueError as e:
            return HttpResponseBadRequest(str(e))
        row = categorize.merchants(conn, name=merchant)
    row = next((r for r in row if r[0] == merchant), None)
    if request.headers.get("HX-Request"):
        return render(request, "ragplay_web/_merchant_row.html", {"r": row, "choices": categorize.CATEGORIES, "saved": True})
    return redirect(request.META.get("HTTP_REFERER") or "categories")


def _review_card(request: HttpRequest, conn: psycopg.Connection) -> dict:
    skipped = set(request.session.get("review_skipped", []))
    queue = [r for r in categorize.merchants(conn, source="llm") if r[0] not in skipped]
    r = queue[0] if queue else None
    return {"r": r, "left": len(queue), "skipped": len(skipped), "choices": categorize.MERCHANT_CATEGORIES,
            "examples": categorize.examples(conn, r[0]) if r else []}


def category_review(request: HttpRequest) -> HttpResponse:
    """One AI-categorised merchant at a time, most transactions first."""
    if request.GET.get("restart"):
        request.session.pop("review_skipped", None)
    with reader() as conn:
        ctx = _review_card(request, conn)
    return render(request, "ragplay_web/review_queue.html", ctx)


@require_POST
def category_review_answer(request: HttpRequest) -> HttpResponse:
    merchant, action = request.POST.get("merchant", ""), request.POST.get("action", "")
    with owner() as conn:
        if action == "skip":
            request.session["review_skipped"] = request.session.get("review_skipped", []) + [merchant]
        else:  # "keep" confirms the current category, any other value is a new category
            current = next((r[1] for r in categorize.merchants(conn, name=merchant) if r[0] == merchant), None)
            try:
                categorize.set_category(conn, merchant, current if action == "keep" else action)
            except ValueError as e:
                return HttpResponseBadRequest(str(e))
        ctx = _review_card(request, conn)
    if request.headers.get("HX-Request"):
        return render(request, "ragplay_web/_review_card.html", ctx)
    return redirect("categories_review")


# ---------------------------------------------------------------- import

def _save_upload(f, directory: Path) -> Path | str:
    """Store one uploaded PDF; returns its path, or an error message."""
    name = Path(f.name or "").name
    if not name.lower().endswith(".pdf"):
        return f"{name or 'file'}: not a PDF"
    if f.size > settings.MAX_PDF_BYTES:
        return f"{name}: larger than {settings.MAX_PDF_BYTES // 2**20} MB"
    head = f.read(5)
    f.seek(0)
    if head != b"%PDF-":
        return f"{name}: not a PDF file"
    name = get_valid_filename(name)
    directory.mkdir(parents=True, exist_ok=True)
    data = f.read()
    path = directory / name
    stem, n = path.stem, 2
    while path.exists():
        if path.read_bytes() == data:  # the same file again: reuse it (ingest will skip it)
            return path
        path, n = directory / f"{stem}-{n}.pdf", n + 1
    path.write_bytes(data)
    path.chmod(0o600)
    return path


def import_page(request: HttpRequest) -> HttpResponse:
    with reader() as conn:
        statements = stats.statements(conn)
    return render(request, "ragplay_web/import.html", {
        "statements": statements, "job": jobs.current(), "max_mb": settings.MAX_PDF_BYTES // 2**20,
        "import_dir": settings.IMPORT_DIR.relative_to(db.ROOT) if settings.IMPORT_DIR.is_relative_to(db.ROOT) else settings.IMPORT_DIR,
    })


@require_POST
def import_upload(request: HttpRequest) -> HttpResponse:
    files = request.FILES.getlist("pdfs")
    paths, problems = [], []
    for f in files:
        out = _save_upload(f, settings.IMPORT_DIR)
        (paths if isinstance(out, Path) else problems).append(out)
    job, error = None, ""
    if paths:
        try:
            job = jobs.start(settings.RAG_SCHEMA, paths)
        except RuntimeError as e:
            error = str(e)
    elif not files:
        error = "Choose one or more statement PDFs."
    return render(request, "ragplay_web/_job.html", {"job": job, "problems": problems, "error": error})


def import_job(request: HttpRequest, job_id: str) -> HttpResponse:
    job = jobs.get(job_id)
    if job is None:
        raise Http404
    resp = render(request, "ragplay_web/_job.html", {"job": job})
    if job.done:
        resp.status_code = 286  # htmx: stop polling
    return resp
