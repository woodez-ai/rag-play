"""`rag` command line. Later phases add ingest | ask | chat | eval."""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragplay import agent, anomalies, budgets, categorize, db, evaluate, ingest, store, text2sql, trace

app = typer.Typer(no_args_is_help=True, help="Credit card statement analyst (local RAG).")
db_app = typer.Typer(no_args_is_help=True, help="Database setup and checks.")
app.add_typer(db_app, name="db")
console = Console()


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply pending sql/ migrations as the owner role."""
    applied = db.migrate()
    console.print(f"Applied: {', '.join(applied)}" if applied else "Up to date.")


@db_app.command("check")
def db_check() -> None:
    """Connect as owner and read-only roles; show TLS, read-only state and schema."""
    table = Table("check", "role", "result")
    for row in db.check():
        table.add_row(*row)
    console.print(table)


@app.command("ingest")
def ingest_cmd(
    paths: list[Path] = typer.Argument(..., help="Statement PDFs or folders of them."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse and reconcile only."),
    schema: str = typer.Option(db.SCHEMA, help="Target schema (ragplay_test for synthetic)."),
) -> None:
    """Parse, reconcile and load statements. Each statement loads all-or-nothing."""
    files = ingest.pdf_paths(paths)
    conn = None if dry_run else db.connect_owner(schema)
    if conn is not None:
        conn.autocommit = True  # each statement is its own conn.transaction()
    table = Table("file", "status", "txns", "problems")
    results = []
    try:
        for f in files:
            r = ingest.ingest_file(conn, f)
            results.append(r)
            style = {"failed": "red", "skipped": "dim"}.get(r.status, "green")
            table.add_row(r.file, f"[{style}]{r.status}[/]", str(r.transactions),
                          "\n".join(r.errors))
    finally:
        if conn is not None:
            conn.close()
    console.print(table)
    failed = sum(r.status == "failed" for r in results)
    console.print(f"{len(results) - failed}/{len(results)} ok")
    if failed:
        raise typer.Exit(1)


@app.command("categorize")
def categorize_cmd(
    schema: str = typer.Option(db.SCHEMA, help="Target schema (ragplay_test for synthetic)."),
    model: str = typer.Option(categorize.MODEL),
    redo_llm: bool = typer.Option(False, "--redo-llm",
                                  help="Re-run the LLM on all non-rule, non-manual merchants."),
) -> None:
    """Normalise merchants and categorise them (rules, then LLM; cached; manual wins)."""
    with db.connect_owner(schema) as conn:
        conn.autocommit = True
        if redo_llm:
            categorize.forget_llm(conn)
        with console.status("categorising...") as status:
            st = categorize.categorize(
                conn, model, progress=lambda i, n: status.update(f"merchant {i}/{n}"))
        uncategorised = conn.execute(
            "SELECT count(*) FROM transactions WHERE category IS NULL").fetchone()[0]
    per = f" ({st.llm_seconds / st.llm:.1f}s each)" if st.llm else ""
    console.print(f"{st.keys} merchant keys: {st.cached} cached, {st.rule} by rule, "
                  f"{st.llm} by LLM{per}. {st.updated} transactions updated; "
                  f"{uncategorised} without a category.")


@app.command("embed")
def embed_cmd(
    model: str = typer.Option(store.DEFAULT_MODEL),
    schema: str = typer.Option(db.SCHEMA),
) -> None:
    """Embed each transaction's "merchant: description" text into pgvector (new/changed only)."""
    with db.connect_owner(schema) as conn:
        conn.autocommit = True
        with console.status(f"embedding with {model}..."):
            written, texts = store.embed_transactions(conn, model)
        total = conn.execute("SELECT count(*) FROM transaction_embeddings WHERE model = %s",
                             (model,)).fetchone()[0]
    console.print(f"{model}: embedded {texts} distinct texts -> {written} transactions "
                  f"({total} stored).")


@app.command("search")
def search_cmd(
    query: str,
    mode: str = typer.Option("hybrid", help="hybrid | vector | text"),
    date_from: str = typer.Option(None, "--from", help="YYYY-MM-DD"),
    date_to: str = typer.Option(None, "--to", help="YYYY-MM-DD"),
    min_amount: float = typer.Option(None, "--min"),
    max_amount: float = typer.Option(None, "--max"),
    category: str = typer.Option(None),
    k: int = typer.Option(10, "-k"),
    model: str = typer.Option(store.DEFAULT_MODEL),
    schema: str = typer.Option(db.SCHEMA),
) -> None:
    """Find transactions by meaning and/or name, with date/amount/category filters."""
    from datetime import date
    from decimal import Decimal

    dec = lambda x: None if x is None else Decimal(str(x))
    day = lambda s: None if s is None else date.fromisoformat(s)
    with db.connect_reader() if schema == db.SCHEMA else db.connect_owner(schema) as conn:
        hits = store.search(conn, query, mode=mode, model=model, k=k, date_from=day(date_from),
                            date_to=day(date_to), min_amount=dec(min_amount),
                            max_amount=dec(max_amount), category=category)
    table = Table("date", "merchant", "category", "amount", "score", "raw")
    for h in hits:
        table.add_row(str(h.txn_date), h.merchant or "", h.category or "", f"{h.amount:,.2f}",
                      f"{h.score:.4f}", h.description_raw)
    console.print(table)


@app.command("ask")
def ask_cmd(
    question: str,
    show_sql: bool = typer.Option(False, "--sql", help="Show tool calls / generated SQL."),
    sql_only: bool = typer.Option(False, "--sql-only", help="Phase 5 text-to-SQL only (no agent)."),
    think: bool = typer.Option(False, help="--sql-only: let qwen3 think before writing SQL."),
    model: str = typer.Option(text2sql.MODEL),
    schema: str = typer.Option(db.SCHEMA),
) -> None:
    """Answer one question with the agent (same as `rag chat`), or with --sql-only the Phase 5
    text-to-SQL path. Traced to data/traces/."""
    if not sql_only:
        with _reader(schema) as conn, console.status("thinking..."):
            ans = agent.ask(conn, question, model=model)
        trace.record(ans, system="agent", model=model, schema=schema)
        _print_agent_answer(ans, show_sql)
        return
    with db.connect_reader() if schema == db.SCHEMA else db.connect_owner(schema) as conn:
        with console.status("thinking..."):
            ans = text2sql.ask(conn, question, model=model, think=think)
    trace.record(ans, system="sql", model=model, schema=schema)
    if show_sql:
        for i, a in enumerate(ans.attempts, 1):
            status = a.error or ("no rows" if a.empty else f"{len(a.rows)} rows")
            console.print(f"[dim]attempt {i}: {status}[/]\n{a.sql}\n")
    final = ans.final
    if final is None:
        console.print("[red]Could not produce a working query.[/]")
        raise typer.Exit(1)
    if final.rows and len(final.rows) > 1 or show_sql:
        table = Table(*final.columns)
        for r in final.rows[:50]:
            table.add_row(*(text2sql._fmt(v) for v in r))
        console.print(table)
    console.print(ans.text)
    if ans.unsupported:
        console.print(f"[yellow]Warning: numbers not in the query result: {', '.join(ans.unsupported)}[/]")
    console.print(f"[dim]{ans.seconds:.0f}s[/]")


def _print_agent_answer(ans, verbose: bool) -> None:
    if ans.standalone != ans.question:
        console.print(f"[dim]({ans.standalone})[/]")
    if verbose:
        for c in ans.calls:
            console.print(f"[dim]{c.name}({c.arguments}) -> "
                          f"{c.error or str(len(c.rows)) + ' rows'}{chr(10) + c.sql if c.sql else ''}[/]")
    console.print(ans.text)
    if ans.unsupported:
        console.print(f"[yellow]Warning: not in any tool result: {', '.join(ans.unsupported)}[/]")
    console.print(f"[dim]{ans.seconds:.0f}s[/]")


@app.command("chat")
def chat_cmd(
    verbose: bool = typer.Option(False, "-v", help="Show tool calls and SQL."),
    model: str = typer.Option(agent.MODEL),
    schema: str = typer.Option(db.SCHEMA),
) -> None:
    """Conversation with the agent. Follow-ups ("and in April?") are rewritten into standalone
    questions from the last few turns. Empty line or Ctrl-D quits."""
    history: list[tuple[str, str]] = []
    session = datetime.now().strftime("chat-%Y%m%d-%H%M%S")
    with db.connect_reader() if schema == db.SCHEMA else db.connect_owner(schema) as conn:
        while True:
            try:
                q = console.input("[bold]> [/]").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            with console.status("thinking..."):
                ans = agent.ask(conn, q, model=model, history=history)
            trace.record(ans, system="agent", model=model, schema=schema, session=session)
            _print_agent_answer(ans, verbose)
            history.append((ans.standalone, ans.text))


def _reader(schema: str):
    return db.connect_reader() if schema == db.SCHEMA else db.connect_owner(schema)


def _table(cols: list[str], rows: list[tuple]) -> None:
    table = Table(*cols)
    for r in rows:
        table.add_row(*("" if v is None else text2sql._fmt(v) for v in r))
    console.print(table)


@app.command("recurring")
def recurring_cmd(category: str = typer.Option(None), all_: bool = typer.Option(
        False, "--all", help="Include series that have stopped."),
        schema: str = typer.Option(db.SCHEMA)) -> None:
    """Recurring charges (subscriptions, bills) with price changes and next expected date."""
    with _reader(schema) as conn:
        series = anomalies.detect_recurring(conn, category=category)
    if not all_:
        series = [s for s in series if s.active]
    _table(*anomalies.recurring_rows(series))
    console.print(f"{len(series)} series; {sum(s.increased for s in series)} with a price increase.")


@app.command("duplicates")
def duplicates_cmd(days: int = typer.Option(0, help="Max days apart (0 = same day)."),
                   schema: str = typer.Option(db.SCHEMA)) -> None:
    """Possible duplicate charges: same merchant and amount within --days."""
    with _reader(schema) as conn:
        _table(*anomalies.duplicate_rows(anomalies.find_duplicates(conn, days=days), conn))


@app.command("outliers")
def outliers_cmd(date_from: str = typer.Option(None, "--from"), date_to: str = typer.Option(None, "--to"),
                 schema: str = typer.Option(db.SCHEMA)) -> None:
    """Unusually large purchases compared with your own history."""
    from datetime import date
    day = lambda s: date.fromisoformat(s) if s else None
    with _reader(schema) as conn:
        _table(*anomalies.outlier_rows(anomalies.spend_outliers(conn, date_from=day(date_from),
                                                                date_to=day(date_to))))


@app.command("review")
def review_cmd(month: str = typer.Option(None, help="YYYY-MM (default: last full month)."),
               facts: bool = typer.Option(False, "--facts", help="Also print the computed facts."),
               model: str = typer.Option(agent.MODEL), schema: str = typer.Option(db.SCHEMA)) -> None:
    """Monthly review: facts computed in code, written up by the LLM."""
    import json
    with _reader(schema) as conn, console.status("reviewing..."):
        f, text, unsupported = agent.review(conn, month, model=model)
    if facts:
        console.print_json(json.dumps(f, default=str))
    console.print(f"[bold]Review of {f['month']}[/]\n{text}")
    if unsupported:
        console.print(f"[yellow]Warning: not in the computed facts: {', '.join(unsupported)}[/]")


budget_app = typer.Typer(no_args_is_help=True, help="Monthly budgets per category ('all' = total).")
app.add_typer(budget_app, name="budget")


@budget_app.command("set")
def budget_set(category: str, amount: str, schema: str = typer.Option(db.SCHEMA)) -> None:
    """Set a monthly budget, e.g. `rag budget set dining 400` or `rag budget set all 3000`."""
    try:
        value = Decimal(amount.replace("$", "").replace(",", ""))
    except InvalidOperation:
        raise typer.BadParameter(f"not an amount: {amount}")
    with db.connect_owner(schema) as conn:
        try:
            budgets.set_budget(conn, category, value)
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1)
    console.print(f"{category}: {value:,.2f} per month")


@budget_app.command("remove")
def budget_remove(category: str, schema: str = typer.Option(db.SCHEMA)) -> None:
    """Remove a budget."""
    with db.connect_owner(schema) as conn:
        console.print("removed" if budgets.remove_budget(conn, category) else f"no budget for {category}")


@budget_app.command("list")
def budget_list(schema: str = typer.Option(db.SCHEMA)) -> None:
    """Show all budgets."""
    with _reader(schema) as conn:
        _table(["category", "monthly budget"], budgets.list_budgets(conn))


@budget_app.command("status")
def budget_status_cmd(month: str = typer.Option(None, help="YYYY-MM (default: current, or latest with data)."),
                      schema: str = typer.Option(db.SCHEMA)) -> None:
    """Spent vs budget for a month, with a projection for a month in progress."""
    with _reader(schema) as conn:
        m, items = budgets.status(conn, month)
    if not items:
        console.print("No budgets set. Try `rag budget set dining 400`.")
        return
    through = items[0].data_through
    partial = through and items[0].projected is not None
    console.print(f"[bold]Budgets for {m}[/]" + (f" [yellow](data only through {through}; "
                                                 "projected = straight-line to month end)[/]" if partial else ""))
    colors = {"over": "red", "at risk": "yellow", "on track": "green"}
    table = Table(*budgets.rows(items)[0][:-1])
    for b in items:
        table.add_row(b.category, f"{b.budget:,.2f}", f"{b.spent:,.2f}", f"{b.remaining:,.2f}",
                      f"{b.pct_used}%", "" if b.projected is None else f"{b.projected:,.2f}",
                      f"[{colors[b.status]}]{b.status}[/]")
    console.print(table)


@app.command("trace")
def trace_cmd(last: int = typer.Option(15, "--last", "-n"),
              show: int = typer.Option(None, "--show", help="Show full detail of entry N from the list.")) -> None:
    """Recent questions from data/traces/ (route, tools, time, warnings); --show N for detail."""
    import json
    entries = trace.recent(last)
    if show is not None:
        if not 1 <= show <= len(entries):
            console.print(f"[red]choose 1..{len(entries)}[/]")
            raise typer.Exit(1)
        console.print_json(json.dumps(entries[show - 1], default=str))
        return
    table = Table("#", "time", "question", "route / tools", "s", "warn")
    for i, e in enumerate(entries, 1):
        tools = ", ".join(s["tool"] for s in e["steps"]) or "-"
        warn = ("unsupported " if e["unsupported"] else "") + ("error" if any(s.get("error") for s in e["steps"]) else "")
        table.add_row(str(i), e["ts"][5:16].replace("T", " "), e["question"][:60],
                      f"{e['route']}: {tools}"[:48], f"{e['seconds']:.0f}", warn)
    console.print(table)


@app.command("li-index")
def li_index_cmd(schema: str = typer.Option(db.SCHEMA)) -> None:
    """Phase 9: (re)build the LlamaIndex PGVectorStore index (schema ragplay_li)."""
    from ragplay_llamaindex.app import build_index
    with console.status("embedding every transaction with LlamaIndex..."):
        n = build_index(schema)
    console.print(f"LlamaIndex index for {schema}: {n} nodes.")


@app.command("li-ask")
def li_ask_cmd(question: str, verbose: bool = typer.Option(False, "-v"),
               schema: str = typer.Option(db.SCHEMA)) -> None:
    """Phase 9: ask the LlamaIndex rebuild (compare with `rag chat`)."""
    from ragplay_llamaindex.app import LlamaIndexAssistant
    with console.status("thinking (LlamaIndex)..."):
        ans = LlamaIndexAssistant(schema).ask(question)
    trace.record(ans, system="llamaindex", model="qwen3:8b", schema=schema)
    _print_agent_answer(ans, verbose)


@app.command("eval")
def eval_cmd(
    set_name: str = typer.Option("real", "--set", help="real (eval/questions.jsonl) | sample"),
    ids: str = typer.Option(None, help="Comma-separated question ids to run."),
    system: str = typer.Option("agent", help="agent (hand-built) | sql (Phase 5 baseline) | "
                                             "llamaindex (Phase 9 rebuild; needs `rag li-index`)"),
    model: str = typer.Option(text2sql.MODEL),
) -> None:
    """Run the eval set, print scores, save the run to eval/runs/ (gitignored).
    The real set prints pass/fail only; answers (with amounts) stay in the run file."""
    questions = evaluate.load(evaluate.SETS[set_name])
    if ids:
        keep = set(ids.split(","))
        questions = [q for q in questions if q["id"] in keep]
    real = set_name == "real"
    connect = db.connect_reader if real else (lambda: db.connect_owner(db.SAMPLE_SCHEMA))
    if not real:
        db.migrate(db.SAMPLE_SCHEMA)
        with connect() as conn, console.status("loading synthetic statements (first run only)..."):
            evaluate.ensure_sample(conn)

    def show(r: evaluate.Result) -> None:
        pr = "" if r.precision is None else f" P={r.precision:.2f} R={r.recall:.2f}"
        flags = (" sql-retry" if r.sql_valid is False else "") + ("" if r.faithful else " UNFAITHFUL") + \
                (f" ERROR {r.error}" if r.error else "") + \
                ("" if r.tool_ok else f" tool={r.tools or '-'} (want {r.tool})")
        console.print(f"[{'green' if r.passed else 'red'}]{'PASS' if r.passed else 'FAIL'}[/] "
                      f"{r.id} {r.type:12} {r.seconds:3.0f}s{pr}{flags}  {r.question}")
        if not real and not r.passed:
            console.print(f"      [dim]{r.sql}\n      {r.answer}[/]")

    conns = [connect()]

    def reconnect():
        console.print("[yellow]connection dropped; reconnecting[/]")
        conns.append(connect())
        return conns[-1]

    try:
        assistant = None
        if system.startswith("llamaindex"):
            from ragplay_llamaindex.app import LlamaIndexAssistant
            assistant = LlamaIndexAssistant(db.SCHEMA if real else db.SAMPLE_SCHEMA,
                                            tuned=system == "llamaindex-tuned")
        results = evaluate.evaluate(conns[0], questions, model=model, system=system,
                                    progress=show, reconnect=reconnect, assistant=assistant)
    finally:
        for c in conns:
            c.close()
    s = evaluate.summary(results)
    path = evaluate.save(set_name, system, model, results)
    table = Table("type", "n", "passed", "precision", "recall")
    for t, v in s["by_type"].items():
        f = lambda x: "" if x is None else f"{x:.2f}"
        table.add_row(t, str(v["n"]), str(v["passed"]), f(v["precision"]), f(v["recall"]))
    console.print(table)
    console.print(f"[bold]{s['passed']}/{s['n']} passed[/] ({system}); SQL valid first try "
                  f"{s['sql_valid'] or 0:.0%}; tool accuracy {s['tool_accuracy']:.0%}; "
                  f"faithful {s['faithful']:.0%}; {s['retried']} retried; {s['crashed']} crashed; "
                  f"{s['seconds']}s.  By expected tool: "
                  + ", ".join(f"{t} {v['passed']}/{v['n']}" for t, v in s["by_tool"].items()))
    console.print(f"[dim]saved {path.relative_to(evaluate.ROOT)}[/]")


categories_app = typer.Typer(no_args_is_help=True, help="Inspect and correct merchant categories.")
app.add_typer(categories_app, name="categories")


def _merchants(conn, where: str = "", params: tuple = ()) -> list[tuple]:
    return conn.execute(
        f"""SELECT m.merchant, m.category, m.source, coalesce(m.description, ''),
                   count(t.id), coalesce(sum(t.amount), 0)
            FROM merchant_categories m LEFT JOIN transactions t ON t.merchant = m.merchant
            {where} GROUP BY 1, 2, 3, 4 ORDER BY count(t.id) DESC, 1""", params).fetchall()


@categories_app.command("summary")
def categories_summary(schema: str = typer.Option(db.SCHEMA)) -> None:
    """Transactions, merchants and spend per category."""
    with db.connect_reader() if schema == db.SCHEMA else db.connect_owner(schema) as conn:
        rows = conn.execute(
            """SELECT coalesce(category, '(none)'), count(*), count(DISTINCT merchant), sum(amount)
               FROM transactions GROUP BY 1 ORDER BY 4 DESC""").fetchall()
    table = Table("category", "txns", "merchants", "net amount")
    for cat, n, m, amt in rows:
        table.add_row(cat, str(n), str(m), f"{amt:,.2f}")
    console.print(table)


@categories_app.command("list")
def categories_list(
    category: str = typer.Option(None, help="Only this category."),
    source: str = typer.Option(None, help="Only rule | llm | manual."),
    schema: str = typer.Option(db.SCHEMA),
) -> None:
    """Merchants with category, source, description and transaction count."""
    conds, params = [], []
    if category:
        conds.append("m.category = %s"), params.append(category)
    if source:
        conds.append("m.source = %s"), params.append(source)
    where = "WHERE " + " AND ".join(conds) if conds else ""
    with db.connect_owner(schema) as conn:
        rows = _merchants(conn, where, tuple(params))
    table = Table("merchant", "category", "source", "description", "txns", "net amount")
    for name, cat, src, desc, n, amt in rows:
        table.add_row(name, cat, src, desc, str(n), f"{amt:,.2f}")
    console.print(table)


@categories_app.command("set")
def categories_set(merchant: str, category: str,
                   description: str = typer.Option(None),
                   schema: str = typer.Option(db.SCHEMA)) -> None:
    """Set a merchant's category by hand (source=manual always wins)."""
    with db.connect_owner(schema) as conn:
        conn.autocommit = True
        categorize.set_category(conn, merchant, category, description)
    console.print(f"{merchant} -> {category} (manual)")


@categories_app.command("review")
def categories_review(
    limit: int = typer.Option(50, help="How many merchants to review."),
    source: str = typer.Option("llm", help="Review merchants from this source."),
    schema: str = typer.Option(db.SCHEMA),
) -> None:
    """Walk through merchants (most transactions first). Enter keeps; a category name changes it;
    'q' stops. Confirmed rows become source=manual so later runs never change them."""
    console.print("Categories: " + ", ".join(categorize.CATEGORIES))
    with db.connect_owner(schema) as conn:
        conn.autocommit = True
        rows = _merchants(conn, "WHERE m.source = %s", (source,))[:limit]
        changed = kept = 0
        for i, (name, cat, _, desc, n, amt) in enumerate(rows, 1):
            examples = [r[0] for r in conn.execute(
                "SELECT DISTINCT description_raw FROM transactions WHERE merchant = %s LIMIT 3",
                (name,))]
            console.print(f"\n[bold]{i}/{len(rows)} {name}[/]  [cyan]{cat}[/]  {desc}"
                          f"  ({n} txns, {amt:,.2f})\n  raw: " + " | ".join(examples))
            ans = typer.prompt("category", default="", show_default=False).strip().lower()
            if ans == "q":
                break
            try:
                categorize.set_category(conn, name, ans or cat)
            except ValueError as e:
                console.print(f"[red]{e}[/] (left unchanged)")
                continue
            changed += bool(ans and ans != cat)
            kept += not (ans and ans != cat)
    console.print(f"\n{kept} confirmed, {changed} changed (all now manual).")


if __name__ == "__main__":
    app()
