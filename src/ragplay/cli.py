"""`rag` command line. Later phases add ingest | ask | chat | eval."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragplay import categorize, db, ingest, store, text2sql

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
    show_sql: bool = typer.Option(False, "--sql", help="Show the generated SQL and attempts."),
    think: bool = typer.Option(False, help="Let qwen3 think before writing SQL (slower)."),
    model: str = typer.Option(text2sql.MODEL),
    schema: str = typer.Option(db.SCHEMA),
) -> None:
    """Answer a question with LLM-written SQL (read-only; the database does the maths)."""
    with db.connect_reader() if schema == db.SCHEMA else db.connect_owner(schema) as conn:
        with console.status("thinking..."):
            ans = text2sql.ask(conn, question, model=model, think=think)
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
