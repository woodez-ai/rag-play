"""`rag` command line. Later phases add ingest | ask | chat | eval."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragplay import categorize, db, ingest

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
