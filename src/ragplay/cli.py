"""`rag` command line. Later phases add ingest | ask | chat | eval."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ragplay import db, ingest

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


if __name__ == "__main__":
    app()
