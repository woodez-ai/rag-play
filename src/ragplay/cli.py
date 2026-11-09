"""`rag` command line. Later phases add ingest | ask | chat | eval."""

import typer
from rich.console import Console
from rich.table import Table

from ragplay import db

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


if __name__ == "__main__":
    app()
