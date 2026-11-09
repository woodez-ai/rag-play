"""PostgreSQL connections (owner + read-only) and schema migrations.

Connection details come from `.env` (gitignored). Never log the conninfo: it contains the password.
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = ROOT / "sql"
SCHEMA = "ragplay"
READER_TIMEOUT_MS = 5000


def _conninfo(user_var: str, password_var: str, options: str = "") -> str:
    load_dotenv(ROOT / ".env")
    missing = [v for v in ("DB_HOST", "DB", user_var, password_var) if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
    return psycopg.conninfo.make_conninfo(
        host=os.environ["DB_HOST"],
        dbname=os.environ["DB"],
        user=os.environ[user_var],
        password=os.environ[password_var],
        sslmode="require",
        connect_timeout=10,
        options=f"-c search_path={SCHEMA},public {options}".strip(),
    )


def connect_owner() -> psycopg.Connection:
    """Owner connection (rag_user): ingest and migrations."""
    conn = psycopg.connect(_conninfo("DB_USER", "DB_PASSWORD"))
    register_vector(conn)
    return conn


def connect_reader() -> psycopg.Connection:
    """Read-only connection (rag_reader) for LLM-generated SQL.

    The role is already read-only with a timeout on the server; the client sets the same
    limits again so a misconfigured role can't silently drop them.
    """
    conn = psycopg.connect(
        _conninfo(
            "DB_READER_USER",
            "DB_READER_PASSWORD",
            f"-c default_transaction_read_only=on -c statement_timeout={READER_TIMEOUT_MS}",
        )
    )
    register_vector(conn)
    return conn


def migrate() -> list[str]:
    """Apply sql/NNN_*.sql files not yet recorded in schema_migrations. Returns applied names."""
    applied = []
    with connect_owner() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        conn.commit()
        done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(SQL_DIR.glob("[0-9][0-9][0-9]_*.sql")):
            if path.name in done:
                continue
            with conn.transaction():  # each migration is all-or-nothing
                conn.execute(path.read_text())
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,))
            applied.append(path.name)
    return applied


def check() -> list[tuple[str, str, str]]:
    """Connectivity/safety report: (check, role, result) rows. Raises on connection failure."""
    rows = []
    for role, connect in (("owner", connect_owner), ("reader", connect_reader)):
        with connect() as conn:
            user, ro, timeout = conn.execute(
                "SELECT current_user, current_setting('transaction_read_only'),"
                " current_setting('statement_timeout')"
            ).fetchone()
            ssl = conn.execute(
                "SELECT ssl, version FROM pg_stat_ssl WHERE pid = pg_backend_pid()"
            ).fetchone()
            tables = conn.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema = %s",
                (SCHEMA,),
            ).fetchone()[0]
            exts = conn.execute(
                "SELECT string_agg(extname || ' ' || extversion, ', ' ORDER BY extname)"
                " FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')"
            ).fetchone()[0]
            rows += [
                ("user", role, user),
                ("tls", role, f"{ssl[0]} {ssl[1]}"),
                ("read-only", role, ro),
                ("statement_timeout", role, timeout),
                (f"tables in {SCHEMA}", role, str(tables)),
                ("extensions", role, exts or "MISSING"),
            ]
    return rows
