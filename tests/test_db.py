"""Integration tests against the remote Postgres (needs .env and LAN access)."""

import psycopg
import pytest

from ragplay import db


def test_reader_can_select_app_tables():
    with db.connect_reader() as conn:
        assert conn.execute("SELECT count(*) FROM transactions").fetchone()[0] >= 0


@pytest.mark.parametrize("sql", [
    "INSERT INTO merchant_categories (merchant, category, source) VALUES ('x', 'y', 'manual')",
    "UPDATE transactions SET amount = 0",
    "DELETE FROM statements",
    "DROP TABLE transactions",
    "CREATE TABLE ragplay.evil (x int)",
])
def test_reader_cannot_write(sql):
    with db.connect_reader() as conn, pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        conn.execute(sql)


def test_reader_cannot_escape_read_only():
    """Generated SQL could try to switch the transaction back to read-write."""
    with db.connect_reader() as conn:
        with pytest.raises(psycopg.Error):
            conn.execute("SET TRANSACTION READ WRITE; DELETE FROM statements")


def test_reader_statement_timeout():
    with db.connect_reader() as conn, pytest.raises(psycopg.errors.QueryCanceled):
        conn.execute("SELECT pg_sleep(%s)", (db.READER_TIMEOUT_MS / 1000 + 1,))


def test_connections_use_tls():
    for connect in (db.connect_owner, db.connect_reader):
        with connect() as conn:
            assert conn.execute(
                "SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()"
            ).fetchone()[0]


def test_pgvector_available_to_reader():
    with db.connect_reader() as conn:
        d = conn.execute("SELECT '[1,0]'::vector <=> '[0,1]'::vector").fetchone()[0]
        assert d == pytest.approx(1.0)
