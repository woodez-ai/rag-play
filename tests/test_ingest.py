"""Ingest into the ragplay_test schema (synthetic data only; needs .env and LAN access)."""

import subprocess
import sys
from pathlib import Path

import pytest

from ragplay import db, ingest
from ragplay.parsers import acme

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pdfs(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake")
    subprocess.run(
        [sys.executable, ROOT / "scripts/make_fake_statements.py", "--months", "2",
         "--end", "2026-01", "--out", out],
        check=True, capture_output=True,
    )
    return sorted(out.glob("*.pdf"))


@pytest.fixture
def conn():
    db.migrate(db.TEST_SCHEMA)
    c = db.connect_owner(db.TEST_SCHEMA)
    c.autocommit = True
    assert c.execute("SELECT current_schema()").fetchone()[0] == db.TEST_SCHEMA
    c.execute("TRUNCATE statements CASCADE")
    yield c
    c.execute("TRUNCATE statements CASCADE")
    c.close()


def test_ingest_loads_and_is_idempotent(conn, pdfs):
    first = [ingest.ingest_file(conn, p) for p in pdfs]
    assert [r.status for r in first] == ["loaded"] * len(pdfs)
    n = conn.execute("SELECT count(*) FROM transactions").fetchone()[0]
    assert n == sum(r.transactions for r in first)

    again = [ingest.ingest_file(conn, p) for p in pdfs]
    assert [r.status for r in again] == ["skipped"] * len(pdfs)
    assert conn.execute("SELECT count(*) FROM transactions").fetchone()[0] == n


def test_db_totals_match_statement(conn, pdfs):
    for p in pdfs:
        ingest.ingest_file(conn, p)
    rows = conn.execute("""
        SELECT s.id, s.previous_balance + sum(t.amount) = s.closing_balance,
               sum(t.amount) FILTER (WHERE t.type = 'purchase') = s.total_purchases
        FROM statements s JOIN transactions t ON t.statement_id = s.id GROUP BY s.id""").fetchall()
    assert rows and all(ok1 and ok2 for _, ok1, ok2 in rows)


def test_failed_reconcile_inserts_nothing(conn, pdfs, monkeypatch):
    real_parse = acme.parse

    def drop_a_row(pages):
        s = real_parse(pages)
        s.transactions.pop()
        return s

    monkeypatch.setattr(acme, "parse", drop_a_row)
    r = ingest.ingest_file(conn, pdfs[0])
    assert r.status == "failed" and r.errors
    assert conn.execute("SELECT count(*) FROM statements").fetchone()[0] == 0


def test_insert_error_rolls_back_whole_statement(conn, pdfs, monkeypatch):
    """A failure after the statement row is written must leave no partial statement."""
    real_insert = ingest.insert_statement

    def fail_midway(c, *a):
        real_insert(c, *a)
        raise RuntimeError("boom")

    monkeypatch.setattr(ingest, "insert_statement", fail_midway)
    with pytest.raises(RuntimeError):
        ingest.ingest_file(conn, pdfs[0])
    assert conn.execute("SELECT count(*) FROM statements").fetchone()[0] == 0
