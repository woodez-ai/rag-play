"""Text-to-SQL guards, parsing, faithfulness check and the retry loop (fake LLM, real DB)."""

from datetime import date
from decimal import Decimal

import psycopg
import pytest

from ragplay import db, text2sql
from ragplay.text2sql import UnsafeSQL, check_sql, extract_sql, unsupported_numbers


@pytest.mark.parametrize("sql", [
    "SELECT 1", "select 1;", "  WITH x AS (SELECT 1) SELECT * FROM x  ",
])
def test_check_sql_allows_single_select(sql):
    assert check_sql(sql).lower().lstrip().startswith(("select", "with"))


@pytest.mark.parametrize("sql", [
    "DELETE FROM statements",
    "SELECT 1; DROP TABLE transactions",
    "UPDATE transactions SET amount = 0",
    "SELECT 1 -- sneaky",
    "SELECT /* x */ 1",
    "SET TRANSACTION READ WRITE",
])
def test_check_sql_refuses(sql):
    with pytest.raises(UnsafeSQL):
        check_sql(sql)


def test_extract_sql_from_code_block():
    assert extract_sql("Here:\n```sql\nSELECT 1\n```\nDone") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


def test_unsupported_numbers():
    rows = [(Decimal("1234.50"), 7)]
    assert unsupported_numbers("You spent $1,234.50 over 7 purchases.", "q", ["t", "n"], rows) == []
    assert unsupported_numbers("You spent $1,300.00.", "q", ["t"], rows) == ["1300.00"]
    assert unsupported_numbers("In 2025 you spent $1,234.50.", "spend in 2025?", ["t"], rows) == []
    # sign-insensitive: refunds stored negative, phrased positive
    assert unsupported_numbers("Refunds: $5.00", "q", ["t"], [(Decimal("-5.00"),)]) == []


def test_run_sql_is_read_only_even_as_owner():
    """run_sql's own READ ONLY transaction holds even on a connection that could write:
    setval() is a write hidden inside a plain SELECT, so it passes check_sql()."""
    with db.connect_owner(db.TEST_SCHEMA) as conn:
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            text2sql.run_sql(conn, "SELECT setval('statements_id_seq', 1)")


def test_writable_cte_is_refused():
    """A DELETE inside WITH passes the SELECT/WITH check; Postgres refuses it (nested by the
    row-cap wrapper, and read-only anyway)."""
    with db.connect_owner(db.TEST_SCHEMA) as conn:
        with pytest.raises(psycopg.Error):
            text2sql.run_sql(conn, "WITH d AS (DELETE FROM statements RETURNING id) SELECT count(*) FROM d")


def test_run_sql_caps_rows():
    with db.connect_owner(db.TEST_SCHEMA) as conn:
        cols, rows = text2sql.run_sql(conn, "SELECT g FROM generate_series(1, 10000) g")
    assert cols == ["g"] and len(rows) == text2sql.ROW_CAP


def test_retry_feeds_error_back(monkeypatch):
    prompts = []
    replies = iter(["```sql\nSELECT no_such_column FROM transactions\n```",
                    "```sql\nSELECT count(*) AS n FROM transactions\n```",
                    "There are some transactions."])

    def fake_chat(prompt, model, schema=None, think=False):
        prompts.append(prompt)
        return next(replies)

    monkeypatch.setattr(text2sql, "_chat", fake_chat)
    with db.connect_owner(db.TEST_SCHEMA) as conn:
        ans = text2sql.ask(conn, "how many?", today=date(2026, 11, 13))
    assert [a.error is None for a in ans.attempts] == [False, True]
    assert "no_such_column" in prompts[1] and "A previous attempt failed" in prompts[1]
    assert ans.final.columns == ["n"]


def test_unsafe_sql_is_refused_not_run(monkeypatch):
    replies = iter(["```sql\nDELETE FROM statements\n```"] * 3 + ["no answer"])
    monkeypatch.setattr(text2sql, "_chat", lambda *a, **k: next(replies))
    with db.connect_owner(db.TEST_SCHEMA) as conn:
        ans = text2sql.ask(conn, "delete everything", today=date(2026, 11, 13))
    assert ans.final is None
    assert all(a.error.startswith("refused") for a in ans.attempts)
    assert len(ans.attempts) == text2sql.MAX_RETRIES + 1
