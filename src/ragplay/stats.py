"""Dashboard numbers, computed in SQL. Spend = purchases + refunds by txn_date (the same
definition as text-to-SQL and budgets)."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import psycopg

SPEND = "type IN ('purchase', 'refund')"


@dataclass
class MonthSummary:
    month: str
    spent: Decimal
    purchases: int
    data_through: date | None  # last transaction date in the month
    complete: bool             # data covers the whole month


def months(conn: psycopg.Connection) -> list[str]:
    """Months with any spending, newest first."""
    return [r[0] for r in conn.execute(
        f"SELECT DISTINCT to_char(txn_date, 'YYYY-MM') m FROM transactions WHERE {SPEND} ORDER BY m DESC")]


def bounds(month: str) -> tuple[date, date]:
    y, m = map(int, month.split("-"))
    return date(y, m, 1), date(y + (m == 12), m % 12 + 1, 1)


def previous(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y - (m == 1)}-{(m - 2) % 12 + 1:02d}"


def summary(conn: psycopg.Connection, month: str) -> MonthSummary:
    start, nxt = bounds(month)
    spent, n, through = conn.execute(
        f"""SELECT coalesce(sum(amount), 0), count(*) FILTER (WHERE type = 'purchase'), max(txn_date)
            FROM transactions WHERE {SPEND} AND txn_date >= %s AND txn_date < %s""", (start, nxt)).fetchone()
    last_all = conn.execute("SELECT max(txn_date) FROM transactions").fetchone()[0]
    complete = last_all is not None and last_all >= date.fromordinal(nxt.toordinal() - 1)
    return MonthSummary(month, spent, n, through, complete)


def by_category(conn: psycopg.Connection, month: str) -> list[tuple[str, Decimal]]:
    start, nxt = bounds(month)
    return conn.execute(
        f"""SELECT coalesce(category, 'other'), sum(amount) FROM transactions
            WHERE {SPEND} AND txn_date >= %s AND txn_date < %s GROUP BY 1 HAVING sum(amount) > 0
            ORDER BY 2 DESC""", (start, nxt)).fetchall()


def trend(conn: psycopg.Connection, end_month: str, n: int = 12) -> list[tuple[str, Decimal]]:
    """Monthly spend for up to n months ending at end_month, starting no earlier than the first
    month with data (later months without data are 0)."""
    first = conn.execute(f"SELECT to_char(min(txn_date), 'YYYY-MM') FROM transactions WHERE {SPEND}").fetchone()[0]
    ms = [end_month]
    while len(ms) < n and first and ms[-1] > first:
        ms.append(previous(ms[-1]))
    ms.reverse()
    start, _ = bounds(ms[0])
    _, nxt = bounds(end_month)
    got = dict(conn.execute(
        f"""SELECT to_char(txn_date, 'YYYY-MM'), sum(amount) FROM transactions
            WHERE {SPEND} AND txn_date >= %s AND txn_date < %s GROUP BY 1""", (start, nxt)).fetchall())
    return [(m, got.get(m, Decimal(0))) for m in ms]


def categories(conn: psycopg.Connection) -> list[str]:
    """Categories that have transactions, alphabetical."""
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT category FROM transactions WHERE category IS NOT NULL ORDER BY 1")]
