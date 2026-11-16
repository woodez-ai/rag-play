"""Monthly budgets per category ('all' = total spending), checked in code.

Spend = purchases + refunds by txn_date in the calendar month (the same definition as
text-to-SQL). When the data stops before the month ends (a month in progress, or a statement that
closed mid-month), the projection is a straight line: spent so far divided by days covered, times
days in the month. "Covered" ends at the last transaction date in that month, not today.
"""

import calendar
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import psycopg

ALL = "all"
CENT = Decimal("0.01")
AT_RISK = Decimal("0.9")  # projected or actual use >= 90% of the budget


@dataclass
class BudgetStatus:
    category: str
    budget: Decimal
    spent: Decimal
    remaining: Decimal
    pct_used: Decimal
    projected: Decimal | None  # only when the data stops before the month ends
    status: str                # over | at risk | on track
    data_through: date | None = None  # last transaction date in the month


def set_budget(conn: psycopg.Connection, category: str, amount: Decimal) -> None:
    known = {r[0] for r in conn.execute("SELECT DISTINCT category FROM transactions WHERE category IS NOT NULL")}
    if category != ALL and category not in known:
        raise ValueError(f"unknown category {category!r}; use one of: {ALL}, {', '.join(sorted(known))}")
    if amount <= 0:
        raise ValueError("budget must be positive")
    conn.execute(
        """INSERT INTO budgets (category, monthly_amount) VALUES (%s, %s)
           ON CONFLICT (category) DO UPDATE SET monthly_amount = EXCLUDED.monthly_amount, updated_at = now()""",
        (category, amount))


def remove_budget(conn: psycopg.Connection, category: str) -> bool:
    return conn.execute("DELETE FROM budgets WHERE category = %s", (category,)).rowcount > 0


def list_budgets(conn: psycopg.Connection) -> list[tuple[str, Decimal]]:
    return conn.execute("SELECT category, monthly_amount FROM budgets ORDER BY category = 'all' DESC, category").fetchall()


def default_month(conn: psycopg.Connection, today: date) -> str:
    """The current month if it has data, else the latest month that does."""
    last = conn.execute("SELECT max(txn_date) FROM transactions").fetchone()[0]
    cur = f"{today:%Y-%m}"
    return cur if last and f"{last:%Y-%m}" >= cur else f"{last:%Y-%m}"


def status(conn: psycopg.Connection, month: str | None = None, *, today: date | None = None,
           category: str | None = None) -> tuple[str, list[BudgetStatus]]:
    today = today or date.today()
    month = month or default_month(conn, today)
    y, m = map(int, month.split("-"))
    start, days = date(y, m, 1), calendar.monthrange(y, m)[1]
    end = date(y, m, days)
    spent_by_cat = dict(conn.execute(
        """SELECT coalesce(category, 'other'), sum(amount) FROM transactions
           WHERE type IN ('purchase', 'refund') AND txn_date BETWEEN %s AND %s GROUP BY 1""",
        (start, end)).fetchall())
    as_of = conn.execute("SELECT max(txn_date) FROM transactions WHERE txn_date BETWEEN %s AND %s",
                         (start, end)).fetchone()[0]
    # Project whenever the DATA stops before the month ends: a month in progress, or a past month
    # whose last statement closed mid-month (real data ends Oct 21). Testing "today is in the
    # month" called a half-covered October "on track at 47%".
    in_progress = as_of is not None and as_of < end
    out = []
    for cat, budget in list_budgets(conn):
        if category and cat != category:
            continue
        spent = (sum(spent_by_cat.values(), Decimal(0)) if cat == ALL
                 else spent_by_cat.get(cat, Decimal(0))).quantize(CENT, ROUND_HALF_UP)
        projected = ((spent / as_of.day * days).quantize(CENT, ROUND_HALF_UP)
                     if in_progress and as_of.day else None)
        pct = (spent / budget * 100).quantize(Decimal("0.1"), ROUND_HALF_UP)
        state = ("over" if spent > budget else
                 "at risk" if max(spent, projected or 0) >= budget * AT_RISK else "on track")
        out.append(BudgetStatus(cat, budget, spent, (budget - spent).quantize(CENT), pct, projected,
                                state, as_of))
    return month, out


def rows(items: list[BudgetStatus]) -> tuple[list[str], list[tuple]]:
    cols = ["category", "budget", "spent", "remaining", "pct_used", "projected", "status", "data_through"]
    return cols, [(b.category, b.budget, b.spent, b.remaining, b.pct_used, b.projected, b.status,
                   b.data_through) for b in items]
