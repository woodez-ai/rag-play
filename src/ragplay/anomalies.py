"""Anomalies and subscriptions, computed deterministically. The LLM only explains the results.

detect_recurring  regular charges from the same merchant (about monthly or yearly), with price
                  changes, whether the series is still active, and the next expected date
find_duplicates   the same merchant and amount within N days
spend_outliers    purchases far above normal for their category and among your largest overall
monthly_review    facts for one month (totals, category changes, new merchants and
                  subscriptions, price changes, duplicates, outliers) for the LLM to narrate
"""

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

import psycopg

from ragplay import budgets

# (name, min gap, max gap, min charges): a charge joins a series if it comes within the gap
# window after the series' last charge at a similar amount. Monthly bills land on a stable day,
# so the window is narrow (weekends/posting shift it a couple of days).
CADENCES = [("monthly", 26, 35, 3), ("yearly", 330, 400, 2)]
AMOUNT_TOL = Decimal("0.05")       # FX-billed subscriptions drift a little month to month
BUSY_TOL = Decimal("0.01")         # busy merchants: only near-exact repeats (e.g. a fixed fee)
BUSY_PER_MONTH = 1.5               # more charges than this per month = a busy merchant
YEARLY_MAX_CHARGES = 4             # yearly series only at merchants with few charges overall
MAX_CHANGES = 2                    # real price changes are rare; more = coincidence
BUSY_GAP_SPREAD = 5                # busy merchants: bills land on a stable day of the month
CHANGE_RATIO = (Decimal("0.8"), Decimal("1.6"))
CATEGORY_MULTIPLE = 3              # outlier: > 3x the category median ...
OVERALL_PCT = 98                   # ... and above the 98th percentile of all purchases


@dataclass
class Charge:
    id: int
    txn_date: date
    merchant: str
    amount: Decimal
    category: str | None


@dataclass
class PriceChange:
    on: date
    old: Decimal
    new: Decimal


@dataclass
class Series:
    merchant: str
    category: str | None
    cadence: str
    charges: list[Charge] = field(default_factory=list)
    changes: list[PriceChange] = field(default_factory=list)
    active: bool = False
    next_expected: date | None = None

    @property
    def amount(self) -> Decimal:
        return self.charges[-1].amount

    @property
    def first(self) -> date:
        return self.charges[0].txn_date

    @property
    def last(self) -> date:
        return self.charges[-1].txn_date

    @property
    def increased(self) -> bool:
        return any(c.new > c.old for c in self.changes)


def _purchases(conn: psycopg.Connection) -> list[Charge]:
    return [Charge(*r) for r in conn.execute(
        """SELECT id, txn_date, coalesce(merchant, description_raw), amount, category
           FROM transactions WHERE type = 'purchase' ORDER BY txn_date, id""")]


def _close(a: Decimal, b: Decimal, tol: Decimal) -> bool:
    return abs(a - b) <= tol * max(a, b)


def _chains(charges: list[Charge], lo: int, hi: int, tol: Decimal) -> list[list[Charge]]:
    """Greedy: each charge extends the open chain whose last charge is lo..hi days earlier at
    a similar amount (closest gap to the middle wins), else starts a new chain."""
    chains: list[list[Charge]] = []
    mid = (lo + hi) / 2
    for c in charges:
        fits = [ch for ch in chains if lo <= (c.txn_date - ch[-1].txn_date).days <= hi
                and _close(c.amount, ch[-1].amount, tol)]
        if fits:
            min(fits, key=lambda ch: abs((c.txn_date - ch[-1].txn_date).days - mid)).append(c)
        else:
            chains.append([c])
    return chains


def detect_recurring(conn: psycopg.Connection, *, category: str | None = None,
                     as_of: date | None = None) -> list[Series]:
    """Recurring series per merchant. A price change splits a chain (amount moves more than the
    tolerance); a following chain that starts one cadence after the previous one ends, at
    0.8-1.6x the price, is merged back in and recorded as a PriceChange. It needs 2+ charges
    unless it holds the merchant's latest charge (an increase that has just happened).

    Guards against coincidence, tuned on synthetic data where the truth is known: busy merchants
    (groceries, coffee) need near-exact amounts; yearly series only at quiet merchants; at most
    two price changes per series."""
    charges = _purchases(conn)
    as_of = as_of or max((c.txn_date for c in charges), default=date.today())
    by_merchant: dict[str, list[Charge]] = {}
    for c in charges:
        by_merchant.setdefault(c.merchant, []).append(c)
    found: list[Series] = []
    for merchant, cs in by_merchant.items():
        used: set[int] = set()
        months = max(1, ((cs[-1].txn_date - cs[0].txn_date).days + 1) / 30.4)
        busy = len(cs) / months > BUSY_PER_MONTH
        tol = BUSY_TOL if busy else AMOUNT_TOL
        for cadence, lo, hi, need in CADENCES:
            if cadence == "yearly" and len(cs) > YEARLY_MAX_CHARGES:
                continue
            pool = [c for c in cs if c.id not in used]
            chains = sorted(_chains(pool, lo, hi, tol), key=lambda ch: ch[0].txn_date)
            for chain in [ch for ch in chains if len(ch) >= need]:
                if any(c.id in used for c in chain):
                    continue
                s = Series(merchant, chain[0].category, cadence, list(chain))
                # merge follow-on chains (price changes), each starting one cadence later
                extended = True
                while extended and len(s.changes) < MAX_CHANGES:
                    extended = False
                    for nxt in chains:
                        gap = (nxt[0].txn_date - s.last).days
                        if (nxt is not chain and lo <= gap <= hi and not any(c.id in used for c in nxt)
                                and nxt[0].id not in {c.id for c in s.charges}
                                and (len(nxt) >= 2 or nxt[-1] is cs[-1])
                                and CHANGE_RATIO[0] <= nxt[0].amount / s.amount <= CHANGE_RATIO[1]):
                            s.changes.append(PriceChange(nxt[0].txn_date, s.amount, nxt[0].amount))
                            s.charges += nxt
                            extended = True
                            break
                gaps = [(b.txn_date - a.txn_date).days for a, b in zip(s.charges, s.charges[1:])]
                if busy and cadence == "monthly" and max(gaps) - min(gaps) > BUSY_GAP_SPREAD:
                    continue  # same coffee at the same price, but not on a billing schedule
                used |= {c.id for c in s.charges}
                typical = statistics.median(gaps)
                s.next_expected = s.last + timedelta(days=round(typical))
                s.active = (as_of - s.last).days <= typical + 10
                found.append(s)
    if category:
        found = [s for s in found if s.category == category]
    return sorted(found, key=lambda s: (not s.active, s.merchant))


def find_duplicates(conn: psycopg.Connection, *, days: int = 0, date_from: date | None = None,
                    date_to: date | None = None) -> list[tuple]:
    """Pairs of purchases: same merchant, same amount, at most `days` apart (default same day:
    on real data, 1 pair same-day vs 11 within a day and 30 within 3 days, mostly the same
    coffee bought on consecutive days).
    Returns rows (id, txn_date, merchant, amount, duplicate_of_id, days_apart)."""
    return conn.execute(
        """SELECT b.id, b.txn_date, b.merchant, b.amount, a.id, b.txn_date - a.txn_date
           FROM transactions a JOIN transactions b
             ON a.merchant = b.merchant AND a.amount = b.amount AND a.id < b.id
            AND abs(b.txn_date - a.txn_date) <= %(days)s
           WHERE a.type = 'purchase' AND b.type = 'purchase'
             AND (%(f)s::date IS NULL OR b.txn_date >= %(f)s)
             AND (%(t)s::date IS NULL OR b.txn_date <= %(t)s)
           ORDER BY b.txn_date""", dict(days=days, f=date_from, t=date_to)).fetchall()


@dataclass
class Outlier:
    id: int
    txn_date: date
    merchant: str
    amount: Decimal
    category: str | None
    reason: str


def spend_outliers(conn: psycopg.Connection, *, date_from: date | None = None,
                   date_to: date | None = None) -> list[Outlier]:
    """Purchases more than 3x their category's median AND in the top 2% of all purchases.
    History is the user's own: 'unusual' means unusual for you. Categories need >= 8 purchases.

    Chosen by measuring on real data (891 purchases, 19 months): a robust z-score on raw amounts
    flagged 87, on log amounts 57 (fixed-price and two-mode categories break it); "new merchant,
    big first charge" added ~28 of mostly ordinary travel. This rule flags 17 (~1/month)."""
    charges = _purchases(conn)
    amounts = [c.amount for c in charges]
    if len(amounts) < 50:
        return []
    top2 = statistics.quantiles(amounts, n=100)[OVERALL_PCT - 1]
    by_cat: dict[str | None, list[Decimal]] = {}
    for c in charges:
        by_cat.setdefault(c.category, []).append(c.amount)
    med = {k: statistics.median(v) for k, v in by_cat.items() if len(v) >= 8}
    return [
        Outlier(c.id, c.txn_date, c.merchant, c.amount, c.category,
                f"{c.amount / med[c.category]:.1f}x your usual {c.category} purchase "
                f"({med[c.category]:.2f}) and in your top 2% of purchases")
        for c in charges
        if c.category in med and c.amount > CATEGORY_MULTIPLE * med[c.category] and c.amount > top2
        and not (date_from and c.txn_date < date_from) and not (date_to and c.txn_date > date_to)
    ]


def recurring_rows(series: list[Series]) -> tuple[list[str], list[tuple]]:
    cols = ["merchant", "category", "cadence", "amount", "charges", "first", "last", "active",
            "price_change", "next_expected"]
    rows = [(s.merchant, s.category, s.cadence, s.amount, len(s.charges), s.first, s.last,
             # explicit "no change", not a blank: the LLM read a blank cell as an increase
             s.active, "; ".join(f"{c.old} -> {c.new} on {c.on}" for c in s.changes) or "no change",
             s.next_expected if s.active else None) for s in series]
    return cols, rows


def duplicate_rows(pairs: list[tuple], conn: psycopg.Connection) -> tuple[list[str], list[tuple]]:
    """One row per transaction in a duplicate pair (both charges are the evidence)."""
    ids = sorted({p[0] for p in pairs} | {p[4] for p in pairs})
    rows = conn.execute(
        "SELECT id, txn_date, merchant, amount FROM transactions WHERE id = ANY(%s) ORDER BY txn_date, id",
        (ids,)).fetchall() if ids else []
    return ["id", "txn_date", "merchant", "amount"], rows


def outlier_rows(outliers: list[Outlier]) -> tuple[list[str], list[tuple]]:
    return (["id", "txn_date", "merchant", "amount", "category", "why"],
            [(o.id, o.txn_date, o.merchant, o.amount, o.category, o.reason) for o in outliers])


def _month_bounds(month: str) -> tuple[date, date]:
    y, m = map(int, month.split("-"))
    start = date(y, m, 1)
    end = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return start, end


def monthly_review(conn: psycopg.Connection, month: str | None = None) -> dict:
    """Facts for one calendar month (default: the last full month with data). Everything is
    computed here; the LLM only turns these facts into prose."""
    last = conn.execute("SELECT max(txn_date) FROM transactions").fetchone()[0]
    if month is None:
        prev = last.replace(day=1) - timedelta(days=1)
        month = f"{prev:%Y-%m}" if last < _month_bounds(f"{last:%Y-%m}")[1] else f"{last:%Y-%m}"
    start, end = _month_bounds(month)
    spend = """SELECT coalesce(sum(amount), 0) FROM transactions
               WHERE type IN ('purchase','refund') AND txn_date BETWEEN %s AND %s"""
    total = conn.execute(spend, (start, end)).fetchone()[0]
    prior = []
    s = start
    for _ in range(3):
        pe = s - timedelta(days=1)
        s = pe.replace(day=1)
        prior.append(conn.execute(spend, (s, pe)).fetchone()[0])
    avg3 = (sum(prior) / len(prior)).quantize(Decimal("0.01")) if prior else None
    cats = conn.execute(
        """WITH m AS (SELECT category, sum(amount) AS amt FROM transactions
                      WHERE type IN ('purchase','refund') AND txn_date BETWEEN %(s)s AND %(e)s GROUP BY 1),
                p AS (SELECT category, round(sum(amount) / 3, 2) AS amt FROM transactions
                      WHERE type IN ('purchase','refund') AND txn_date >= %(ps)s AND txn_date < %(s)s GROUP BY 1)
           SELECT coalesce(m.category, p.category), coalesce(m.amt, 0), coalesce(p.amt, 0),
                  coalesce(m.amt, 0) - coalesce(p.amt, 0) AS diff
           FROM m FULL JOIN p USING (category) ORDER BY abs(coalesce(m.amt, 0) - coalesce(p.amt, 0)) DESC LIMIT 5""",
        dict(s=start, e=end, ps=s)).fetchall()
    new_merchants = conn.execute(
        """SELECT merchant, min(txn_date), sum(amount) FROM transactions t
           WHERE type = 'purchase' AND txn_date BETWEEN %s AND %s
             AND NOT EXISTS (SELECT 1 FROM transactions o WHERE o.merchant = t.merchant
                             AND o.type = 'purchase' AND o.txn_date < %s)
           GROUP BY 1 ORDER BY 3 DESC""", (start, end, start)).fetchall()
    series = detect_recurring(conn, as_of=end)
    return {
        "month": month,
        "total_spend": total,
        "previous_3_months_average": avg3,
        "category_changes_vs_3_month_average": [
            {"category": c, "this_month": a, "usual": u, "difference": d} for c, a, u, d in cats],
        "new_merchants": [{"merchant": m, "first_charge": d, "spent": a} for m, d, a in new_merchants[:8]],
        "new_recurring_charges": [
            {"merchant": s.merchant, "cadence": s.cadence, "amount": s.amount, "since": s.first}
            for s in series if start <= s.charges[min(1, len(s.charges) - 1)].txn_date <= end
            and s.first >= start - timedelta(days=40)],
        "price_changes": [
            {"merchant": s.merchant, "on": ch.on, "old": ch.old, "new": ch.new}
            for s in series for ch in s.changes if start <= ch.on <= end],
        "recurring_charges_missing_this_month": [
            {"merchant": s.merchant, "last_charged": s.last, "amount": s.amount}
            for s in series if s.cadence == "monthly" and s.next_expected
            and start <= s.next_expected <= end and s.last < start
            and not any(start <= c.txn_date <= end for c in s.charges)],
        "possible_duplicates": [
            {"merchant": r[2], "date": r[1], "amount": r[3]} for r in find_duplicates(conn, date_from=start, date_to=end)],
        "unusual_purchases": [
            {"merchant": o.merchant, "date": o.txn_date, "amount": o.amount, "why": o.reason}
            for o in spend_outliers(conn, date_from=start, date_to=end)],
        "budgets": [
            {"category": b.category, "budget": b.budget, "spent": b.spent, "remaining": b.remaining,
             "status": b.status} for b in budgets.status(conn, month, today=end)[1]],
    }
