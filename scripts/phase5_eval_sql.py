"""Phase 5 eval: text-to-SQL accuracy.

--schema ragplay_test (default): synthetic data; truth computed from the generator's JSON
    sidecars (independent of SQL). Includes the 5 Phase 0 baseline questions.
--schema ragplay: real data; truth = reference SQL written by hand. Prints pass/fail only,
    never amounts or merchants.

A question passes if every truth value appears in the first row of the final query result
(numbers within $0.01, sign-insensitive; months compared as YYYY-MM). Also reports retries and
numbers in the phrased answer that aren't in the result ("unsupported").

Usage: python scripts/phase5_eval_sql.py [--schema ragplay_test|ragplay] [--think]
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from ragplay import db, text2sql

TODAY = date(2026, 11, 13)


def synthetic_cases(fake: Path) -> list[tuple[str, list]]:
    tx = [t for js in sorted(fake.glob("*.json")) for t in json.loads(js.read_text())["transactions"]]
    stmt = json.loads((fake / "acme_4821_2026-03.json").read_text())["transactions"]
    spend = lambda rows, cats=None: sum(
        (Decimal(t["amount"]) for t in rows if t["type"] in ("purchase", "refund")
         and (cats is None or t["category"] in cats)), Decimal(0))
    by_month = defaultdict(Decimal)
    for t in tx:
        if t["type"] in ("purchase", "refund"):
            by_month[t["txn_date"][:7]] += Decimal(t["amount"])
    gas = [Decimal(t["amount"]) for t in tx if t["type"] == "purchase" and t["category"] == "gas"]
    q4 = [t for t in tx if "2025-10-01" <= t["txn_date"] <= "2025-12-31"]
    return [
        # Phase 0 baseline questions, scoped to the same statement
        ("How much did I spend on groceries on my March 2026 statement?", [spend(stmt, {"groceries"})]),
        ("How much did I spend on dining, i.e. restaurants, food delivery and coffee shops, on my March 2026 statement?",
         [spend(stmt, {"dining", "coffee"})]),
        ("How many separate coffee shop purchases are on my March 2026 statement?",
         [sum(1 for t in stmt if t["category"] == "coffee" and t["type"] == "purchase")]),
        ("How much did I spend at gas stations on my March 2026 statement?", [spend(stmt, {"gas"})]),
        ("What was the single largest purchase on my March 2026 statement?",
         [max(Decimal(t["amount"]) for t in stmt if t["type"] == "purchase")]),
        # whole history
        ("Which month had the highest total spending?", [max(by_month, key=by_month.get)]),
        ("How much have I spent on subscriptions in total?", [spend(tx, {"subscriptions"})]),
        ("How many times was I charged by Netflix?",
         [sum(1 for t in tx if t["description"].startswith("NETFLIX") and t["type"] == "purchase")]),
        ("What is my average gas purchase amount?", [(sum(gas) / len(gas)).quantize(Decimal("0.01"))]),
        ("What is the total of all refunds I received?",
         [sum((Decimal(t["amount"]) for t in tx if t["type"] == "refund"), Decimal(0))]),
        ("How much did I spend at Trader Joe's in Q4 2025?",
         [spend([t for t in q4 if t["description"].startswith("TRADER JOE")])]),
        ("How much did I spend in 2019?", [None]),
    ]


# Real data: reference SQL (single row). Questions contain no private data.
REAL = [
    ("How much did I spend on dining in Q2 2026?",
     "SELECT SUM(amount) FROM transactions WHERE category = 'dining' AND type IN ('purchase','refund')"
     " AND txn_date >= '2026-04-01' AND txn_date < '2026-07-01'"),
    ("Which month had the highest total spending?",
     "SELECT to_char(date_trunc('month', txn_date), 'YYYY-MM') FROM transactions"
     " WHERE type IN ('purchase','refund') GROUP BY 1 ORDER BY SUM(amount) DESC LIMIT 1"),
    ("How many coffee purchases have I made in total?",
     "SELECT count(*) FROM transactions WHERE category = 'coffee' AND type = 'purchase'"),
    ("How much have I spent at McDonald's in total?",
     "SELECT SUM(amount) FROM transactions WHERE merchant ILIKE '%mcdonald%' AND type IN ('purchase','refund')"),
    ("Which category did I spend the most on in 2025?",
     "SELECT category FROM transactions WHERE type IN ('purchase','refund') AND txn_date >= '2025-01-01'"
     " AND txn_date < '2026-01-01' GROUP BY 1 ORDER BY SUM(amount) DESC LIMIT 1"),
    ("How many times was I charged by Rogers?",
     "SELECT count(*) FROM transactions WHERE merchant ILIKE '%rogers%' AND type = 'purchase'"),
    ("What is my average gas purchase amount?",
     "SELECT ROUND(AVG(amount), 2) FROM transactions WHERE category = 'gas' AND type = 'purchase'"),
    ("What was my single largest purchase?",
     "SELECT MAX(amount) FROM transactions WHERE type = 'purchase'"),
    ("How much did I spend last month?",
     "SELECT SUM(amount) FROM transactions WHERE type IN ('purchase','refund')"
     " AND txn_date >= '2026-10-01' AND txn_date < '2026-11-01'"),
    ("What was the closing balance on my September 2026 statement?",
     "SELECT closing_balance FROM statements WHERE period_end >= '2026-09-01' AND period_end < '2026-10-01'"),
    ("How much did I spend in 2019?",
     "SELECT SUM(amount) FROM transactions WHERE txn_date >= '2019-01-01' AND txn_date < '2020-01-01'"),
]


def norm(v):
    if v is None:
        return None
    if isinstance(v, (date, datetime)):
        return v.strftime("%Y-%m")
    if isinstance(v, (int, float, Decimal)):
        return abs(Decimal(str(v))).quantize(Decimal("0.01"))
    s = str(v).strip()
    m = re.search(r"(\d{4})-(\d{2})", s)
    return m.group(0) if m else s.lower()


def passes(truth: list, final) -> bool:
    if truth == [None]:  # no data: empty, NULL or zero is right
        return final is not None and (not final.rows or all(v in (None, 0) for v in final.rows[0]))
    if final is None or not final.rows:
        return False
    got = {norm(v) for v in final.rows[0]}
    return all(norm(t) in got or (isinstance(norm(t), Decimal) and any(
        isinstance(g, Decimal) and abs(g - norm(t)) <= Decimal("0.01") for g in got)) for t in truth)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--schema", default=db.TEST_SCHEMA)
    ap.add_argument("--fake", type=Path, default=Path("data/fake"))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--model", default=text2sql.MODEL)
    args = ap.parse_args()
    real = args.schema == db.SCHEMA
    connect = db.connect_reader if real else (lambda: db.connect_owner(args.schema))
    ok = retried = unsupported = 0
    secs = 0.0
    with connect() as conn:
        if real:
            cases = []
            for q, ref in REAL:
                _, rows = text2sql.run_sql(conn, ref)
                cases.append((q, list(rows[0]) if rows and rows[0][0] is not None else [None]))
        else:
            cases = synthetic_cases(args.fake)
        for i, (q, truth) in enumerate(cases, 1):
            ans = text2sql.ask(conn, q, model=args.model, today=TODAY, think=args.think)
            good = passes(truth, ans.final)
            ok += good
            retried += len(ans.attempts) > 1
            unsupported += bool(ans.unsupported)
            secs += ans.seconds
            # never print values for real data
            detail = "" if real or good else f"  truth={truth} got={ans.final.rows[:1] if ans.final else None}"
            print(f"[{'OK ' if good else 'BAD'}] {i:2}. {q}  ({len(ans.attempts)} attempt(s), "
                  f"{ans.seconds:.0f}s{', UNSUPPORTED NUMBERS' if ans.unsupported else ''}){detail}", flush=True)
    print(f"\n{ok}/{len(cases)} correct; {retried} needed a retry; {unsupported} answers had "
          f"numbers not in the result; {secs:.0f}s total (think={'on' if args.think else 'off'})")


if __name__ == "__main__":
    main()
