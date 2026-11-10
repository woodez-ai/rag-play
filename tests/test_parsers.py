"""Parsers against synthetic statements (exact ground truth) + reconcile/unit checks."""

import dataclasses
import json
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ragplay.parsers import classify, infer_year, parse_file
from ragplay.parsers.visa import PERIOD_RE
from ragplay.validate import reconcile

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fake_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake")
    subprocess.run(
        [sys.executable, ROOT / "scripts/make_fake_statements.py", "--months", "4",
         "--end", "2026-01", "--out", out],
        check=True, capture_output=True,
    )
    return out


def pairs(fake_dir):
    return [(p, json.loads(p.with_suffix(".json").read_text())) for p in sorted(fake_dir.glob("*.pdf"))]


def test_acme_matches_ground_truth_exactly(fake_dir):
    for pdf, truth in pairs(fake_dir):
        s = parse_file(pdf)
        got = [(t.txn_date.isoformat(), t.post_date.isoformat(), t.description, t.amount, t.type)
               for t in s.transactions]
        want = [(t["txn_date"], t["post_date"], t["description"], Decimal(t["amount"]), t["type"])
                for t in truth["transactions"]]
        assert got == want, pdf.name
        assert (s.period_start.isoformat(), s.period_end.isoformat()) == (
            truth["period_start"], truth["period_end"])
        assert s.card_last4 == truth["card_last4"]
        assert s.new_balance == Decimal(truth["summary"]["new_balance"])


def test_acme_reconciles(fake_dir):
    for pdf, _ in pairs(fake_dir):
        assert reconcile(parse_file(pdf)) == [], pdf.name


def test_full_card_number_never_kept(fake_dir):
    for pdf, _ in pairs(fake_dir):
        s = parse_file(pdf)
        assert "4111" not in repr(s) and "JANE" not in repr(s)


def test_reconcile_catches_a_wrong_amount(fake_dir):
    s = parse_file(next(fake_dir.glob("*.pdf")))
    s.transactions[3] = dataclasses.replace(s.transactions[3], amount=s.transactions[3].amount + Decimal("0.01"))
    errors = reconcile(s)
    assert any(e.startswith("purchases") for e in errors)
    assert any(e.startswith("balance roll-forward") for e in errors)


def test_reconcile_catches_a_missed_row(fake_dir):
    s = parse_file(next(fake_dir.glob("*.pdf")))
    s.transactions.pop()
    assert reconcile(s)


@pytest.mark.parametrize("month,end,expected", [
    (12, date(2026, 1, 14), date(2025, 12, 20)),
    (1, date(2026, 1, 14), date(2026, 1, 20)),
    (3, date(2026, 3, 14), date(2026, 3, 20)),
])
def test_infer_year_rollover(month, end, expected):
    assert infer_year(month, 20, end) == expected


@pytest.mark.parametrize("desc,amount,kind", [
    ("PAYMENT - THANK YOU", "-100.00", "payment"),
    ("REFUND AMAZON.COM", "-5.00", "refund"),
    ("FOREIGN TRANSACTION FEE", "1.20", "fee"),
    ("ANNUAL FEE", "120.00", "fee"),
    ("PURCHASE INTEREST", "3.10", "interest"),
    ("TICKETS SVC FEE TORONTO ON", "8.00", "purchase"),  # merchant with FEE in its name
    ("PLANET FITNESS CLUB FEES NH", "24.99", "purchase"),
])
def test_classify(desc, amount, kind):
    assert classify(desc, Decimal(amount)) == kind


@pytest.mark.parametrize("header,start_year", [
    ("STATEMENT FROM SEP 22 TO OCT 21, 2026", None),
    ("STATEMENT FROM DEC 22, 2025 TO JAN 21, 2026", "2025"),
])
def test_visa_period_header(header, start_year):
    m = PERIOD_RE.search(header)
    assert m and m[3] == start_year and m[6] == "2026"
