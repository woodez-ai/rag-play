"""Anomaly detection against the generator's planted truths (12 synthetic months).

Planted: 5 monthly charges (Netflix, Spotify, Apple, Comcast, Planet Fitness), a Netflix price
increase 15.49 -> 17.99 from month 6, and one duplicate Target charge in month 4.
Everything else is random spending that must NOT look recurring.
"""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from ragplay import anomalies, categorize, db, ingest
from ragplay.categorize import Classification

ROOT = Path(__file__).resolve().parents[1]
PLANTED = {"netflix", "spotify", "apple", "comcast", "planet"}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake")
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_statements.py", "--months", "12",
                    "--end", "2026-09", "--out", out], check=True, capture_output=True)
    truth = [t for js in sorted(out.glob("*.json")) for t in json.loads(js.read_text())["transactions"]]
    cats = {t["description"].split()[0]: t["category"] for t in truth}
    db.migrate(db.TEST_SCHEMA)
    c = db.connect_owner(db.TEST_SCHEMA)
    assert c.execute("SELECT current_schema()").fetchone()[0] == db.TEST_SCHEMA
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories CASCADE")
    for pdf in sorted(out.glob("*.pdf")):
        assert ingest.ingest_file(c, pdf).status == "loaded"
    mp = pytest.MonkeyPatch()
    mp.setattr(categorize, "llm_classify", lambda key, raw, model=None: Classification(
        key.split()[0].split(".")[0].title(), cats.get(raw.split()[0], "other") or "other", "x", "llm"))
    categorize.categorize(c)
    mp.undo()
    yield c, truth
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories CASCADE")
    c.close()


def first_word(name: str) -> str:
    return name.lower().split()[0]


def test_finds_exactly_the_planted_subscriptions(data):
    conn, _ = data
    series = anomalies.detect_recurring(conn)
    assert {first_word(s.merchant) for s in series} == PLANTED
    assert all(s.cadence == "monthly" and s.active and len(s.charges) == 12 for s in series)


def test_netflix_price_increase(data):
    conn, truth = data
    netflix = next(s for s in anomalies.detect_recurring(conn) if first_word(s.merchant) == "netflix")
    first_new = min(t["txn_date"] for t in truth if t["description"].startswith("NETFLIX") and t["amount"] == "17.99")
    assert netflix.increased and len(netflix.changes) == 1
    ch = netflix.changes[0]
    assert (ch.old, ch.new, ch.on.isoformat()) == (Decimal("15.49"), Decimal("17.99"), first_new)
    assert netflix.amount == Decimal("17.99")
    assert not any(s.changes for s in anomalies.detect_recurring(conn) if first_word(s.merchant) != "netflix")


def test_category_filter(data):
    conn, _ = data
    subs = anomalies.detect_recurring(conn, category="subscriptions")
    assert {first_word(s.merchant) for s in subs} == {"netflix", "spotify", "apple"}


def test_planted_duplicate(data):
    conn, truth = data
    pairs = anomalies.find_duplicates(conn)
    assert len(pairs) == 1 and pairs[0][2].lower().startswith("target") and pairs[0][5] == 0
    cols, rows = anomalies.duplicate_rows(pairs, conn)
    assert len(rows) == 2 and rows[0][3] == rows[1][3]


def test_no_outliers_in_bounded_random_spending(data):
    conn, _ = data
    assert anomalies.spend_outliers(conn) == []


def test_review_reports_the_price_change_in_its_month(data):
    conn, truth = data
    first_new = min(t["txn_date"] for t in truth if t["description"].startswith("NETFLIX") and t["amount"] == "17.99")
    facts = anomalies.monthly_review(conn, first_new[:7])
    assert [(first_word(p["merchant"]), p["old"], p["new"]) for p in facts["price_changes"]] == \
        [("netflix", Decimal("15.49"), Decimal("17.99"))]
    other = anomalies.monthly_review(conn, "2026-02")
    assert other["price_changes"] == []


def test_review_month_with_duplicate(data):
    conn, truth = data
    dup_month = anomalies.find_duplicates(conn)[0][1].strftime("%Y-%m")
    assert len(anomalies.monthly_review(conn, dup_month)["possible_duplicates"]) == 1
