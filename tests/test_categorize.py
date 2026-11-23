"""Merchant normalisation (unit) and the categorisation cache (ragplay_test, fake LLM)."""

import subprocess
import sys
from pathlib import Path

import pytest

from ragplay import categorize, db, ingest
from ragplay.categorize import Classification, merchant_key

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("raw,key", [
    ("SQ *BLUE BOTTLE COF 0423 OAKLAND CA", "BLUE BOTTLE COF OAKLAND"),
    ("STARBUCKS STORE 02609 SPRINGFIELD", "STARBUCKS STORE SPRINGFIELD"),
    ("SHELL OIL 00000003144 SPRINGFIELD ST", "SHELL OIL SPRINGFIELD ST"),
    ("THE HOME DEPOT #5106 SPRINGFIELD", "THE HOME DEPOT SPRINGFIELD"),
    ("AMAZON.COM*2K4X1Q AMZN.COM/BILL WA", "AMAZON.COM AMZN.COM/BILL"),
    ("SPOTIFY USA 877-778-1161 NY", "SPOTIFY"),
    ("7-ELEVEN 34512 TORONTO ON", "7-ELEVEN TORONTO"),
    ("TST* LUIGI'S TRATTORIA SPRINGFIELD", "LUIGI'S TRATTORIA SPRINGFIELD"),
    ("12345678", "12345678"),  # nothing left -> fall back to the raw string
])
def test_merchant_key(raw, key):
    assert merchant_key(raw) == key


def test_store_numbers_collapse_to_one_key():
    assert merchant_key("SAFEWAY #0012 SPRINGFIELD ST") == merchant_key("SAFEWAY #9931 SPRINGFIELD ST")


def test_rules_before_llm():
    assert categorize.rule_classify("UBER EATS TORONTO").category == "dining"
    assert categorize.rule_classify("UBER TRIP HELP.UBER.COM").category == "transport"
    assert categorize.rule_classify("LUIGI'S TRATTORIA") is None


@pytest.mark.parametrize("raw, merchant, category", [
    ("AMAZON.CA PRIME MEMBER AMAZON.CA/PRIBC", "Amazon Prime", "subscriptions"),
    ("AMAZON WEB SERVICES WWW.AMAZON.CAON", "Amazon Web Services", "subscriptions"),
    ("AMAZON.CA*5O8AV7I81 866-216-1072 ON", "Amazon", "shopping"),
    ("AMZN MKTP CA WWW.AMAZON.CAON", "Amazon", "shopping"),
])
def test_amazon_rules(raw, merchant, category):
    c = categorize.rule_classify(merchant_key(raw))
    assert (c.merchant, c.category) == (merchant, category)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_statements.py", "--months", "1",
                    "--end", "2026-01", "--out", tmp_path], check=True, capture_output=True)
    db.migrate(db.TEST_SCHEMA)
    c = db.connect_owner(db.TEST_SCHEMA)
    c.autocommit = True
    assert c.execute("SELECT current_schema()").fetchone()[0] == db.TEST_SCHEMA
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories CASCADE")
    for pdf in tmp_path.glob("*.pdf"):
        assert ingest.ingest_file(c, pdf).status == "loaded"
    calls = []

    def fake_llm(key, raw, model=None):
        calls.append(key)
        return Classification(key.split()[0].title(), "other", "test business", "llm")

    monkeypatch.setattr(categorize, "llm_classify", fake_llm)
    c.llm_calls = calls
    yield c
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories CASCADE")
    c.close()


def test_every_transaction_gets_a_category(conn):
    categorize.categorize(conn)
    assert conn.execute("SELECT count(*) FROM transactions WHERE category IS NULL").fetchone()[0] == 0
    assert conn.execute(
        "SELECT DISTINCT category FROM transactions WHERE type = 'payment'").fetchall() == [("payment",)]


def test_llm_called_once_per_new_key(conn):
    first = categorize.categorize(conn)
    assert first.llm == len(conn.llm_calls) == len(set(conn.llm_calls))
    again = categorize.categorize(conn)
    assert again.llm == 0 and again.cached == first.keys


def test_manual_wins(conn):
    categorize.categorize(conn)
    name = conn.execute("SELECT merchant FROM merchant_categories WHERE source = 'llm' LIMIT 1").fetchone()[0]
    categorize.set_category(conn, name, "travel")
    conn.execute("DELETE FROM merchant_aliases WHERE source = 'llm' AND merchant <> %s", (name,))
    categorize.categorize(conn)  # re-run re-classifies deleted aliases; must not touch manual
    assert conn.execute("SELECT category, source FROM merchant_categories WHERE merchant = %s",
                        (name,)).fetchone() == ("travel", "manual")
    cats = conn.execute("SELECT DISTINCT category FROM transactions WHERE merchant = %s", (name,)).fetchall()
    assert cats == [("travel",)]


def test_changed_rule_reroutes_cached_keys(conn, monkeypatch):
    categorize.categorize(conn)
    key = conn.execute("SELECT merchant_key FROM merchant_aliases WHERE merchant = 'Netflix' LIMIT 1").fetchone()[0]
    monkeypatch.setattr(categorize, "RULES", [(r"^NETFLIX", "Netflix Video", "entertainment", "streaming")]
                        + categorize.RULES)
    st = categorize.categorize(conn)
    assert st.rule == 1 and st.llm == 0
    assert conn.execute("SELECT DISTINCT merchant, category FROM transactions WHERE merchant_key = %s",
                        (key,)).fetchall() == [("Netflix Video", "entertainment")]


def test_set_category_rejects_unknown(conn):
    categorize.categorize(conn)
    with pytest.raises(ValueError):
        categorize.set_category(conn, "Netflix", "not-a-category")
