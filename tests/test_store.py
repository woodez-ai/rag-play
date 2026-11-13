"""Embeddings + search in ragplay_test with a fake deterministic embedder (no Ollama needed)."""

import hashlib
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from ragplay import categorize, db, ingest, store
from ragplay.categorize import Classification

ROOT = Path(__file__).resolve().parents[1]
DIM = 16


def fake_embed(texts, model=None, kind="document"):
    """Same text -> same vector; texts sharing a first word point roughly the same way."""
    out = []
    for t in texts:
        word = t.split(":")[0].split()[0].lower()
        base = np.frombuffer(hashlib.sha256(word.encode()).digest()[:DIM], dtype=np.uint8)
        out.append(base.astype(np.float32) + 1)
    return np.stack(out)


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake")
    subprocess.run([sys.executable, ROOT / "scripts/make_fake_statements.py", "--months", "2",
                    "--end", "2026-01", "--out", out], check=True, capture_output=True)
    db.migrate(db.TEST_SCHEMA)
    c = db.connect_owner(db.TEST_SCHEMA)
    c.autocommit = True
    assert c.execute("SELECT current_schema()").fetchone()[0] == db.TEST_SCHEMA
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories CASCADE")
    for pdf in sorted(out.glob("*.pdf")):
        assert ingest.ingest_file(c, pdf).status == "loaded"
    mp = pytest.MonkeyPatch()
    mp.setattr(categorize, "llm_classify",
               lambda key, raw, model=None: Classification(key.split()[0].title(), "other", "test shop", "llm"))
    mp.setattr(store, "embed", fake_embed)
    categorize.categorize(c)
    store.embed_transactions(c, "fake")
    yield c
    mp.undo()
    c.execute("TRUNCATE statements, merchant_aliases, merchant_categories CASCADE")
    c.close()


def test_every_transaction_embedded_once(conn):
    n_tx = conn.execute("SELECT count(*) FROM transactions").fetchone()[0]
    assert conn.execute("SELECT count(*) FROM transaction_embeddings WHERE model = 'fake'").fetchone()[0] == n_tx
    assert store.embed_transactions(conn, "fake") == (0, 0)  # nothing new


def test_changed_description_is_reembedded(conn):
    merchant = conn.execute("SELECT merchant FROM merchant_categories WHERE source = 'llm' LIMIT 1").fetchone()[0]
    n = conn.execute("SELECT count(*) FROM transactions WHERE merchant = %s", (merchant,)).fetchone()[0]
    conn.execute("UPDATE merchant_categories SET description = 'changed' WHERE merchant = %s", (merchant,))
    assert store.embed_transactions(conn, "fake") == (n, 1)
    content = conn.execute(
        "SELECT DISTINCT e.content FROM transaction_embeddings e JOIN transactions t ON t.id = e.transaction_id"
        " WHERE t.merchant = %s", (merchant,)).fetchall()
    assert content == [(f"{merchant}: changed",)]


@pytest.mark.parametrize("mode", ["vector", "text", "hybrid"])
def test_filters_are_hard_constraints(conn, mode):
    hits = store.search(conn, "safeway", mode=mode, model="fake", k=50,
                        date_from=date(2025, 12, 20), date_to=date(2026, 1, 10),
                        min_amount=Decimal("20"), max_amount=Decimal("100"))
    for h in hits:
        assert date(2025, 12, 20) <= h.txn_date <= date(2026, 1, 10)
        assert Decimal("20") <= h.amount <= Decimal("100")


def test_category_filter(conn):
    hits = store.search(conn, "anything", model="fake", k=50, category="payment")
    assert hits and all(h.category == "payment" for h in hits)


def test_text_mode_finds_exact_name_and_typo(conn):
    for q in ("safeway", "safewya"):
        hits = store.search(conn, q, mode="text", model="fake", k=5)
        assert hits and all(h.merchant == "Safeway" for h in hits), q


def test_vector_mode_ranks_same_merchant_first(conn):
    hits = store.search(conn, "Safeway", mode="vector", model="fake", k=3)
    assert all(h.merchant == "Safeway" for h in hits)


def test_unknown_mode():
    with pytest.raises(ValueError):
        store.search(None, "x", mode="nope")
