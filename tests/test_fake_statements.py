"""The synthetic generator must produce statements whose summary box reconciles exactly."""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pdfplumber
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fake_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("fake")
    subprocess.run(
        [sys.executable, ROOT / "scripts/make_fake_statements.py", "--months", "3",
         "--end", "2026-01", "--out", out],
        check=True, capture_output=True,
    )
    return out


def load(fake_dir):
    return [(p, json.loads(p.read_text())) for p in sorted(fake_dir.glob("*.json"))]


def test_summary_reconciles(fake_dir):
    for _, t in load(fake_dir):
        s = {k: Decimal(v) for k, v in t["summary"].items()}
        tx = t["transactions"]
        total = lambda kinds: sum((Decimal(x["amount"]) for x in tx if x["type"] in kinds), Decimal(0))
        assert total({"purchase"}) == s["purchases"]
        assert total({"payment", "refund"}) == s["payments_credits"]
        assert total({"fee"}) == s["fees"]
        assert sum(v for k, v in s.items() if k != "new_balance") == s["new_balance"]


def test_spans_year_rollover(fake_dir):
    periods = [(t["period_start"], t["period_end"]) for _, t in load(fake_dir)]
    assert any(a[:4] != b[:4] for a, b in periods)


def test_pdf_contains_every_amount(fake_dir):
    for path, t in load(fake_dir):
        with pdfplumber.open(path.with_suffix(".pdf")) as pdf:
            text = "\n".join(p.extract_text() for p in pdf.pages)
        for x in t["transactions"]:
            assert f"{Decimal(x['amount']):,.2f}" in text


def test_deterministic(fake_dir, tmp_path):
    subprocess.run(
        [sys.executable, ROOT / "scripts/make_fake_statements.py", "--months", "3",
         "--end", "2026-01", "--out", tmp_path],
        check=True, capture_output=True,
    )
    for p, t in load(fake_dir):
        assert json.loads((tmp_path / p.name).read_text()) == t
