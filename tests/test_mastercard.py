"""Mastercard (photographed page images) parser on synthetic OCR words, plus image grouping.

The synthetic page reproduces what OCR of a phone photo gives after levelling: pieces of one row a
few points apart, foreign-currency info lines, transaction dates cropped at the left edge, and a
section total wrapped onto a second line. Fictional merchants and amounts only.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ragplay import ingest
from ragplay.extract import Word
from ragplay.parsers import ParseError, mastercard
from ragplay.validate import reconcile


def line(top: float, *cells: tuple[float, str]) -> list[Word]:
    """Words at (x0, text) on one line; each word's x1 is estimated from its length."""
    return [Word(x, x + 5 * len(t), top, t) for x, t in cells]


def phrase(top: float, x: float, text: str) -> list[Word]:
    out, cur = [], x
    for t in text.split():
        out.append(Word(cur, cur + 5 * len(t), top, t))
        cur += 5 * len(t) + 5
    return out


def page1(section_total: str = "$61.50") -> list[Word]:
    w: list[Word] = []
    w += phrase(40, 290, "Example Bank Rewards Mastercard®")
    w += phrase(90, 20, "Statement date: Oct. 21, 2026")
    w += phrase(103, 20, "Statement period: Sept. 22, 2026 - Oct. 21, 2026")
    w += phrase(150, 20, "Statement details") + phrase(150, 360, "Account summary")
    w += phrase(170, 362, "XXXX XXXX XX12 3456")
    w += phrase(185, 20, "Transaction Posting")
    w += phrase(195, 20, "date date") + phrase(195, 362, "Previous Balance") + phrase(195, 490, "$500.00")
    w += phrase(205, 20, "dd/mm dd/mm") + phrase(205, 117, "Account activity") + phrase(205, 320, "Amount")
    w += phrase(220, 362, "+ Purchases") + phrase(220, 490, "$111.50")
    w += phrase(232, 362, "+ Cash Advances") + phrase(232, 506, "$0.00")
    w += phrase(244, 362, "+ Interest") + phrase(244, 506, "$0.00")
    w += phrase(256, 362, "+ Fees") + phrase(256, 506, "$0.00")
    w += phrase(268, 362, "+ Other charges") + phrase(268, 506, "$0.00")
    w += phrase(280, 362, "- Payments - Thank you") + phrase(280, 496, "$500.00")
    w += phrase(292, 362, "- Other credits") + phrase(292, 506, "$0.00")
    w += phrase(304, 362, "Statement Balance") + phrase(304, 496, "$111.50")
    w += phrase(316, 362, "Credit Limit") + phrase(316, 440, "$5,000.00")  # after the balance: ignored
    # payments section
    w += line(235, (24, "02/10"), (70, "03/10")) + phrase(235, 116, "PAYMENT THANK YOU") + line(235, (309, "-$500.00"))
    w += phrase(248, 117, "Total payment activity") + line(248, (309, "-$500.00"))
    # card 1111: a split row (amount 5 points below the dates), a foreign purchase with its info
    # line, a cropped transaction date, then a total wrapped onto the next line
    w += line(262, (24, "24/09"), (70, "25/09")) + phrase(262, 116, "MAPLE GROCERY #12 TORONTO ON")
    w += line(267, (322, "$40.00"))
    w += line(280, (20, "27/09"), (66, "29/09")) + phrase(280, 112, "HARBOUR BOOKS LONDON GB GBR") + line(280, (322, "$13.50"))
    w += line(289, (112, "10.00"), (140, "GBP"), (160, "1.350000000"))
    w += line(300, (0, "/10"), (48, "02/10")) + phrase(300, 99, "CORNER CAFE TORONTO ON") + line(300, (325, "$8.00"))
    w += phrase(315, 112, "Sub total for ALEX EXAMPLE") + line(315, (300, section_total))
    w += phrase(324, 112, "SMITH - 1111")
    return w


def page2() -> list[Word]:
    w: list[Word] = []
    w += phrase(100, 40, "Transaction Posting")
    w += phrase(120, 40, "dd/mm dd/mm") + phrase(120, 161, "Account activity") + phrase(120, 427, "Amount")
    w += line(140, (43, "05/10"), (101, "06/10")) + phrase(140, 159, "RIVERSIDE HARDWARE") + line(140, (423, "$50.00"))
    w += phrase(141, 520, "Earn points")  # right-margin promo text: ignored
    w += phrase(160, 159, "Sub total for SAM EXAMPLE - 2222") + line(160, (414, "$50.00"))
    return w


def test_parses_and_reconciles():
    s = mastercard.parse([page1(), page2()])
    assert (s.parser, s.card_last4, s.period_start, s.period_end) == (
        "mastercard", "3456", date(2026, 9, 22), date(2026, 10, 21))
    assert not reconcile(s)
    got = [(t.txn_date, t.post_date, t.description, t.amount, t.type, t.card_last4) for t in s.transactions]
    assert got == [
        (date(2026, 10, 2), date(2026, 10, 3), "PAYMENT THANK YOU", Decimal("-500.00"), "payment", "3456"),
        (date(2026, 9, 24), date(2026, 9, 25), "MAPLE GROCERY #12 TORONTO ON", Decimal("40.00"), "purchase", "1111"),
        (date(2026, 9, 27), date(2026, 9, 29), "HARBOUR BOOKS LONDON GB GBR", Decimal("13.50"), "purchase", "1111"),
        # day cropped away: latest date in October on or before the posting date
        (date(2026, 10, 2), date(2026, 10, 2), "CORNER CAFE TORONTO ON", Decimal("8.00"), "purchase", "1111"),
        (date(2026, 10, 5), date(2026, 10, 6), "RIVERSIDE HARDWARE", Decimal("50.00"), "purchase", "2222"),
    ]


def test_section_total_mismatch_is_refused():
    with pytest.raises(ParseError, match="section"):
        mastercard.parse([page1(section_total="$61.60"), page2()])


@pytest.mark.parametrize("token, expected", [
    ("24/09", date(2026, 9, 24)),
    ("4/10", date(2026, 10, 4)),    # one day digit left: 4, 14, 24 or 34 -> latest fit is the 4th
    ("/10", date(2026, 10, 6)),     # no day digits: the posting date itself
    ("09", date(2026, 9, 30)),      # month only, previous month: its last day
    (None, date(2026, 10, 6)),
])
def test_cropped_transaction_dates(token, expected):
    assert mastercard._txn_date(token, date(2026, 10, 6), date(2026, 10, 21)) == expected


def test_detect():
    text = " ".join(w.text for w in page1())
    assert mastercard.detect(text)
    assert not mastercard.detect("Visa STATEMENT FROM SEP 22 TO OCT 21, 2026")


def test_page_images_grouped_into_statements(tmp_path):
    for name in ["mc-bill-Oct2026-2.jpg", "mc-bill-Oct2026-10.jpg", "mc-bill-Oct2026-1.jpg",
                 "a.pdf", "scan.png", "notes.txt"]:
        (tmp_path / name).write_bytes(name.encode())
    got = ingest.statement_files([tmp_path])
    names = sorted([g.name] if isinstance(g, Path) else [p.name for p in g] for g in got)
    assert names == [["a.pdf"], ["mc-bill-Oct2026-1.jpg", "mc-bill-Oct2026-2.jpg", "mc-bill-Oct2026-10.jpg"],
                     ["scan.png"]]
    pages = next(g for g in got if isinstance(g, list) and len(g) == 3)
    assert ingest.statement_name(pages) == "mc-bill-Oct2026-1..3.jpg"
    assert ingest.file_hash(pages) != ingest.file_hash(pages[::-1])  # page order matters
