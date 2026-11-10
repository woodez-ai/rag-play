"""Parser for the real Visa statement layout (card ...2423 in cc_statements/).

Layout, worked out from a masked dump (scripts/layout_dump.py) so no real content was needed:
- Header on every page: `STATEMENT FROM SEP 22 TO OCT 21, 2026`. The start year is printed only
  across a year end: `STATEMENT FROM DEC 22, 2025 TO JAN 21, 2026`.
- Two columns. Left (x < 380): transaction rows `MON DD  MON DD  description  $amount`, each
  followed by a 23-digit reference line, and sometimes a `Foreign Currency - USD ...` line or a
  wrapped second description line. Card sub-headers `9999 99** **** 9999 - NAME` start each
  card's list. The list ends at `TOTAL ACCOUNT BALANCE`.
- Right panel (x >= 380) on page 1: the balance calculation (Previous Account Balance,
  Payments & credits, Purchases & debits, Cash advances, Interest, Fees, Total Account Balance).
- Page 1 ends with a remittance slip, which is ignored (its lines never start with two dates).
"""

import re
from datetime import date
from decimal import Decimal

from ragplay.extract import Line, Word, group_lines
from ragplay.parsers import (
    MONTHS, ParsedStatement, ParsedTxn, ParseError, classify, infer_year, money,
)

COLUMN_SPLIT = 380
# The start year is printed only when it differs from the end year (Dec -> Jan statements).
PERIOD_RE = re.compile(
    r"STATEMENT FROM ([A-Z]{3}) (\d{1,2})(?:, (\d{4}))? TO ([A-Z]{3}) (\d{1,2}), (\d{4})")
CARD_RE = re.compile(r"\b\d{4} \d{2}\*\* \*\*\*\* (\d{4})\b")
REF_RE = re.compile(r"^\d{23}$")
DESC_X = (124, 136)  # description column start (x=128)
SUMMARY = {
    "previous account balance": "previous_balance",
    "payments & credits": "payments_credits",
    "purchases & debits": "purchases",
    "cash advances": "cash_advances",
    "interest": "interest",
    "fees": "fees",
    "total account balance": "new_balance",
}


def detect(first_page_text: str) -> bool:
    return "Visa" in first_page_text and PERIOD_RE.search(first_page_text) is not None


def _is_date(words: list[Word], i: int) -> bool:
    return (len(words) > i + 1 and words[i].text in MONTHS
            and words[i + 1].text.isdigit() and len(words[i + 1].text) <= 2)


def _summary(right: list[Line]) -> dict[str, Decimal]:
    totals: dict[str, Decimal] = {}
    for ln in right:
        label = " ".join(w.text for w in ln.words[:-1]).lower()
        amount = ln.words[-1].text
        if label in SUMMARY and "$" in amount and SUMMARY[label] not in totals:
            totals[SUMMARY[label]] = money(amount)
    missing = set(SUMMARY.values()) - totals.keys()
    if missing:
        raise ParseError(f"balance panel missing: {sorted(missing)}")
    return totals


def parse(pages: list[list[Word]]) -> ParsedStatement:
    if not pages:
        raise ParseError("empty PDF")
    page1 = group_lines(pages[0], 1)
    text1 = "\n".join(ln.text for ln in page1)
    m = PERIOD_RE.search(text1)
    if not m:
        raise ParseError("statement period not found")
    sm, sd, em, ed, ey = MONTHS[m[1]], int(m[2]), MONTHS[m[4]], int(m[5]), int(m[6])
    end = date(ey, em, ed)
    start = date(int(m[3]), sm, sd) if m[3] else infer_year(sm, sd, end)
    card = CARD_RE.search(text1)
    if not card:
        raise ParseError("card number not found")
    statement_card = card[1]
    issuer = page1[0].text.replace("®", "").replace("‡", "").strip()

    right = group_lines([w for w in pages[0] if w.x0 >= COLUMN_SPLIT], 1)
    totals = _summary(right)

    txns: list[ParsedTxn] = []
    current_card = statement_card
    for pno, words in enumerate(pages, 1):
        left = group_lines([w for w in words if w.x0 < COLUMN_SPLIT], pno)
        in_row = False
        last_top = 0.0
        for ln in left:
            w = ln.words
            if ln.text.startswith("TOTAL ACCOUNT BALANCE"):
                break
            if _is_date(w, 0) and _is_date(w, 2):
                if len(w) < 6 or "$" not in w[-1].text:
                    raise ParseError(f"page {pno}: transaction row without an amount")
                amount = money(w[-1].text)
                desc = " ".join(x.text for x in w[4:-1])
                txns.append(ParsedTxn(
                    txn_date=infer_year(MONTHS[w[0].text], int(w[1].text), end),
                    post_date=infer_year(MONTHS[w[2].text], int(w[3].text), end),
                    description=desc, amount=amount, type=classify(desc, amount),
                    card_last4=current_card,
                ))
                in_row, last_top = True, ln.top
                continue
            if (c := CARD_RE.match(ln.text)) and ln.top > 150:
                current_card, in_row = c[1], False
                continue
            if in_row and ln.top - last_top < 14:
                if REF_RE.match(ln.text) or ln.text.startswith("Foreign Currency"):
                    last_top = ln.top
                    continue
                if DESC_X[0] <= ln.x0 <= DESC_X[1] and "$" not in ln.text:
                    txns[-1].description += " " + ln.text
                    txns[-1].type = classify(txns[-1].description, txns[-1].amount)
                    last_top = ln.top
                    continue
            in_row = False

    return ParsedStatement(
        parser="visa", issuer=issuer, card_last4=statement_card, period_start=start,
        period_end=end, transactions=txns, **totals,
    )
