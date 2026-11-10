"""Parser for the synthetic ACME BANK layout (scripts/make_fake_statements.py).

Rows: `MM/DD  MM/DD  description  amount` with columns at x=50/110/170 and amounts
right-aligned. Long descriptions wrap onto a second line that starts in the description column.
"""

import re
from datetime import date
from decimal import Decimal

from ragplay.extract import Word, group_lines
from ragplay.parsers import (
    ParsedStatement, ParsedTxn, ParseError, classify, infer_year, money,
)

DATE_RE = re.compile(r"^(\d{2})/(\d{2})$")
DESC_X = (165, 175)  # description column start
SUMMARY = {
    "Previous Balance": "previous_balance",
    "Payments and Other Credits": "payments_credits",
    "Purchases": "purchases",
    "Fees Charged": "fees",
    "Interest Charged": "interest",
    "New Balance": "new_balance",
}


def detect(first_page_text: str) -> bool:
    return "ACME BANK" in first_page_text


def parse(pages: list[list[Word]]) -> ParsedStatement:
    lines = [ln for i, ws in enumerate(pages, 1) for ln in group_lines(ws, i)]
    text = "\n".join(ln.text for ln in lines)

    m = re.search(r"Statement Period: (\d{2})/(\d{2})/(\d{4}) - (\d{2})/(\d{2})/(\d{4})", text)
    if not m:
        raise ParseError("statement period not found")
    sm, sd, sy, em, ed, ey = map(int, m.groups())
    start, end = date(sy, sm, sd), date(ey, em, ed)
    card = re.search(r"Account Number: (?:\d{4} ){3}(\d{4})", text)
    if not card:
        raise ParseError("card number not found")
    last4 = card.group(1)

    totals: dict[str, Decimal] = {}
    for ln in lines:
        label, amount = " ".join(w.text for w in ln.words[:-1]), ln.words[-1].text
        if label in SUMMARY and amount.startswith(("$", "-$")):
            totals[SUMMARY[label]] = money(amount)
    missing = set(SUMMARY.values()) - totals.keys()
    if missing:
        raise ParseError(f"summary box missing: {sorted(missing)}")

    txns: list[ParsedTxn] = []
    prev = None  # last transaction row (Line) for wrapped descriptions
    for ln in lines:
        w = ln.words
        d1 = DATE_RE.match(w[0].text)
        d2 = DATE_RE.match(w[1].text) if len(w) > 1 else None
        if d1 and d2 and len(w) >= 4:
            amount = money(w[-1].text)
            desc = " ".join(x.text for x in w[2:-1])
            txns.append(ParsedTxn(
                txn_date=infer_year(int(d1[1]), int(d1[2]), end),
                post_date=infer_year(int(d2[1]), int(d2[2]), end),
                description=desc, amount=amount, type=classify(desc, amount), card_last4=last4,
            ))
            prev = ln
        elif (prev is not None and ln.page == prev.page and DESC_X[0] <= ln.x0 <= DESC_X[1]
              and ln.top - prev.top < 14):
            txns[-1].description += " " + ln.text
            txns[-1].type = classify(txns[-1].description, txns[-1].amount)
            prev = ln
        else:
            prev = None

    return ParsedStatement(
        parser="acme", issuer="ACME BANK", card_last4=last4, period_start=start,
        period_end=end, cash_advances=Decimal("0.00"), transactions=txns, **totals,
    )
