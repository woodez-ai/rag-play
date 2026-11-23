"""Parser for the Mastercard statement, photographed as page images (cc_statements/mc-bill-*.jpg).

Layout, worked out from a masked OCR dump (scripts/layout_dump.py on the images), so no real
content was needed:
- Page 1 header: `Statement date: Oct. 21, 2026` and
  `Statement period: Sept. 22, 2026 - Oct. 21, 2026`.
- Page 1 right panel, under `Account summary`: the card number `XXXX XXXX XX12 3456`, then
  `Previous Balance`, `+ Purchases`, `+ Cash Advances`, other `+ ...` lines, `+ Interest`,
  `+ Fees`, `- Payments - Thank you`, `- Other credits`, `Statement Balance`. Limits and the
  minimum payment follow and are ignored. Panel amounts are unsigned; the sign is the label's.
- Transactions on every page under `Transaction date | Posting date | Account activity | Amount`:
  `DD/MM  DD/MM  description  $amount` (credits `-$amount`; dates are day first). A foreign purchase adds a line
  `12.34 USD 1.234567890`. The payments list ends with `Total payment activity $total`, each
  card's list with `... total for NAME - 1234 $total` (sometimes wrapped onto two lines).

OCR of a phone photo is not as tidy as a PDF text layer, even after extract.read_image levels
the page:
- A row's date, description and amount can land a few points apart vertically, so each row is
  built around its amount: other words join the nearest amount within ROW_TOL.
- The left edge can be cropped, cutting the transaction date's day to `5/10`, `/10` or `10`.
  It is rebuilt from what is left and the posting date (a transaction is never after its
  posting date); with no day digits left it is approximate.
- Every card section must add up to its printed total, on top of validate.reconcile(), so a
  misread digit is refused rather than loaded.
"""

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from ragplay.extract import Word, group_lines
from ragplay.parsers import MONTHS, ParsedStatement, ParsedTxn, ParseError, classify, infer_year, money

MONEY_RE = re.compile(r"^-?\$[\d,]+\.\d{2}$")
DATE_RE = re.compile(r"^(?:\d{0,2}/?\d{1,2}/?|/)$")  # full DD/MM or what is left of it after cropping
FULL_DATE_RE = re.compile(r"^(\d{1,2})/(\d{2})/?$")
MDY = r"([A-Z][a-z]{2,4})\.? (\d{1,2}), (\d{4})"
PERIOD_RE = re.compile(rf"Statement period: {MDY} - {MDY}")
CARD_RE = re.compile(r"\b[A-Z*•]{4} [A-Z*•]{4} [A-Z*•]{2}\d{2} (\d{4})\b")
SECTION_CARD_RE = re.compile(r"-\s*(\d{4})\b")
FX_RATE_RE = re.compile(r"^\d+\.\d{5,}$")
PLAIN_AMOUNT_RE = re.compile(r"^\d[\d,]*\.\d{2}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
ROW_TOL = 8.0    # rows are 15-19 points apart; split pieces of one row are within ~6
WRAP_TOL = 18.0  # a wrapped description line sits about one line from its row
ZERO = Decimal("0.00")


def detect(first_page_text: str) -> bool:
    return "Mastercard" in first_page_text and "Statement period" in first_page_text


def _month(name: str) -> int:
    m = MONTHS.get(name[:3].upper())
    if m is None:
        raise ParseError(f"unknown month {name!r}")
    return m


@dataclass
class _Row:
    page: int
    amount: Word
    words: list[Word] = field(default_factory=list)  # on the row, any order
    wrap: list[Word] = field(default_factory=list)   # wrapped description line(s) below it

    def tokens(self) -> list[str]:
        return [w.text for w in sorted(self.words, key=lambda w: w.x0)] + \
               [w.text for w in sorted(self.wrap, key=lambda w: (round(w.top), w.x0))]


def _summary(page1: list[Word]) -> dict[str, Decimal]:
    head = next((a for ln in group_lines(page1, 1, y_tol=4.0)
                 for a, b in zip(ln.words, ln.words[1:]) if (a.text, b.text) == ("Account", "summary")), None)
    if head is None:
        raise ParseError("account summary not found")
    panel = group_lines([w for w in page1 if w.x0 >= head.x0 - 15 and w.top > head.top], 1, y_tol=4.0)
    t = {k: ZERO for k in ("purchases", "cash_advances", "interest", "fees", "credits")}
    seen: set[str] = set()
    for ln in panel:
        amounts = [w for w in ln.words if MONEY_RE.match(w.text)]
        if not amounts:
            continue
        label = " ".join(w.text for w in ln.words if w not in amounts).lower()
        amount = money(amounts[-1].text).copy_abs()
        if "previous balance" in label:
            t["previous_balance"] = amount
        elif "statement balance" in label:
            t["new_balance"] = amount
            break
        elif label.startswith("+"):
            key = ("cash_advances" if "cash" in label else "interest" if "interest" in label
                   else "fees" if "fees" in label else "purchases")  # other charges count as purchases
            t[key] += amount
            seen.add(key)
        elif label.startswith("-"):
            t["credits"] += amount
    missing = {"previous_balance", "new_balance"} - t.keys() | {"purchases", "interest", "fees"} - seen
    if missing:
        raise ParseError(f"account summary missing: {sorted(missing)}")
    return {"previous_balance": t["previous_balance"], "payments_credits": -t["credits"],
            "purchases": t["purchases"], "cash_advances": t["cash_advances"],
            "interest": t["interest"], "fees": t["fees"], "new_balance": t["new_balance"]}


def _drop_fx(words: list[Word], page: int) -> list[Word]:
    """Remove foreign-currency info (`12.34 USD 1.234567890`), even when OCR merged it into a row."""
    keep: list[Word] = []
    for ln in group_lines(words, page, y_tol=4.0):
        ws = ln.words
        drop = {i for i, w in enumerate(ws) if FX_RATE_RE.match(w.text)}
        for i in range(1, len(ws)):
            if CURRENCY_RE.match(ws[i].text) and PLAIN_AMOUNT_RE.match(ws[i - 1].text):
                drop |= {i - 1, i}
        keep += [w for i, w in enumerate(ws) if i not in drop]
    return keep


def _page_rows(words: list[Word], page: int) -> list[_Row]:
    lines = group_lines(words, page, y_tol=4.0)
    head = next((ln for ln in lines if "Account" in ln.text and "activity" in ln.text), None)
    if head is None:
        return []
    amount_head = min((w for ln in lines for w in ln.words if w.text == "Amount" and abs(ln.top - head.top) < 40),
                      key=lambda w: abs(w.top - head.top), default=None)
    if amount_head is None:
        raise ParseError(f"page {page}: Amount column not found")
    lo, hi = amount_head.x0 - 30, amount_head.x1 + 40
    body = _drop_fx([w for w in words if w.top > head.top + 4 and w.x0 < hi], page)
    rows = [_Row(page, w) for w in body if lo <= w.x0 and MONEY_RE.match(w.text)]
    if not rows:
        return []
    loose = []
    for w in body:
        if w.x0 >= lo:
            continue
        near = min(rows, key=lambda r: abs(r.amount.top - w.top))
        if abs(near.amount.top - w.top) <= ROW_TOL:
            near.words.append(w)
        else:
            loose.append(w)
    for w in loose:  # wrapped description lines and stray pieces: the nearest row, if close
        near = min(rows, key=lambda r: abs(r.amount.top - w.top))
        if abs(near.amount.top - w.top) <= WRAP_TOL:
            near.wrap.append(w)
    return sorted(rows, key=lambda r: r.amount.top)


def _dates(row: _Row) -> tuple[str | None, str | None, list[Word]]:
    """(transaction date token, posting date token, description words) for one row."""
    ws = sorted(row.words, key=lambda w: w.x0)
    lead = []
    while ws and len(lead) < 2 and DATE_RE.match(ws[0].text):
        lead.append(ws.pop(0))
    if len(lead) == 2:
        return lead[0].text, lead[1].text, ws
    if len(lead) == 1:
        # Only one date read: the posting date sits right before the description, the
        # transaction date a column further left.
        gap = (ws[0].x0 if ws else lead[0].x1 + 100) - lead[0].x1
        return (None, lead[0].text, ws) if gap < 40 else (lead[0].text, None, ws)
    return None, None, ws


def _post_date(token: str | None, end: date) -> date | None:
    m = FULL_DATE_RE.match(token or "")
    if not m or not 1 <= int(m[2]) <= 12:
        return None
    try:
        return infer_year(int(m[2]), int(m[1]), end)
    except ValueError:
        return None


def _txn_date(token: str | None, post: date, end: date) -> date:
    """Transaction date from a DD/MM token that may have lost day digits at the cropped left
    edge ('5/10', '/10', '10'): the latest date on or before the posting date (and at most 31
    days before it) that fits the digits that are left. Falls back to the posting date."""
    tok = (token or "").rstrip("/")
    day_part, _, month_part = tok.rpartition("/") if "/" in tok else ("", "", tok)
    months = [int(month_part)] if month_part.isdigit() and 1 <= int(month_part) <= 12 else []
    if not months:
        return post
    if len(day_part) == 2 and day_part.isdigit():
        days = [int(day_part)]
    elif len(day_part) == 1 and day_part.isdigit():
        days = [int(day_part) + 10 * k for k in range(4)]
    else:
        days = list(range(1, 32))
    fits = []
    for mo in months:
        for d in days:
            try:
                c = infer_year(mo, d, end)
            except ValueError:
                continue
            if timedelta(0) <= post - c <= timedelta(days=31):
                fits.append(c)
    return max(fits, default=post)


def parse(pages: list[list[Word]]) -> ParsedStatement:
    if not pages:
        raise ParseError("no pages")
    page1 = group_lines(pages[0], 1, y_tol=4.0)
    text1 = "\n".join(ln.text for ln in page1)
    m = PERIOD_RE.search(text1)
    if not m:
        raise ParseError("statement period not found")
    start = date(int(m[3]), _month(m[1]), int(m[2]))
    end = date(int(m[6]), _month(m[4]), int(m[5]))
    card = CARD_RE.search(text1)
    if not card:
        raise ParseError("card number not found")
    statement_card = card[1]
    issuer = next((ln.text for ln in page1 if "Mastercard" in ln.text), "Mastercard")
    issuer = issuer.replace("®", "").strip()
    totals = _summary(pages[0])

    txns: list[ParsedTxn] = []
    pending: list[ParsedTxn] = []
    for pno, words in enumerate(pages, 1):
        for row in _page_rows(words, pno):
            amount = money(row.amount.text)
            text = " ".join(row.tokens())
            if re.search(r"\btotal\b", text, re.I):
                if "payment activity" in text.lower():
                    section_card = statement_card
                elif c := SECTION_CARD_RE.search(text):
                    section_card = c[1]
                else:
                    raise ParseError(f"page {pno}: section total without a card number")
                got = sum((t.amount for t in pending), ZERO)
                if got != amount:
                    raise ParseError(f"card ...{section_card} section: rows add to {got}, "
                                     f"printed total {amount} (off by {got - amount})")
                for t in pending:
                    t.card_last4 = section_card
                txns += pending
                pending = []
                continue
            txn_tok, post_tok, desc_words = _dates(row)
            post = _post_date(post_tok, end)
            if post is None:
                raise ParseError(f"page {pno}: row without a readable posting date")
            desc = " ".join([w.text for w in desc_words] +
                            [w.text for w in sorted(row.wrap, key=lambda w: (round(w.top), w.x0))])
            if not desc:
                raise ParseError(f"page {pno}: row without a description")
            pending.append(ParsedTxn(
                txn_date=_txn_date(txn_tok, post, end), post_date=post, description=desc,
                amount=amount, type=classify(desc, amount), card_last4=statement_card))
    if pending:
        raise ParseError(f"{len(pending)} row(s) after the last section total")
    return ParsedStatement(parser="mastercard", issuer=issuer, card_last4=statement_card,
                           period_start=start, period_end=end, transactions=txns, **totals)
