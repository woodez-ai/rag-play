"""One deterministic parser per issuer layout -> ParsedStatement.

Sign convention (same as the DB): charges are positive, payments and credits negative.
Statement summary totals keep the sign printed on the statement (payments_credits <= 0).
"""

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

from ragplay.extract import Word, read_pages

MONTHS = {m: i for i, m in enumerate(
    "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), start=1)}
AMOUNT_RE = re.compile(r"^-?\$?[\d,]+\.\d{2}$")


@dataclass
class ParsedTxn:
    txn_date: date
    post_date: date | None
    description: str
    amount: Decimal
    type: str  # purchase | payment | refund | fee | interest
    card_last4: str


@dataclass
class ParsedStatement:
    parser: str
    issuer: str
    card_last4: str
    period_start: date
    period_end: date
    previous_balance: Decimal
    payments_credits: Decimal  # as printed: zero or negative
    purchases: Decimal
    cash_advances: Decimal
    interest: Decimal
    fees: Decimal
    new_balance: Decimal
    transactions: list[ParsedTxn] = field(default_factory=list)


class ParseError(ValueError):
    pass


def money(text: str) -> Decimal:
    """'-$1,234.56' / '1,234.56' / '$0.00' -> Decimal."""
    if not AMOUNT_RE.match(text):
        raise ParseError(f"not an amount: {text!r}")
    return Decimal(text.replace("$", "").replace(",", ""))


def infer_year(month: int, day: int, period_end: date) -> date:
    """Statements print dates without a year. A month after the closing month belongs to the
    previous year (a December charge on a January statement)."""
    year = period_end.year - 1 if month > period_end.month else period_end.year
    return date(year, month, day)


def classify(description: str, amount: Decimal) -> str:
    """Bank-generated charges start or end with the keyword ("FOREIGN TRANSACTION FEE",
    "INTEREST CHARGE"). Merchant descriptors end with a location, so a merchant whose name
    contains FEE ("... SVC FEE TORONTO ON") stays a purchase. Reconcile catches any miss."""
    d = description.upper()
    if amount < 0:
        return "payment" if re.search(r"\bPAYMENT\b", d) else "refund"
    if re.search(r"^INTEREST\b|\bINTEREST$", d):
        return "interest"
    if re.search(r"^FEES?\b|\bFEES?$", d):
        return "fee"
    return "purchase"


def parse_file(path: str | Path | list[Path]) -> ParsedStatement:
    """Pick the parser whose detect() matches page 1 and run it. `path` is one PDF, or the
    page images of one statement in page order (OCR)."""
    from ragplay.extract import IMAGE_SUFFIXES, read_image
    from ragplay.parsers import acme, mastercard, visa

    paths = [Path(p) for p in path] if isinstance(path, list) else [Path(path)]
    if paths[0].suffix.lower() in IMAGE_SUFFIXES:
        pages = [read_image(p) for p in paths]
    else:
        pages = read_pages(paths[0])
    first = " ".join(w.text for w in pages[0]) if pages else ""
    for mod in (acme, visa, mastercard):
        if mod.detect(first):
            return mod.parse(pages)
    raise ParseError(f"no parser recognises {paths[0].name}")


__all__ = ["ParsedStatement", "ParsedTxn", "ParseError", "Word", "parse_file"]
