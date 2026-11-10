"""Generate synthetic credit card statement PDFs plus ground-truth JSON.

Deterministic for a given --seed. Built for development, tests and anything shared, so real
statements never need to leave data/raw/.

Layout features that later phases must handle:
- statement period runs from the 15th to the 14th, and dates print as MM/DD (Dec/Jan year rollover)
- payments and refunds are negative amounts
- some long descriptions wrap onto a second line
- the summary box has totals to reconcile against
- multi-page transaction lists with a repeated header
- name/address and a full (fake) card number that ingest must drop or mask
- a recurring subscription with a price increase, and one duplicate charge (Phases 8)

Usage: python scripts/make_fake_statements.py --months 12 --end 2026-09 --out data/fake
"""

import argparse
import json
import random
import textwrap
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

ISSUER = "ACME BANK"
CARD_FULL = "4111 2222 3333 4821"
CARD_LAST4 = "4821"
HOLDER = ["JANE Q SAMPLE", "123 EXAMPLE STREET APT 4", "SPRINGFIELD, ST 00000"]

# (description template, category, low, high, expected visits per statement)
VARIABLE = [
    ("SAFEWAY #{n:04d} SPRINGFIELD ST", "groceries", 25, 160, 4),
    ("TRADER JOE'S #{n:03d} SPRINGFIELD ST", "groceries", 18, 95, 3),
    ("WHOLE FOODS MKT #{n:05d} SPRINGFIELD", "groceries", 30, 140, 1.5),
    ("SQ *BLUE BOTTLE COF {n:04d} OAKLAND CA", "coffee", 4, 12, 4),
    ("STARBUCKS STORE {n:05d} SPRINGFIELD", "coffee", 4, 9, 3),
    ("TST* LUIGI'S TRATTORIA SPRINGFIELD", "dining", 35, 120, 1.5),
    ("DOORDASH*THAI BASIL SAN FRANCISCO CA", "dining", 22, 55, 2),
    ("CHIPOTLE {n:04d} SPRINGFIELD ST", "dining", 11, 24, 2),
    ("SHELL OIL {n:011d} SPRINGFIELD ST", "gas", 35, 75, 2.5),
    ("CHEVRON {n:07d} SPRINGFIELD ST", "gas", 30, 70, 1),
    ("AMAZON.COM*{code} AMZN.COM/BILL WA", "shopping", 9, 180, 4),
    ("TARGET {n:08d} SPRINGFIELD ST", "shopping", 15, 140, 1.5),
    ("THE HOME DEPOT #{n:04d} SPRINGFIELD", "home", 12, 260, 0.8),
    ("CVS/PHARMACY #{n:05d} SPRINGFIELD ST", "health", 8, 60, 1),
    ("UBER *TRIP HELP.UBER.COM CA", "transport", 9, 45, 2),
    ("AMC THEATRES {n:04d} SPRINGFIELD", "entertainment", 14, 48, 0.6),
]

# (description, category, day of month, amount before, amount after, increase from month index)
SUBSCRIPTIONS = [
    ("NETFLIX.COM LOS GATOS CA", "subscriptions", 3, "15.49", "17.99", 6),
    ("SPOTIFY USA 877-778-1161 NY", "subscriptions", 9, "11.99", "11.99", None),
    ("APPLE.COM/BILL 866-712-7753 CA", "subscriptions", 21, "2.99", "2.99", None),
    ("COMCAST CABLE COMM 800-COMCAST PA", "utilities", 12, "89.00", "89.00", None),
    ("PLANET FITNESS CLUB FEES NH", "health", 17, "24.99", "24.99", None),
]

CENT = Decimal("0.01")


@dataclass
class Txn:
    txn_date: str
    post_date: str
    description: str
    amount: str  # Decimal as string; positive = charge, negative = payment/credit
    type: str
    category: str | None


def money(x) -> Decimal:
    return Decimal(str(x)).quantize(CENT, rounding=ROUND_HALF_UP)


def fmt(d: Decimal) -> str:
    return f"-${abs(d):,.2f}" if d < 0 else f"${d:,.2f}"


def add_month(d: date, n: int) -> date:
    m = d.month - 1 + n
    return d.replace(year=d.year + m // 12, month=m % 12 + 1)


def make_statement(rng: random.Random, start: date, idx: int, prev_balance: Decimal):
    end = add_month(start, 1) - timedelta(days=1)
    days = (end - start).days + 1
    txns: list[Txn] = []

    def add(d: date, desc: str, amount: Decimal, kind: str, cat: str | None):
        post = min(d + timedelta(days=rng.choice([0, 1, 1, 2])), end)
        txns.append(Txn(d.isoformat(), post.isoformat(), desc, str(amount), kind, cat))

    for tmpl, cat, lo, hi, freq in VARIABLE:
        visits = sum(rng.random() < freq / 4 for _ in range(4)) + (rng.random() < freq % 1)
        store = rng.randint(1, 9999)
        for _ in range(visits):
            d = start + timedelta(days=rng.randrange(days))
            desc = tmpl.format(n=store, code=f"{rng.randrange(36**6):06X}")
            add(d, desc, money(rng.uniform(lo, hi)), "purchase", cat)

    for desc, cat, day, before, after, from_idx in SUBSCRIPTIONS:
        d = start.replace(day=day) if day >= start.day else add_month(start, 1).replace(day=day)
        price = after if from_idx is not None and idx >= from_idx else before
        add(d, desc, money(price), "purchase", cat)

    # One foreign purchase plus its 3% fee in some months.
    if rng.random() < 0.4:
        d = start + timedelta(days=rng.randrange(days))
        amt = money(rng.uniform(40, 300))
        add(d, "AIRBNB * HM4K2QXYZ LONDON GBR FOREIGN CURRENCY TRANSACTION GBP", amt, "purchase", "travel")
        add(d, "FOREIGN TRANSACTION FEE", money(amt * Decimal("0.03")), "fee", "fees")

    # Duplicate charge in month 4 (Phase 8 duplicate detection).
    if idx == 4:
        dup = next(t for t in txns if t.description.startswith("TARGET")) if any(
            t.description.startswith("TARGET") for t in txns) else txns[0]
        txns.append(Txn(**{**asdict(dup)}))

    # Occasional refund.
    if rng.random() < 0.35:
        src = rng.choice([t for t in txns if t.category == "shopping"] or txns)
        d = date.fromisoformat(src.txn_date) + timedelta(days=rng.randint(2, 8))
        if d <= end:
            add(d, f"REFUND {src.description}", -money(Decimal(src.amount)), "refund", src.category)

    # Previous balance paid in full around the 5th day.
    if prev_balance > 0:
        add(start + timedelta(days=5), "PAYMENT - THANK YOU AUTOPAY", -prev_balance, "payment", None)

    txns.sort(key=lambda t: (t.post_date, t.txn_date, t.description))
    amounts = lambda kind: sum((Decimal(t.amount) for t in txns if t.type in kind), Decimal(0))
    summary = {
        "previous_balance": prev_balance,
        "payments_credits": amounts({"payment", "refund"}),
        "purchases": amounts({"purchase"}),
        "fees": amounts({"fee"}),
        "interest": Decimal("0.00"),
    }
    summary["new_balance"] = sum(summary.values(), Decimal(0))
    return end, txns, summary


def draw_pdf(path: Path, start: date, end: date, txns: list[Txn], summary: dict):
    c = canvas.Canvas(str(path), pagesize=letter)
    w, h = letter

    def header(page: int):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, h - 50, f"{ISSUER}  Platinum Rewards Card")
        c.setFont("Helvetica", 9)
        c.drawRightString(w - 50, h - 50, f"Page {page}")
        c.drawString(50, h - 66, f"Account Number: {CARD_FULL}")
        c.drawString(50, h - 78, f"Statement Period: {start:%m/%d/%Y} - {end:%m/%d/%Y}")

    page = 1
    header(page)
    y = h - 100
    c.setFont("Helvetica", 9)
    for line in HOLDER:
        c.drawString(50, y, line)
        y -= 11

    # Summary box
    y -= 10
    rows = [
        ("Previous Balance", summary["previous_balance"]),
        ("Payments and Other Credits", summary["payments_credits"]),
        ("Purchases", summary["purchases"]),
        ("Fees Charged", summary["fees"]),
        ("Interest Charged", summary["interest"]),
        ("New Balance", summary["new_balance"]),
    ]
    box_top = y
    c.rect(330, box_top - 14 * len(rows) - 22, w - 380, 14 * len(rows) + 22)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(340, box_top - 14, "Account Summary")
    for i, (label, val) in enumerate(rows):
        yy = box_top - 30 - 14 * i
        c.setFont("Helvetica-Bold" if label == "New Balance" else "Helvetica", 9)
        c.drawString(340, yy, label)
        c.drawRightString(w - 60, yy, fmt(val))
    y = box_top - 14 * len(rows) - 50

    def table_header(y):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Transactions")
        y -= 16
        c.setFont("Helvetica-Bold", 9)
        c.drawString(50, y, "Trans Date")
        c.drawString(110, y, "Post Date")
        c.drawString(170, y, "Description")
        c.drawRightString(w - 50, y, "Amount")
        c.line(50, y - 3, w - 50, y - 3)
        return y - 14

    y = table_header(y)
    c.setFont("Helvetica", 9)
    for t in txns:
        desc = t.description
        lines = textwrap.wrap(desc, 48)  # wrap at spaces, as real statements do
        if y - 12 * len(lines) < 60:
            c.showPage()
            page += 1
            header(page)
            y = table_header(h - 105)
            c.setFont("Helvetica", 9)
        c.drawString(50, y, f"{date.fromisoformat(t.txn_date):%m/%d}")
        c.drawString(110, y, f"{date.fromisoformat(t.post_date):%m/%d}")
        c.drawString(170, y, lines[0])
        c.drawRightString(w - 50, y, f"{Decimal(t.amount):,.2f}")
        for extra in lines[1:]:
            y -= 11
            c.drawString(170, y, extra)
        y -= 13
    c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--end", default="2026-09", help="YYYY-MM of the last statement's start month")
    ap.add_argument("--out", type=Path, default=Path("data/fake"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    y, m = map(int, args.end.split("-"))
    first = add_month(date(y, m, 15), -(args.months - 1))
    args.out.mkdir(parents=True, exist_ok=True)

    balance = Decimal("612.40")
    for idx in range(args.months):
        start = add_month(first, idx)
        end, txns, summary = make_statement(rng, start, idx, balance)
        stem = f"acme_{CARD_LAST4}_{end:%Y-%m}"
        draw_pdf(args.out / f"{stem}.pdf", start, end, txns, summary)
        truth = {
            "issuer": ISSUER,
            "card_last4": CARD_LAST4,
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "summary": {k: str(v) for k, v in summary.items()},
            "transactions": [asdict(t) for t in txns],
        }
        (args.out / f"{stem}.json").write_text(json.dumps(truth, indent=2))
        print(f"{stem}: {len(txns)} txns, new balance {fmt(summary['new_balance'])}")
        balance = summary["new_balance"]


if __name__ == "__main__":
    main()
