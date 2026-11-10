"""Reconcile parsed transactions against the statement's own summary totals.

Every check must hold to the cent, or ingest refuses the statement. A parser bug (a missed
row, a wrapped line read as a new row, a wrong sign) shows up here instead of in an answer.
"""

from decimal import Decimal

from ragplay.parsers import ParsedStatement

ZERO = Decimal("0.00")


def reconcile(s: ParsedStatement) -> list[str]:
    """Return a list of failed checks (empty = reconciled)."""
    def total(*kinds: str) -> Decimal:
        return sum((t.amount for t in s.transactions if t.type in kinds), ZERO)

    errors = []

    def check(name: str, got: Decimal, want: Decimal) -> None:
        if got != want:
            errors.append(f"{name}: parsed {got} vs statement {want} (off by {got - want})")

    check("summary arithmetic",
          s.previous_balance + s.payments_credits + s.purchases + s.cash_advances
          + s.interest + s.fees, s.new_balance)
    check("payments & credits", total("payment", "refund"), s.payments_credits)
    check("purchases", total("purchase"), s.purchases + s.cash_advances)
    check("fees", total("fee"), s.fees)
    check("interest", total("interest"), s.interest)
    check("balance roll-forward",
          s.previous_balance + sum((t.amount for t in s.transactions), ZERO), s.new_balance)
    if not s.transactions:
        errors.append("no transactions parsed")
    out = [t for t in s.transactions
           if not (s.period_start <= t.post_date <= s.period_end)] if s.transactions else []
    if out:
        errors.append(f"{len(out)} transaction(s) posted outside the statement period")
    return errors
