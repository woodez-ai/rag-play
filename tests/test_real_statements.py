"""Regression check on the real statements, when they are present (cc_statements/ is gitignored).

Failure messages name files only, never amounts or descriptions.
"""

from pathlib import Path

import pytest

from ragplay.parsers import ParseError, parse_file
from ragplay.validate import reconcile

REAL = sorted((Path(__file__).resolve().parents[1] / "cc_statements").glob("*.pdf"))
pytestmark = pytest.mark.skipif(not REAL, reason="no real statements in cc_statements/")


def test_every_real_statement_reconciles():
    failed = []
    for p in REAL:
        try:
            if reconcile(parse_file(p)):
                failed.append(p.name)
        except ParseError:
            failed.append(p.name)
    assert not failed, f"do not reconcile: {failed}"


def test_statements_chain():
    ss = [parse_file(p) for p in REAL]
    breaks = [b.period_end.isoformat() for a, b in zip(ss, ss[1:])
              if a.new_balance != b.previous_balance or (b.period_start - a.period_end).days != 1]
    assert not breaks, f"balance/period chain breaks at statements ending {breaks}"
