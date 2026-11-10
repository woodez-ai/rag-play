"""Statement PDF -> parse -> reconcile -> PostgreSQL, one DB transaction per statement.

- Idempotent: keyed on the file's SHA-256 (statements.file_hash is UNIQUE). Re-running skips
  files already loaded.
- Fails loudly: a statement that doesn't reconcile to the cent is not inserted at all.
- Only the file name is stored, never its full path. Card numbers are reduced to the last 4
  digits by the parser; names and addresses are never read into rows.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from ragplay.parsers import ParsedStatement, parse_file
from ragplay.validate import reconcile


@dataclass
class Result:
    file: str
    status: str  # loaded | skipped | dry-run | failed
    transactions: int = 0
    errors: list[str] = field(default_factory=list)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def insert_statement(conn: psycopg.Connection, s: ParsedStatement, source_file: str,
                     digest: str) -> int:
    sid = conn.execute(
        """INSERT INTO statements (card_last4, issuer, parser, period_start, period_end,
               previous_balance, closing_balance, total_purchases, total_payments,
               total_cash_advances, total_fees, total_interest, source_file, file_hash)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (s.card_last4, s.issuer, s.parser, s.period_start, s.period_end, s.previous_balance,
         s.new_balance, s.purchases, s.payments_credits, s.cash_advances, s.fees, s.interest,
         source_file, digest),
    ).fetchone()[0]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO transactions (statement_id, txn_date, post_date, description_raw,
                   amount, type, card_last4) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            [(sid, t.txn_date, t.post_date, t.description, t.amount, t.type, t.card_last4)
             for t in s.transactions],
        )
    return sid


def ingest_file(conn: psycopg.Connection | None, path: Path) -> Result:
    """Ingest one PDF. conn=None is a dry run (parse + reconcile only)."""
    digest = file_hash(path)
    if conn is not None and conn.execute(
            "SELECT 1 FROM statements WHERE file_hash = %s", (digest,)).fetchone():
        return Result(path.name, "skipped")
    try:
        s = parse_file(path)
    except ValueError as e:
        return Result(path.name, "failed", errors=[f"parse: {e}"])
    errors = reconcile(s)
    if errors:
        return Result(path.name, "failed", len(s.transactions), errors)
    if conn is None:
        return Result(path.name, "dry-run", len(s.transactions))
    with conn.transaction():  # statement + all its rows, or nothing
        insert_statement(conn, s, path.name, digest)
    return Result(path.name, "loaded", len(s.transactions))


def pdf_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    return out
