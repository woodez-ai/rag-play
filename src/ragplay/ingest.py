"""Statement (a PDF, or page images) -> parse -> reconcile -> PostgreSQL, one DB transaction per
statement.

- Page images of one statement are named `<stem>-1.jpg`, `<stem>-2.jpg`, ... and are grouped
  and read (OCR, on-device) in page order.

- Idempotent: keyed on the SHA-256 of the file (all pages, in order, for images) (statements.file_hash is UNIQUE). Re-running skips
  files already loaded.
- Fails loudly: a statement that doesn't reconcile to the cent is not inserted at all.
- Only the file name is stored, never its full path. Card numbers are reduced to the last 4
  digits by the parser; names and addresses are never read into rows.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import psycopg

from ragplay.extract import IMAGE_SUFFIXES
from ragplay.parsers import ParsedStatement, parse_file
from ragplay.validate import reconcile


@dataclass
class Result:
    file: str
    status: str  # loaded | skipped | dry-run | failed
    transactions: int = 0
    errors: list[str] = field(default_factory=list)


PAGE_RE = re.compile(r"^(.*?)[-_ ]*(\d+)$")  # "mc-bill-Oct2026-3" -> ("mc-bill-Oct2026", "3")


def file_hash(path: Path | list[Path]) -> str:
    h = hashlib.sha256()
    for p in path if isinstance(path, list) else [path]:
        h.update(p.read_bytes())
    return h.hexdigest()


def statement_name(paths: list[Path]) -> str:
    """The name stored in statements.source_file: the file, or `<stem>-1..4.jpg` for pages."""
    if len(paths) == 1:
        return paths[0].name
    m = PAGE_RE.match(paths[0].stem)
    stem = m[1] if m else paths[0].stem
    return f"{stem}-1..{len(paths)}{paths[0].suffix}"


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


def ingest_file(conn: psycopg.Connection | None, path: Path | list[Path]) -> Result:
    """Ingest one statement: a PDF, or its page images in order. conn=None is a dry run
    (parse + reconcile only)."""
    paths = path if isinstance(path, list) else [path]
    name = statement_name(paths)
    digest = file_hash(paths)
    if conn is not None and conn.execute(
            "SELECT 1 FROM statements WHERE file_hash = %s", (digest,)).fetchone():
        return Result(name, "skipped")
    try:
        s = parse_file(paths)
    except ValueError as e:
        return Result(name, "failed", errors=[f"parse: {e}"])
    errors = reconcile(s)
    if errors:
        return Result(name, "failed", len(s.transactions), errors)
    if conn is None:
        return Result(name, "dry-run", len(s.transactions))
    with conn.transaction():  # statement + all its rows, or nothing
        insert_statement(conn, s, name, digest)
    return Result(name, "loaded", len(s.transactions))


def statement_files(paths: list[Path]) -> list[Path | list[Path]]:
    """Expand folders, then group page images into statements: `<stem>-<n>.jpg` files that
    share a stem are one statement, in page-number order. PDFs are one statement each."""
    files: list[Path] = []
    for p in paths:
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in {".pdf", *IMAGE_SUFFIXES}) \
            if p.is_dir() else [p]
    out: list[Path | list[Path]] = []
    pages: dict[tuple[Path, str], list[tuple[int, Path]]] = {}
    for f in files:
        m = PAGE_RE.match(f.stem)
        if f.suffix.lower() in IMAGE_SUFFIXES and m:
            pages.setdefault((f.parent, m[1]), []).append((int(m[2]), f))
        elif f.suffix.lower() in IMAGE_SUFFIXES:
            out.append([f])
        else:
            out.append(f)
    out += [[f for _, f in sorted(group)] for group in pages.values()]
    return out
