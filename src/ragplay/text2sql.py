"""Text-to-SQL: the LLM writes one PostgreSQL SELECT, the database computes the answer.

Flow (`ask`): prompt (schema + categories + conventions + examples + today's date) -> qwen3:8b
returns a ```sql block -> guard (single SELECT/WITH, row cap) -> run read-only with a timeout ->
on error or no rows, feed the problem back and retry (max 2) -> the LLM phrases the answer from
the returned rows -> code checks every number in the answer appears in those rows.

Safety is layered: rag_reader can only SELECT (server), the session is read-only with a
statement timeout (client), and check_sql() refuses anything but one SELECT. Generated SQL is
untrusted; the database permissions, not the prompt, are what make it safe.
"""


import re
import time
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import ollama
import psycopg

MODEL = "qwen3:8b"
MAX_RETRIES = 2
ROW_CAP = 200
TIMEOUT_MS = 5000
NUM_CTX = 8192

SCHEMA_DOC = """\
Tables (PostgreSQL 17):

statements -- one row per monthly credit card statement
  id bigint, card_last4 char(4), period_start date, period_end date,
  previous_balance numeric(12,2), closing_balance numeric(12,2),
  total_purchases numeric(12,2), total_payments numeric(12,2) -- negative,
  total_fees numeric(12,2), total_interest numeric(12,2)

transactions -- one row per line on a statement
  id bigint, statement_id bigint -> statements.id,
  txn_date date        -- when the purchase happened (use this for "when" questions)
  post_date date,
  merchant text        -- clean name, e.g. 'Starbucks', 'Trader Joe''s'
  description_raw text -- as printed by the bank, e.g. 'SQ *BLUE BOTTLE COF 0423 OAKLAND CA'
  amount numeric(12,2) -- purchases/fees/interest POSITIVE; payments and refunds NEGATIVE
  type text            -- 'purchase' | 'refund' | 'payment' | 'fee' | 'interest'
  category text        -- one of: {categories}

merchant_categories -- one row per merchant
  merchant text, category text, description text -- e.g. 'specialty coffee roaster and cafe'
"""

RULES = """\
Conventions:
- "Spending"/"spent" = SUM(amount) over type IN ('purchase', 'refund') (refunds net off).
  Never include payments. Fees and interest only when asked.
- Interest or fees CHARGED: use statements.total_interest / statements.total_fees (by
  period_end), wrapped in COALESCE(..., 0). They may not appear as transactions, and zero
  is a real answer, not missing data.
- "Charges"/"charged by"/"purchases" = type = 'purchase'.
- Kinds of business ("gas stations", "coffee shops", "restaurants") are CATEGORIES: filter
  with category = '...' or category IN (...) when the question names several kinds.
  Use merchant ILIKE '%name%' only for a named business (names may differ slightly).
- Time ranges: filter on txn_date. Today is {today}. "Last month" = the previous calendar
  month. Quarters: Q1 Jan-Mar, Q2 Apr-Jun, Q3 Jul-Sep, Q4 Oct-Dec. If a year is not given,
  use the most recent one that has data for that period.
- Only if the question contains the word "statement" ("on my March 2026 statement"): use the
  statement whose period_end is in that month, filter ONLY by statement_id and do NOT also
  filter txn_date (a statement spans two calendar months). Months, quarters and years without
  the word "statement" always use txn_date.
- Return just the columns needed to answer, with clear aliases (e.g. total, month, n).
  Round money with ROUND(x, 2). Order lists sensibly and LIMIT them (<= 50 rows).
- One SELECT statement only (CTEs allowed). No semicolons, no comments.
"""

EXAMPLES = """\
Examples:
Q: How much did I spend on groceries in January 2026?
SQL: SELECT ROUND(SUM(amount), 2) AS total FROM transactions WHERE category = 'groceries' AND type IN ('purchase', 'refund') AND txn_date >= DATE '2026-01-01' AND txn_date < DATE '2026-02-01'
Q: Top 3 merchants by spend last year?
SQL: SELECT merchant, ROUND(SUM(amount), 2) AS total FROM transactions WHERE type IN ('purchase', 'refund') AND txn_date >= date_trunc('year', DATE '{today}') - INTERVAL '1 year' AND txn_date < date_trunc('year', DATE '{today}') GROUP BY merchant ORDER BY total DESC LIMIT 3
Q: How much did I spend on travel and transport on my June 2026 statement?
SQL: SELECT ROUND(SUM(t.amount), 2) AS total FROM transactions t JOIN statements s ON s.id = t.statement_id WHERE s.period_end >= DATE '2026-06-01' AND s.period_end < DATE '2026-07-01' AND t.category IN ('travel', 'transport') AND t.type IN ('purchase', 'refund')
Q: What did I spend at Costco in Q3 2025?
SQL: SELECT ROUND(SUM(amount), 2) AS total FROM transactions WHERE merchant ILIKE '%costco%' AND type IN ('purchase', 'refund') AND txn_date >= DATE '2025-07-01' AND txn_date < DATE '2025-10-01'
Q: Spend per month?
SQL: SELECT to_char(date_trunc('month', txn_date), 'YYYY-MM') AS month, ROUND(SUM(amount), 2) AS total FROM transactions WHERE type IN ('purchase', 'refund') GROUP BY 1 ORDER BY 1
"""

class UnsafeSQL(ValueError):
    pass


def check_sql(sql: str) -> str:
    """Allow exactly one SELECT (or WITH ... SELECT). Returns it without a trailing semicolon."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise UnsafeSQL("only one statement is allowed")
    if "--" in s or "/*" in s:
        raise UnsafeSQL("comments are not allowed")
    if not re.match(r"(?is)^(select|with)\b", s):
        raise UnsafeSQL("only SELECT queries are allowed")
    return s


def run_sql(conn: psycopg.Connection, sql: str) -> tuple[list[str], list[tuple]]:
    """Run in a read-only transaction with a timeout and a row cap (on top of the role's limits)."""
    safe = check_sql(sql)
    with conn.transaction():
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute(f"SET LOCAL statement_timeout = {TIMEOUT_MS}")
        cur = conn.execute(f"SELECT * FROM ({safe}) AS q LIMIT {ROW_CAP}")
        return [d.name for d in cur.description], cur.fetchall()


@dataclass
class Attempt:
    sql: str
    error: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    empty: bool = False  # ran fine but returned no rows


@dataclass
class Answer:
    question: str
    attempts: list[Attempt]
    text: str = ""
    unsupported: list[str] = field(default_factory=list)  # numbers in text not in the result
    seconds: float = 0.0

    @property
    def final(self) -> Attempt | None:
        """Last attempt that ran (prefer one with rows)."""
        ran = [a for a in self.attempts if a.error is None]
        with_rows = [a for a in ran if a.rows]
        return (with_rows or ran or [None])[-1]


def _chat(prompt: str, model: str, schema: dict | None = None, think: bool = False) -> str:
    resp = ollama.chat(model=model, think=think, format=schema,
                       options={"temperature": 0, "num_ctx": NUM_CTX},
                       messages=[{"role": "user", "content": prompt}])
    if resp.prompt_eval_count and resp.prompt_eval_count >= NUM_CTX:
        raise RuntimeError("prompt filled the context window; it was truncated")
    return resp.message.content


def categories(conn: psycopg.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT category FROM transactions WHERE category IS NOT NULL ORDER BY 1")]


def sql_prompt(question: str, cats: list[str], today: date, history: list[Attempt]) -> str:
    p = (SCHEMA_DOC.format(categories=", ".join(cats)) + "\n" + RULES.format(today=today)
         + "\n" + EXAMPLES.format(today=today)
         + f"\nWrite one PostgreSQL query that answers:\nQ: {question}\n")
    for a in history:
        if not (a.error or a.empty):
            continue
        problem = a.error or "it returned no rows"
        p += (f"\nA previous attempt failed.\nSQL: {a.sql}\nProblem: {problem}\n"
              "Fix the query (check column names, category values, date ranges and signs).\n")
    # A ```sql block, not JSON: inside a JSON string qwen3 reliably dropped the closing quote of
    # the last literal (... < DATE '2020-01-01"}), even with format= on, and retried identically.
    return p + "Reply with only the SQL in a ```sql code block."


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


def _fmt(v) -> str:
    if isinstance(v, Decimal):
        return f"{v:.2f}" if v.as_tuple().exponent < 0 else str(v)
    return str(v)


def result_text(cols: list[str], rows: list[tuple], limit: int = 50) -> str:
    lines = [" | ".join(cols)] + [" | ".join(_fmt(v) for v in r) for r in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more rows")
    return "\n".join(lines)


NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def numbers(text: str) -> set[Decimal]:
    out = set()
    for m in NUM_RE.findall(text):
        try:
            out.add(Decimal(m.replace("$", "").replace(",", "")))
        except ArithmeticError:
            pass
    return out


def unsupported_numbers(answer: str, question: str, cols: list[str], rows: list[tuple]) -> list[str]:
    """Numbers in the answer that appear in neither the result rows nor the question.
    Compared by value, so 1234.5 == 1,234.50, and sign-insensitive ("refunds of $5" vs -5.00)."""
    allowed = {abs(n) for n in numbers(question)}
    cells = [_fmt(v) for r in rows for v in r]
    for cell in cells:
        allowed |= {abs(n) for n in numbers(cell)}
    allowed |= {Decimal(len(rows))}
    if not rows or any(v is None for r in rows for v in r):
        allowed.add(Decimal(0))  # an empty/NULL SUM phrased as "$0" is a fair reading
    # ISO dates are checked whole (not as 2026, -03, -12) against dates in the result/question
    dates_ok = set(ISO_DATE.findall(question)) | {d for c in cells for d in ISO_DATE.findall(c)}
    bad = [d for d in ISO_DATE.findall(answer) if d not in dates_ok]
    answer = ISO_DATE.sub(" ", LIST_MARKER.sub("", answer))  # list numbering isn't a claim
    return sorted(bad + [str(n) for n in numbers(answer) if abs(n) not in allowed])


LIST_MARKER = re.compile(r"(?m)^\s*\d{1,3}[.)]\s+|(?<=\s)\d{1,3}\.\s+(?=\d{4}-\d{2}-\d{2})")
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


ANSWER_PROMPT = """Question: {question}
SQL that was run: {sql}
Result:
{result}

Answer the question in one or two sentences using ONLY the numbers in the result, copied
exactly (you may add a $ sign). Do not calculate anything yourself. If the result is empty or
NULL, say there is no data for that."""


def ask(conn: psycopg.Connection, question: str, *, model: str = MODEL, today: date | None = None,
        think: bool = False, phrase: bool = True) -> Answer:
    t0 = time.time()
    today = today or date.today()
    cats = categories(conn)
    ans = Answer(question, [])
    for _ in range(MAX_RETRIES + 1):
        raw = _chat(sql_prompt(question, cats, today, ans.attempts), model, None, think)
        att = Attempt(extract_sql(raw))
        ans.attempts.append(att)
        try:
            att.columns, att.rows = run_sql(conn, att.sql)
        except UnsafeSQL as e:
            att.error = f"refused: {e}"
            continue
        except psycopg.Error as e:
            att.error = (e.diag.message_primary or str(e)).strip()
            continue
        if att.rows or any(a.empty for a in ans.attempts[:-1]):
            break
        att.empty = True  # no rows may be a wrong filter: worth one retry, then accept it
    final = ans.final
    if phrase and final is not None:
        ans.text = _chat(ANSWER_PROMPT.format(question=question, sql=final.sql,
                                              result=result_text(final.columns, final.rows)), model).strip()
        ans.unsupported = unsupported_numbers(ans.text, question, final.columns, final.rows)
    ans.seconds = time.time() - t0
    return ans
