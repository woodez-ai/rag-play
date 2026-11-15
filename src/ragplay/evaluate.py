"""Evaluation harness (`rag eval`). Measure first; judge every later change against a baseline.

Question file (JSONL), one object per line:
  {"id": "t01", "type": "total|find|list|unanswerable", "tool": "sql|search|anomaly",
   "question": "...", "ref_sql": "..."}
Truth is computed by running `ref_sql` at eval time, so the real question file contains no
amounts (it is still gitignored: questions mention merchants). ref_sql conventions:
  total -> one row; every value in it must be in the answer (numbers exact to the cent)
  find  -> rows of (id, txn_date, amount): the expected transactions
  list  -> rows whose first column is an expected name (merchant/category); may be empty
           ("none"). Scored on the names the ANSWER mentions.
  unanswerable -> no ref_sql; the system must say there's no data (or decline)

Metrics (all checked in code, never by an LLM):
  answer exact      total: the answer text contains every truth value
  result exact      total: the final SQL result's first row contains every truth value
  precision/recall  find: returned transactions (by id, or by matching date + amount);
                    list: names in the result vs expected names
  declined          unanswerable: empty/NULL/0 result, or a "no data" answer
  sql valid         first generated query ran without error
  faithful          every number in the answer is in the result or the question
  tool              expected tool: sql (calculation) | search (return rows) | none (decline)
                    | anomaly (Phase 8); tool accuracy = first tool used == expected
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import psycopg

from ragplay import text2sql

TOOL_LABEL = {"query_database": "sql", "find_transactions": "search", "detect_recurring": "anomaly",
              "find_duplicates": "anomaly", "spend_outliers": "anomaly"}
EVAL_TODAY = date(2026, 11, 14)  # fixed so "last month" means the same thing on every run
ROOT = Path(__file__).resolve().parents[2]
SETS = {"real": ROOT / "eval/questions.jsonl", "sample": ROOT / "eval/questions.sample.jsonl"}
RUNS = ROOT / "eval/runs"
CENT = Decimal("0.01")
NO_DATA = re.compile(r"\b(no (data|transactions?|records?|purchases?|spending|information|charges?)"
                     r"|not (available|found|recorded)|(don't|do not|doesn't|does not|cannot|can't) "
                     r"(have|find|answer|determine|track|contain)|there (is|are|were) no|none)\b", re.I)


@dataclass
class Result:
    id: str
    type: str
    tool: str
    question: str
    passed: bool = False
    answer_exact: bool | None = None
    result_exact: bool | None = None
    precision: float | None = None
    recall: float | None = None
    sql_valid: bool | None = False
    faithful: bool = True
    tools: str = ""        # tools actually used, in order (sql | search | none)
    tool_ok: bool = False  # first tool used == expected tool
    attempts: int = 0
    seconds: float = 0.0
    sql: str = ""
    answer: str = ""
    error: str = ""


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _norm(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, (int, float, Decimal)):
        return abs(Decimal(str(v))).quantize(CENT, ROUND_HALF_UP)  # like Postgres ROUND
    return str(v).strip().lower()


def _contains(values: set, truth) -> bool:
    """Exact to the cent. Both sides are rounded to cents, so a ±0.01 tolerance would only
    accept wrong answers ($674.00 for $673.99)."""
    t = _norm(truth)
    if t is None:  # reference returned NULL: nothing to match (a broken/empty dataset)
        return False
    if isinstance(t, Decimal):
        return t in values
    return any(isinstance(v, str) and t in v for v in values)  # also months: "2026-09" in "...2026-09-01"


def _answer_values(text: str) -> set:
    vals = {abs(n).quantize(CENT, ROUND_HALF_UP) for n in text2sql.numbers(text)}
    low = text.lower()
    vals.add(low)
    # months written out ("March 2026") also count as YYYY-MM
    for m in re.finditer(r"(january|february|march|april|may|june|july|august|september|october|"
                         r"november|december) (\d{4})", low):
        vals.add(f"{m[2]}-{datetime.strptime(m[1], '%B').month:02d}")
    return vals


def score_total(truth: list, final, answer: str) -> tuple[bool, bool]:
    row = {_norm(v) for v in final.rows[0]} if final and final.rows else set()
    result_ok = bool(truth) and all(_contains(row, t) for t in truth)
    answer_ok = bool(truth) and all(_contains(_answer_values(answer), t) for t in truth)
    return answer_ok, result_ok


def _row_keys(row: tuple) -> set[tuple]:
    dates = {v.date() if isinstance(v, datetime) else v for v in row if isinstance(v, date)}
    amts = {abs(Decimal(str(v))).quantize(CENT, ROUND_HALF_UP) for v in row
            if isinstance(v, (int, Decimal, float)) and not isinstance(v, bool)}
    return {(d, a) for d in dates for a in amts}


def score_find(expected: list[tuple], final) -> tuple[float, float]:
    """expected: (id, txn_date, amount). Set-based: by id if the result has an id column,
    else by (date, amount) key. precision = returned rows matching an expected item;
    recall = expected items (keys) that some row matches. A grouped row ("Target, 2026-03-14,
    91.12, n=2") therefore counts for both duplicates; one row per transaction also scores 1.0.
    Leniency: two expected transactions with the same date and amount are one key."""
    if not final or not final.rows:
        return (0.0 if expected else 1.0), (0.0 if expected else 1.0)
    cols = [c.lower() for c in final.columns]
    if "id" in cols:
        want = {e[0] for e in expected}
        got = [{row[cols.index("id")]} for row in final.rows]
    else:
        want = {(e[1], abs(Decimal(str(e[2]))).quantize(CENT, ROUND_HALF_UP)) for e in expected}
        got = [_row_keys(row) for row in final.rows]
    precise = sum(bool(g & want) for g in got)
    found = want & set().union(*got)
    return precise / len(got), (len(found) / len(want) if want else 1.0)


NEGATION = re.compile(r"\b(no|not|none|unchanged|didn't|did not|same|stayed|remained)\b")


CHANGE_Q = re.compile(r"\b(went up|go up|increase|decrease|chang|duplicate|unusual)", re.I)


def score_list_answer(expected: list[str], answer: str, known: list[str],
                      question: str = "") -> tuple[float, float]:
    """End to end, like totals: which known names (merchants, categories) does the ANSWER
    mention? precision = mentioned names that are expected; recall = expected names mentioned.
    An empty truth ("none went up") passes only if the answer names nothing.
    Needed because tools like detect_recurring return every series, so the rows alone can't
    say which ones the answer picked."""
    exp = {e.lower() for e in expected}
    # For "which went up / which are duplicates" questions, a line saying a name did NOT qualify
    # ("Apple: no price change") isn't a claim about it. For other questions the same words are
    # just detail ("Apple - monthly, no price change" still answers "which do I pay regularly").
    claims = answer.lower()
    if CHANGE_Q.search(question):
        claims = "\n".join(line for line in re.split(r"\n|(?<=[.;])\s+", claims)
                           if not NEGATION.search(line))
    mentioned = {n.lower() for n in known
                 if re.search(rf"(?<![\w']){re.escape(n.lower())}(?![\w'])", claims)}
    # a name that contains another mentioned name wins ("Uber Eats" over "Uber")
    mentioned = {m for m in mentioned if not any(m != o and m in o for o in mentioned)}
    hits = mentioned & exp
    precision = len(hits) / len(mentioned) if mentioned else (1.0 if not exp else 0.0)
    recall = len(hits) / len(exp) if exp else 1.0
    return precision, recall


def score_list(expected: list[str], final) -> tuple[float, float]:
    """Distinct names from the result's merchant (or category) column, else its first column,
    so transaction rows from find_transactions (id first) are scored by their merchants."""
    exp = [e.lower() for e in expected]
    got = []
    if final and final.rows:
        cols = [c.lower() for c in final.columns]
        i = next((cols.index(c) for c in ("merchant", "category") if c in cols), 0)
        got = list(dict.fromkeys(str(r[i]).lower() for r in final.rows))
    found = {e for e in exp if any(e in g or g in e for g in got)}
    good_rows = sum(any(e in g or g in e for e in exp) for g in got)
    return (good_rows / len(got) if got else (1.0 if not exp else 0.0)), (len(found) / len(exp) if exp else 1.0)


def declined(final, answer: str) -> bool:
    """Declined = said there's nothing, and did NOT state an amount. An empty result alone isn't
    enough: the LLM once phrased an empty result as "The total spent in 2019 was $2543.75"."""
    if any(abs(n) > 0 for n in text2sql.numbers(" ".join(re.findall(r"\$\s?[\d,]+(?:\.\d+)?", answer)))):
        return False
    empty = final is None or not final.rows or all(v in (None, 0) for v in final.rows[0])
    return empty or bool(NO_DATA.search(answer))


def ensure_sample(conn: psycopg.Connection, fake_dir: Path = ROOT / "data/fake") -> None:
    """Load the synthetic statements into the sample schema if it's empty (ingest, categorise
    with the LLM, embed). Its own schema: the test suite truncates ragplay_test."""
    from ragplay import categorize, ingest, store

    if conn.execute("SELECT count(*) FROM transactions").fetchone()[0]:
        return
    pdfs = sorted(fake_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"no synthetic PDFs in {fake_dir}; run scripts/make_fake_statements.py")
    for pdf in pdfs:
        r = ingest.ingest_file(conn, pdf)
        if r.status == "failed":
            raise RuntimeError(f"{pdf.name}: {r.errors}")
    categorize.categorize(conn)
    store.embed_transactions(conn)


def evaluate(conn: psycopg.Connection, questions: list[dict], *, model: str = text2sql.MODEL,
             system: str = "sql", progress=None, reconnect=None) -> list[Result]:
    """system: 'sql' = text-to-SQL for every question (Phase 5 baseline);
               'agent' = the Phase 7 tool-calling agent (it chooses the tool).
    reconnect: callable returning a fresh connection. The connection is pinged before each
    question and replaced if the server dropped it (a laptop that sleeps mid-run), so one
    dropped connection doesn't fail every remaining question."""
    from ragplay import agent

    out = []
    for q in questions:
        if reconnect:
            try:
                conn.execute("SELECT 1")
            except psycopg.OperationalError:
                conn = reconnect()
        r = Result(q["id"], q["type"], q.get("tool", "sql"), q["question"])
        t0 = time.time()
        try:
            ref_cols, ref_rows = (text2sql.run_sql(conn, q["ref_sql"]) if q.get("ref_sql") else ([], []))
            if system == "agent":
                ans = agent.ask(conn, q["question"], model=model, today=EVAL_TODAY)
                r.attempts = len(ans.calls)
                r.sql_valid = ans.sql_first_try if ans.sql_first_try is not None else None
                r.tools = ",".join(TOOL_LABEL.get(t, t) for t in ans.tools) or "none"
                r.sql = "; ".join(f"{c.name}({c.sql or c.arguments})" for c in ans.calls)
            else:
                ans = text2sql.ask(conn, q["question"], model=model, today=EVAL_TODAY)
                r.attempts = len(ans.attempts)
                r.sql_valid = bool(ans.attempts) and ans.attempts[0].error is None
                r.tools = "sql"
                r.sql = ans.final.sql if ans.final else (ans.attempts[-1].sql if ans.attempts else "")
        except Exception as e:  # a crash is a failed question, not a failed eval
            r.error, r.seconds = f"{type(e).__name__}: {e}", time.time() - t0
            out.append(r)
            if progress:
                progress(r)
            continue
        final = ans.final
        r.seconds, r.answer = ans.seconds, ans.text
        r.tool_ok = r.tools.split(",")[0] == r.tool
        r.faithful = not ans.unsupported
        if q["type"] == "total":
            truth = list(ref_rows[0]) if ref_rows else []
            r.answer_exact, r.result_exact = score_total(truth, final, ans.text)
            r.passed = r.answer_exact
        elif q["type"] == "find":
            r.precision, r.recall = score_find([tuple(x[:3]) for x in ref_rows], final)
            r.passed = r.precision == 1.0 and r.recall == 1.0
        elif q["type"] == "list":
            # compare like with like: category answers against category names, merchant answers
            # against merchant names ("subscriptions" in a merchant list is not a claimed item)
            cats = [x[0] for x in conn.execute(
                "SELECT DISTINCT category FROM transactions WHERE category IS NOT NULL")]
            expected = [x[0] for x in ref_rows]
            is_cat = bool(expected) and all(e in cats for e in expected)
            known = cats if is_cat else [x[0] for x in conn.execute("SELECT merchant FROM merchant_categories")]
            r.precision, r.recall = score_list_answer(expected, ans.text, known, q["question"])
            r.passed = r.precision == 1.0 and r.recall == 1.0
        elif q["type"] == "unanswerable":
            r.passed = declined(final, ans.text)
        else:
            raise ValueError(f"{q['id']}: unknown type {q['type']!r}")
        out.append(r)
        if progress:
            progress(r)
    return out


def summary(results: list[Result]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None
    by_type = {}
    for t in ("total", "find", "list", "unanswerable"):
        rs = [r for r in results if r.type == t]
        if rs:
            by_type[t] = {"n": len(rs), "passed": sum(r.passed for r in rs),
                          "precision": mean([r.precision for r in rs]),
                          "recall": mean([r.recall for r in rs]),
                          "result_exact": mean([r.result_exact for r in rs])}
    return {
        "n": len(results), "passed": sum(r.passed for r in results),
        "sql_valid": mean([r.sql_valid for r in results]),
        "faithful": mean([r.faithful for r in results]),
        "tool_accuracy": mean([r.tool_ok for r in results]),
        "retried": sum(r.attempts > 1 for r in results),
        "crashed": sum(bool(r.error) for r in results),
        "seconds": round(sum(r.seconds for r in results)),
        "by_type": by_type,
        "by_tool": {t: {"n": sum(r.tool == t for r in results),
                        "passed": sum(r.passed for r in results if r.tool == t)}
                    for t in sorted({r.tool for r in results})},
    }


def save(set_name: str, system: str, model: str, results: list[Result]) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}-{set_name}-{system}.json"
    path.write_text(json.dumps({"set": set_name, "system": system, "model": model,
                                "today": EVAL_TODAY.isoformat(), "summary": summary(results),
                                "results": [asdict(r) for r in results]}, indent=2, default=str))
    return path
