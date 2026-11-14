"""Evaluation harness (`rag eval`). Measure first; judge every later change against a baseline.

Question file (JSONL), one object per line:
  {"id": "t01", "type": "total|find|list|unanswerable", "tool": "sql|search|anomaly",
   "question": "...", "ref_sql": "..."}
Truth is computed by running `ref_sql` at eval time, so the real question file contains no
amounts (it is still gitignored: questions mention merchants). ref_sql conventions:
  total -> one row; every value in it must be in the answer (numbers exact to the cent)
  find  -> rows of (id, txn_date, amount): the expected transactions
  list  -> rows whose first column is an expected name (merchant/category)
  unanswerable -> no ref_sql; the system must say there's no data (or decline)

Metrics (all checked in code, never by an LLM):
  answer exact      total: the answer text contains every truth value
  result exact      total: the final SQL result's first row contains every truth value
  precision/recall  find: returned transactions (by id, or by matching date + amount);
                    list: names in the result vs expected names
  declined          unanswerable: empty/NULL/0 result, or a "no data" answer
  sql valid         first generated query ran without error
  faithful          every number in the answer is in the result or the question
  tool              the question's expected tool (scored once Phase 7 routes between tools)
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
    sql_valid: bool = False
    faithful: bool = True
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


def score_list(expected: list[str], final) -> tuple[float, float]:
    exp = [e.lower() for e in expected]
    got = [str(r[0]).lower() for r in final.rows] if final and final.rows else []
    found = {e for e in exp if any(e in g or g in e for g in got)}
    good_rows = sum(any(e in g or g in e for e in exp) for g in got)
    return (good_rows / len(got) if got else (1.0 if not exp else 0.0)), (len(found) / len(exp) if exp else 1.0)


def declined(final, answer: str) -> bool:
    empty = final is None or not final.rows or all(v in (None, 0) for v in final.rows[0])
    return empty or bool(NO_DATA.search(answer))


def evaluate(conn: psycopg.Connection, questions: list[dict], *, model: str = text2sql.MODEL,
             progress=None) -> list[Result]:
    """System under test: text-to-SQL for every question (the Phase 5 'always SQL' baseline)."""
    out = []
    for q in questions:
        r = Result(q["id"], q["type"], q.get("tool", "sql"), q["question"])
        ref_cols, ref_rows = (text2sql.run_sql(conn, q["ref_sql"]) if q.get("ref_sql") else ([], []))
        t0 = time.time()
        try:
            ans = text2sql.ask(conn, q["question"], model=model, today=EVAL_TODAY)
        except Exception as e:  # a crash is a failed question, not a failed eval
            r.error, r.seconds = f"{type(e).__name__}: {e}", time.time() - t0
            out.append(r)
            continue
        final = ans.final
        r.attempts, r.seconds, r.answer = len(ans.attempts), ans.seconds, ans.text
        r.sql = final.sql if final else (ans.attempts[-1].sql if ans.attempts else "")
        r.sql_valid = bool(ans.attempts) and ans.attempts[0].error is None
        r.faithful = not ans.unsupported
        if q["type"] == "total":
            truth = list(ref_rows[0]) if ref_rows else []
            r.answer_exact, r.result_exact = score_total(truth, final, ans.text)
            r.passed = r.answer_exact
        elif q["type"] == "find":
            r.precision, r.recall = score_find([tuple(x[:3]) for x in ref_rows], final)
            r.passed = r.precision == 1.0 and r.recall == 1.0
        elif q["type"] == "list":
            r.precision, r.recall = score_list([x[0] for x in ref_rows], final)
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
