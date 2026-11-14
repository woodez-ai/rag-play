# Phase 5 — Text-to-SQL

**Date:** 2026-11-14
**Code:** `src/ragplay/text2sql.py`, `rag ask "..." [--sql] [--think]`
**Eval:** `scripts/phase5_eval_sql.py` (synthetic: truth from generator JSON; real: hand-written
reference SQL, pass/fail only)

## How it works
prompt (schema doc, live category list, conventions, 5 examples, today's date) → `qwen3:8b`
(thinking off, temperature 0) returns a ```sql block → `check_sql` (one SELECT/WITH, no
comments) → run in a READ ONLY transaction with a 5 s timeout and a 200-row cap, as `rag_reader`
→ on error or no rows, feed the problem back and retry (max 2) → `qwen3:8b` phrases the answer
from the rows → code flags any number in the answer that isn't in the result.

## Results

| | Phase 0 (LLM reads statement) | Phase 5 (LLM writes SQL) |
|---|---|---|
| 5 baseline questions (synthetic, Mar 2026 statement) | **2/5**, errors up to +$327 | **5/5** exact |
| Time per question | 2–9 min | ~14 s (SQL + phrasing) |
| 12 synthetic questions (whole history, incl. "2019?") | — | **12/12**, 165 s total |
| 11 real questions (reference SQL) | — | **11/11**, 2 recovered by retry |
| Answers with numbers not in the result | — | 0 |

Thinking on: still running after 72 minutes on the 12 synthetic questions (vs 165 s off), and
off was already 12/12. Stopped; thinking isn't worth it for this prompt size.

## How we got from 6/12 to 12/12 (synthetic, first run → final)
1. **Conflicting rules** ("use txn_date for time" vs "on my March statement = period_end"). The
   model applied both, so it filtered the statement *and* March and lost the Feb half → 3 baseline
   failures. Fix: the statement rule is an explicit exception tied to the word "statement".
2. **Kinds of business are categories.** It used `merchant ILIKE '%gas%'` for "gas stations" and
   only `dining` for "restaurants, delivery and coffee shops". Fix: one rule (category / IN).
3. **"Charged by Netflix" → `type = 'fee'`.** Fix: "charges = type 'purchase'".
4. **JSON output broke SQL.** Inside `{"sql": "..."}` qwen3 consistently dropped the closing quote
   of the last literal (`< DATE '2020-01-01"}`), with or without `format=`. At temperature 0 the
   retry produced the same string twice. Fix: ask for a ```sql code block. 0 retries after that.
5. **Examples beat rules.** After fix 1 the model used statement periods for "Q4 2025" too, even
   with a rule saying not to. A rule rewrite didn't fix it; adding one example
   (merchant + quarter on `txn_date`) did.

## Safety layers (tested)
- Server: `rag_reader` has SELECT only; role-level read-only + 5 s timeout.
- Client: `run_sql` opens a READ ONLY transaction with `SET LOCAL statement_timeout`, even on an
  owner connection (`setval()` hidden in a SELECT is refused). Row cap 200.
- `check_sql`: one statement, SELECT/WITH only, no comments. A writable CTE passes this check,
  but Postgres refuses it (nested by the row-cap wrapper, and read-only anyway).
- Faithfulness: numbers in the phrased answer must come from the result or the question.

## Limits / next
- 23 questions is a smoke test, not an eval. Phase 6 builds the real one (25–40 questions,
  unanswerables, tool-selection accuracy) and should judge every later change.
- No "find" questions here ("show coffee charges in March"): that's search (Phase 4); routing
  between the two is Phase 7.
