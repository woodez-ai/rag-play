# Phase 10 — Trace log and budgets

## What was built
- **Trace log** (`src/ragplay/trace.py`): every `rag ask`, `rag chat` and `rag li-ask` question is
  appended to `data/traces/YYYY-MM-DD.jsonl` (gitignored). An entry holds the question, the
  rewritten standalone question, the route (`router: sql`, `agent`, `agent: no tool`, `text2sql`),
  each tool call (args, SQL, columns, up to 50 rows, error, seconds), the answer, unsupported
  numbers and total time. `rag trace` lists recent entries and `rag trace --show N` prints one.
- **`rag ask` now uses the agent**, the same path as chat; `--sql-only` keeps the Phase 5 path.
- **Budgets** (`sql/006_budgets.sql`, `src/ragplay/budgets.py`): a monthly amount per category, or
  `all` for total spending. Spend uses the same definition as text-to-SQL (purchases + refunds by
  `txn_date`). Status is `over`, `at risk` (≥90% actual or projected) or `on track`.
  - CLI: `rag budget set|remove|list|status [--month YYYY-MM]`.
  - Agent tool `budget_status` (always returns every budget), routed away from SQL by `ANOMALY_RE`.
  - `rag review` includes budgets for the month.

## Design decisions
- **Budget maths is in code**, like every other number: the LLM only reads the rows.
- **Projection follows the data, not the calendar.** A month is "in progress" whenever the last
  transaction in it falls before month end. The first version tested "today is in this month", so
  October (real data ends Oct 21) showed "on track, 47% used". Now it projects straight-line from
  spend so far and reports `data_through`.
- **A month with no data is an error, not "on track".** "Am I on budget this month?" in November,
  with no November statement, used to report $0 spent. The tool now says there are no
  transactions for that month and when the data ends.

## What the trace log caught (first use)
1. The tool had a `category` parameter where `all` meant the total budget. Asked "which budgets did
   I go over?", the model passed `category='all'` (meaning *all my budgets*), got back only the
   total row and missed that dining was over. Fix: removed the parameter; the tool returns every
   budget. An argument whose name collides with a natural-language word is a trap for the model.
2. The "this month with no data" case above.

Neither showed up in the answer text alone; both were obvious from the recorded tool arguments and
rows. This is the case for tracing: the answer says *what*, the trace says *why*.

## Eval
- Sample set gained `b01` (groceries budget left, total) and `b02` (budgets gone over, list). The
  sample schema gets fixed budgets (dining 300, groceries 700, all 2500) in `ensure_sample`.
- Regression after the tool changes: real **32/32**, sample **18/18**, faithful 100%.
- 134 unit tests pass (`tests/test_budgets_trace.py` adds 9).

## Not done
- Real budgets are not set; that is the user's choice (`rag budget set dining 400`).
- Remaining stretch items: CSV/OFX import, MCP server, terminal charts.
