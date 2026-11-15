# Phase 7 — The agent: tool-calling router

**Date:** 2026-11-15
**Code:** `src/ragplay/agent.py`, `store.find()`, `rag chat [-v]`, `rag eval --system agent|sql`

## Design
`qwen3:8b` (Ollama tool calling, thinking off) chooses per question:
- **`query_database()`**: no arguments. It runs Phase 5 text-to-SQL on the user's own
  (standalone) question. Used for totals, counts, averages, max, comparisons, rankings.
- **`find_transactions(merchant, category, date_from, date_to, min_amount, max_amount, text)`**:
  returns transaction rows, newest first. Plain filters; hybrid search only when `text` is given.
- **Decline**: no tool call when the data can't answer (credit score, other cards, out of range).
- Max 4 tool rounds; every number and ISO date in the answer must appear in some tool output, the
  question, or the facts the system prompt states (data range, today).
- **`rag chat`**: follow-ups are rewritten into standalone questions from the last 3 turns, then
  answered like any other question ("and in April?" → "How much did I spend on groceries in
  April 2026?").

## Results (same eval as Phase 6)

| | Phase 6 baseline (always SQL) | Phase 7 agent |
|---|---|---|
| **real** | 23/31 | **30/31** |
| ↳ totals | 18/19 | **19/19** |
| ↳ find (P / R) | 2/7 (0.29 / 0.29) | **7/7 (1.00 / 1.00)** |
| ↳ list | 1/2 | 1/2 |
| ↳ unanswerable | 2/3 | **3/3** |
| faithful (real) | 97% | 97% (one mistyped date, flagged) |
| tool accuracy (real) | — | 81% (sql 20/20, search 6/6, none 3/3 by outcome) |
| **sample** | 12/16 | **15/16** |
| time (real, 31 q) | 662 s | 844 s |

The one remaining failure in each set is an anomaly question ("which subscriptions do I pay for
**regularly**", "which subscriptions **went up** in price"), which is Phase 8's job. Tool accuracy
is below the pass rate because some questions pass via the other tool (e.g. "how many times was I
charged by Netflix" answered by counting `find_transactions` rows).

## What made it work: each fix came from reading a failure
1. **Paraphrasing changes meaning.** When `query_database` took a question argument, qwen3
   rewrote "on my March 2026 statement" as "in March 2026" (a different period), even when told to
   pass it word for word. **Fix: the tool takes no arguments** and always gets the user's question.
   The model still invents arguments for it; they're ignored.
2. **Stacked filters return nothing.** "Hardware store purchases" became `category='home' AND
   merchant='hardware store'` → 0 rows. Neither a tool-description rule nor a "try fewer filters"
   hint in the result made it retry. **Fix: deterministic relaxation in code** (drop `text`, then
   `merchant`, while a category remains).
3. **Bare category names aren't enough.** It mapped "hardware store" to `shopping`. **Fix: give the
   agent the same category definitions Phase 3 used to assign them.**
4. **Invented arguments to do maths.** For "largest purchase" it called
   `find_transactions(sort='amount', limit=1)`. Those arguments don't exist and were silently
   ignored, so it picked a max by reading 40 rows, and got it wrong. **Fix: unknown arguments are
   an error that points to `query_database`.** It now recovers on the next step.
5. **Row order matters when the model sees only 40 rows.** "Most recent McDonald's purchase" (71
   rows, oldest first) → wrong. **Fix: newest first.**
6. **Zero is not "no data".** The agent declined "interest charged in 2025" because the prompt only
   mentioned balances; then SQL summed interest *transactions* (none) → NULL. **Fix: the system
   prompt says statements hold fee and interest totals, and a text-to-SQL rule says to use
   `statements.total_interest` with `COALESCE(…, 0)`.**

## Faithfulness check: real catches and false alarms
- Real: `2026-03.12` (a mistyped date) and, on the real set, another date not in any tool output.
  Neither changes the scored rows, but both would mislead a reader.
- False alarms fixed in the checker, not the answers: list numbering ("1.", "2."), ISO dates split
  into numbers, facts from the system prompt (data range), and "$0" for an empty SUM.

## Infrastructure lessons (a laptop that sleeps)
Two eval runs died overnight when the Mac slept:
- A dead client left a session **idle in transaction for 76 min**, holding locks that blocked the
  test suite's `TRUNCATE` for 10 minutes. → Connections are **autocommit** (every write path
  already uses an explicit transaction), with **TCP keepalives** and
  **`idle_in_transaction_session_timeout` = 10 min**.
- One dropped connection failed every remaining question. → The eval **pings and reconnects**
  before each question.
- The sample eval shared `ragplay_test` with the tests, which truncate it. → Its own schema,
  **`ragplay_sample`**, loaded automatically on first use. A NULL truth now fails one question
  instead of crashing the run.
- Long runs use **`caffeinate -i`**.

## Not done
- `llama3.2:3b` comparison (plan: "is a 3B model good enough?"): needs a ~2 GB pull; optional.
- Anomaly tools (`detect_recurring`, `find_duplicates`, `spend_outliers`): Phase 8.
