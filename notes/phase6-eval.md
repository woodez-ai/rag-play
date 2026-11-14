# Phase 6 — Evaluation (baseline before tuning)

**Date:** 2026-11-14
**Code:** `src/ragplay/evaluate.py`, `rag eval --set real|sample [--ids ...]`
**Data:** `eval/questions.jsonl` (real, 31 questions, gitignored), `eval/questions.sample.jsonl`
(synthetic, 16, committed). Runs saved to `eval/runs/` (gitignored).

## Design
- **Truth = reference SQL run at eval time**, so the question file holds no amounts and stays
  right if data is re-ingested or re-categorised. (Synthetic truth was validated against the
  generator JSON in Phase 5.)
- Four question types: **total** (number/month/name must be in the *answer text*, exact to the
  cent), **find** (precision/recall of returned transactions, by id or date + amount), **list**
  (precision/recall of names), **unanswerable** (must decline: empty/NULL/0 or "no data").
- Every metric is checked in code: answer exact, result exact, P/R, declined, SQL valid on the
  first try, faithful (every number in the answer is in the result or question).
- Each question has an **expected tool** (`sql`, `search`, `anomaly`) for Phase 7 routing.
- `EVAL_TODAY` is fixed (2026-11-14) so "last month" means the same thing on every run.
- The real run prints pass/fail only; answers with amounts live only in the gitignored run file.

## Scorer bugs found before trusting any number
The first sample run said 10/16; three "failures" were the scorer's fault:
1. A month in a sentence ("…was 2026-09, with…") didn't match: I checked `startswith`.
2. A grouped duplicate row (`Target, 2026-03-14, 91.12, n=2`) scored recall 0.5. Find is now
   set-based on (date, amount), so grouped and per-transaction answers both score 1.0.
3. A ±$0.01 tolerance accepted $674.00 for $673.99. Both sides are rounded to cents, so the
   tolerance only lets wrong answers through. Now exact, with Postgres-style half-up rounding.
9 scorer unit tests pin these down.

## Baseline: system = text-to-SQL for every question (Phase 5)

| Set | total | find | list | unanswerable | **all** | SQL valid 1st try | faithful | time |
|---|---|---|---|---|---|---|---|---|
| sample (synthetic) | 9/9 | 1/3 | 1/2 | 1/2 | **12/16** | 94% | 100% | 341 s |
| **real** | **18/19** | **2/7** (P 0.29, R 0.29) | 1/2 | 2/3 | **23/31** | 90% | 97% | 662 s |

By expected tool (real): sql 22/28, search 0/1, anomaly 1/2.

## Failure patterns (real), to beat in Phases 7–8
1. **"Show me / list …" gets aggregated (5 of 7 find questions).** Every one returns
   `SUM(amount)` instead of the transactions; all prompt examples are aggregates and nothing says
   "return rows". → Phase 7: route finds to `search_transactions` or a row-returning SQL path.
2. **Zero vs no data (t19).** "Interest in 2025" sums `type = 'interest'` transactions (there are
   none) → NULL → "There is no data". Truth is $0 from `statements.total_interest`.
3. **"Regularly" ignored (l01).** Lists every subscription merchant, not recurring ones
   (P 0.33, R 1.0). → Phase 8 `detect_recurring`.
4. **Doesn't decline (u03).** "What is my credit score?" → `SELECT 'Unknown'` →
   "Your credit score is $Unknown$." → Phase 7: the agent must be allowed to say "I can't".
5. 4 of 31 needed a retry (first-try SQL valid 90%); all recovered.

## How to use it
```
rag eval --set real            # ~11 min, prints pass/fail + summary, saves a run
rag eval --set sample          # synthetic (load ragplay_test first: scripts/phase3_eval_categories.py)
rag eval --ids f01,f02         # just some questions
```
Every later change is judged against this baseline.
