# Phase 8 — Anomalies and subscriptions

**Date:** 2026-11-15
**Code:** `src/ragplay/anomalies.py`; agent tools `detect_recurring`, `find_duplicates`,
`spend_outliers`; `rag recurring | duplicates | outliers | review`

## Principle: compute in code, let the LLM explain
- **`detect_recurring`**: per merchant, chain charges that come 26–35 days (monthly) or 330–400
  days (yearly) after the previous one at a similar amount (±5%, FX drift). A price change splits
  a chain; the next chain, starting one cadence later at 0.8–1.6× the price, is merged back and
  recorded as a change (even a single new charge, if it's the latest). Each series gets current
  amount, price changes, active, and next expected date.
- **`find_duplicates`**: same merchant and amount within N days (default same day).
- **`spend_outliers`**: more than 3× the category median **and** in the top 2% of all purchases.
- **`monthly_review`**: facts for one month (total vs 3-month average, biggest category changes,
  new merchants, new recurring charges, price changes, missing recurring charges, duplicates,
  outliers). `rag review` has the LLM write them up and flags any number not in the facts.

## Tuning on planted truth, checked on real data
The generator plants 5 monthly charges, a Netflix increase (15.49 → 17.99) and one duplicate.
- First version: all 5 found plus **16 false series** (Starbucks, Safeway, Trader Joe's with "3
  price changes", two gas fill-ups a year apart). Busy merchants have so many similar charges that
  a chance monthly chain is easy. Guards added: narrower monthly window; busy merchants
  (>1.5 charges/month) need ±1% amounts **and** a stable billing day (gap spread ≤ 5 days);
  yearly only at merchants with ≤ 4 charges; ≤ 2 price changes, at 0.8–1.6×.
  → **exactly the 5, with the Netflix change on the right date** (7 tests pin this).
- Real data (counts only): 19 series, 6 active, 1 increase. The detector skipped one merchant the
  eval called a "regular subscription". Its gaps (119, 56, 54, 23, 1, 4, 15, 202 days) and amounts
  (0.6×–3.3× median) show irregular in-app purchases, so the **reference** was wrong. It now also
  requires stable amounts (max ≤ 1.25 × min), changed *after* seeing the detector, with this
  evidence.
- Outliers, measured on 891 real purchases: robust z on raw amounts flagged **87**; on log amounts
  **57**; "new merchant, big first charge" added ~28 ordinary travel/shopping. The simple rule
  (> 3× category median and top 2% overall) flags **17 (~1/month)**, 0 on synthetic. The
  statistics were removed.
- Duplicates on real: 1 same-day pair vs 11 within a day and 30 within 3 days (the same coffee on
  consecutive days) → default same day.

## The agent got worse first: more tools, worse routing

| | Phase 7 | + 3 anomaly tools | + calculation router & fixes |
|---|---|---|---|
| real | 30/31 | 27/32 | **32/32** |
| sample | 15/16 | 11/16 | **16/16** |
| faithful (real / sample) | 97% / 100% | 97% / 94% | **100% / 100%** |

1. **With 5 tools, qwen3 answered "how much / how many / largest" with `find_transactions` and
   did the arithmetic itself** (a count over the wrong period, a max by reading rows with
   `category='all'`, a total off by $30), despite "never calculate" in the prompt. → A
   **deterministic router**: questions matching calculation words go straight to text-to-SQL
   (LLM fallback if no working query). The LLM still routes show/list, subscriptions, duplicates,
   unusual purchases and declines.
2. **An empty result was phrased as a number.** "How much did I spend in 2019?" → correct SQL, 0
   rows → the phrasing LLM wrote "**The total spent in 2019 was $2543.75**". And the eval *passed*
   it, because `declined()` only looked at the empty result. → Empty/NULL results are never sent to
   the LLM (fixed "no data" text), and a "decline" that states an amount fails.
3. **A blank cell read as an increase.** "Which subscriptions went up?" → the tool said no change
   (blank `price_change`), the LLM said one went up. No number, so the faithfulness check couldn't
   catch it. → "no change" is explicit, and `price_increased_only` makes the tool return exactly
   the answer set (none → no rows).

## Scorer fixes (list questions now scored on the answer text, like totals)
- Tools like `detect_recurring` return every series, so rows can't show which ones the answer
  picked. List answers are scored by the known names they mention.
- Like with like: merchant answers against merchant names ("subscriptions" is a category word,
  not an extra item).
- Negation only for change questions: "Apple: no price change" excludes Apple from "which went
  up?" but still counts for "which do I pay regularly?".

## Infrastructure
- `caffeinate -i` doesn't stop a **closed lid** from sleeping a MacBook. One run took 9.5 h of
  sleep/dark-wake; the eval reconnected and completed, but timings were meaningless. Keep the lid
  open for long runs.

## Checkpoint (owner)
`rag recurring --all` on real data: does it find the subscriptions you know you have, and one
you'd forgotten? (The assistant only saw counts: 19 series, 6 active, 1 price increase.)
