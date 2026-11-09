# Phase 0 — Baseline: LLM reads the statement and does the maths

**Date:** 2026-11-08
**Setup:** `scripts/baseline.py` extracts one synthetic statement's text with pdfplumber
(`data/fake/acme_4821_2026-03.pdf`, 40 transactions, 2 pages), pastes it into `qwen3:8b`
(thinking on, temperature 0, `num_ctx` 16384) and asks 5 questions. Ground truth comes from the
generator's JSON sidecar.

## Results — 2/5 exact

| Question | Expected | qwen3:8b | Result | Time |
|---|---|---|---|---|
| Total spent on groceries | 673.99 | 944.85 | wrong (+270.86) | 292s |
| Total spent on dining (restaurants, delivery, coffee) | 351.96 | 351.96 | correct | 466s |
| Number of coffee shop purchases | 8 | 7 | wrong (missed one) | 542s |
| Total spent on gas stations | 201.66 | 528.86 | wrong (+327.20) | 195s |
| Single largest purchase | 291.21 | 291.21 | correct | 132s |

Total run time about 27 minutes on an M3 / 16 GB.

## Observations
- **Errors are large and look confident.** Over-counts of 40–160% suggest it pulled in the wrong
  lines (inclusion errors), not just slips in adding up.
- **Correct answers can't be told apart from wrong ones.** Each answer comes with a confident
  explanation. For money, that's worse than no answer.
- **Picking out the largest value worked**, because it needs no adding up across rows.
- **Very slow:** 2–9 minutes per question, almost all of it qwen3 "thinking" through every line.
- **Repeat runs are identical** at temperature 0 (the first two groceries runs both gave 944.85),
  so repeating questions measures nothing. Run each question once.
- **Gotcha found:** Ollama's default context is 4096 tokens and it **truncates silently**.
  Statement plus thinking needs more. The script now sets `num_ctx=16384` and fails if the
  prompt fills the window.

## Conclusion
This confirms the design rule: **the LLM never does arithmetic.** In the real system it only
turns the question into SQL, Postgres computes exact totals, and code checks that every number in
the answer appears in the query result. Phase 5/6 will rerun these same 5 questions through
text-to-SQL for a direct before/after comparison.
