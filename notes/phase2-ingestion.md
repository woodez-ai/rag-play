# Phase 2 — Ingestion: PDF → structured transactions

**Date:** 2026-11-10
**Code:** `src/ragplay/extract.py`, `parsers/{acme,visa}.py`, `validate.py`, `ingest.py`, `rag ingest`
**Scripts:** `scripts/layout_dump.py` (masked layout), `scripts/phase2_llm_extract.py` (LLM comparison)

## Result
`rag ingest cc_statements` loads **all 19 real statements (912 transactions); every one reconciles
to the cent**. Re-running skips all 19 (idempotent on `file_hash`). Checked again in SQL as
`rag_reader`: `previous_balance + Σ amount = closing_balance` for 19/19, and each statement's
previous balance equals the prior statement's closing balance, with no gaps between periods.

## Privacy: building a parser without seeing the data
The real parser was written from `scripts/layout_dump.py`, which keeps x/y positions and each
word's *shape* (`XXXX 99`, `$9,999.99`) and shows only a short list of generic labels
("Purchases", "Payment", month names). Column positions, date/amount formats and signs survive;
names, merchants and amounts don't. All later checks printed counts or masked shapes only.

## The real layout (from the masked dump)
- Two columns: transactions on the left (x < 380), the balance calculation on the right.
- Row: `MON DD  MON DD  description  $amount`, then a 23-digit reference line, and sometimes a
  `Foreign Currency - USD … Exchange rate …` line or a wrapped second description line.
- Header `STATEMENT FROM SEP 22 TO OCT 21, 2026`; the start year appears **only** across a year end.
- A remittance slip at the bottom of page 1 repeats the card number and NEW BALANCE.

## Reconciliation earned its keep
The first run reconciled 17/19. The two failures were real parser bugs that would otherwise have
gone silently into the database:
1. **Dec→Jan header** prints both years (`FROM DEC 22, 2025 TO JAN 21, 2026`). The regex expected one.
2. **A merchant with "FEE" in its name** was typed as a bank fee. The statement counts it under
   Purchases, so the fee and purchase totals were both off. Rule now: bank-generated charges
   *start or end* with the keyword; merchant descriptors end with a city/province.

Checks (all to the cent): summary arithmetic, payments & credits, purchases, fees, interest,
balance roll-forward, every post date inside the period. A failed statement inserts nothing (one
DB transaction per statement; tested by failing after the statement row is written).

## LLM extraction vs the deterministic parser
Each page sent to a local model with a JSON schema (Ollama `format=`), temperature 0. Reference =
parser rows (which reconcile). One synthetic statement + three real ones (the "FEE" month, the
Dec→Jan month, the latest month).

| Statement | Mode | Rows (parser / LLM) | Exact row | Date + amount | Reconciles | Time |
|---|---|---|---|---|---|---|
| synthetic Dec→Jan | text | 46 / 46 | 1.00 | 1.00 | yes | 223 s |
| synthetic Dec→Jan | vision | 46 / 46 | 0.96 | 1.00 | yes | 362 s |
| real May 2025 | text | 43 / 43 | 0.86 | 0.98 | **no** | 322 s |
| real May 2025 | vision | 43 / 43 | 0.91 | 1.00 | yes | 507 s |
| real Jan 2026 | text | 41 / 40 | 0.93 | 0.98 | **no** | 360 s |
| real Jan 2026 | vision | 41 / 41 | 0.39 | 0.41 | **no** | 531 s |
| real Oct 2026 | text | 34 / 34 | 0.94 | 1.00 | yes | 313 s |
| real Oct 2026 | vision | 34 / 34 | 0.41 | 0.50 | **no** | 495 s |
| **parser, all 19 real** | — | 912 | — | — | **19/19** | **7 s total** |

Text = `qwen3:8b` on `extract_text()` (thinking off). Vision = `qwen3-vl:8b-instruct` on a 150-dpi page image.

### What went wrong, by kind
- **Text: one wrong date/amount (May), one dropped row (Jan).** Either breaks reconcile. A
  dropped row is invisible without totals to check against.
- **Text: descriptions truncated.** Most "exact" misses dropped the trailing city/province, a
  domain, or the wrapped second line. In one row the model **pulled in text from the right-hand
  balance panel**: `extract_text()` interleaves the two columns line by line.
- **Vision Jan: 24 January rows dated December 2025.** The amounts were right; the year-end
  reasoning wasn't. This is the worst kind of error for "spend by month" questions: totals look
  plausible. Only the post-date-in-period check caught it.
- **Vision Oct: amounts misread** on ~17 of 34 rows (same months, different amounts).
- **Synthetic data flattered the LLM.** Clean single-column layout → 100%. Real layouts → 1/3
  reconcile per mode. Test on real layouts before trusting an extraction method.

### Ollama structured-output gotchas
- A schema field's `description` ("YYYY-MM-DD") is only a hint: qwen3 copied `01/07` as printed.
  A `pattern` is enforced by constrained decoding.
- Ollama 0.20 rejects `\d` in a `pattern` with a misleading "failed to load model vocabulary
  required for format" error. `[0-9]` works.

## Conclusions
- **Deterministic parser + reconciliation is the ingest path.** Exact, ~0.4 s per statement, and
  it fails loudly. The LLM is ~2,000× slower and failed to reconcile 4 of the 6 real runs.
- **Use the LLM only as a fallback for a layout with no parser, and never without reconcile.**
  Reconcile turns a silent LLM error into a loud failure; that's the precondition for using it.
- **Vision is a last resort** (scanned PDFs). It was better than text on the one statement where
  text interleaved columns, and much worse on the other two.
- Not built: an automatic vision fallback inside `rag ingest`. All 19 statements have a text layer
  and parse, so there's nothing to fall back for yet.
