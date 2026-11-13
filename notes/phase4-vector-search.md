# Phase 4 — Vector search with pgvector (+ hybrid)

**Date:** 2026-11-13
**Code:** `src/ragplay/store.py`, `rag embed`, `rag search`, `sql/004`, `sql/005`
**Eval:** `scripts/phase4_eval_search.py` (synthetic and real; prints scores only)

## What's stored
One embedding per transaction per model in `transaction_embeddings`, made from the enriched
text `"<merchant>: <description>"` (Phase 3). Dates and amounts are never in the text: they're
columns and become `WHERE` filters (Phase 1: embeddings can't tell March from May).
- Identical texts share one embedding call: 912 real transactions → **167 distinct texts**.
- `content` column (migration 004) stores the embedded text, so a manual fix to a merchant's
  name or description is detected and re-embedded on the next `rag embed`.
- Both `mxbai-embed-large` (1024-d) and `nomic-embed-text` (768-d) are stored side by side;
  `model` picks one at query time.

## Search = one SQL query
`store.search(query, mode, date_from, date_to, min_amount, max_amount, category, k)`:
- **filters** are a CTE every method ranks within, so they're hard constraints (tested).
- **vector**: `ORDER BY embedding <=> :qvec` (cosine distance), exact scan.
- **text**: full-text on the merchant name (terms OR-ed) + trigram `word_similarity` (typos).
- **hybrid**: Reciprocal Rank Fusion of the three rank lists, `Σ 1/(60 + rank)`, in CTEs.

## Results (precision over min(10, relevant txns))

| Data | Queries | Model | vector | text | hybrid |
|---|---|---|---|---|---|
| synthetic | concept (10) | mxbai | 1.00 | 0.00 | 1.00 |
| synthetic | name (6, incl. typo) | mxbai | 0.83 | 1.00 | 0.92 |
| real | concept (10) | mxbai | **0.82** | 0.05 | 0.81 |
| real | name (6, incl. typo) | mxbai | **1.00** | 1.00 | **1.00** |
| real | concept | nomic | 0.76 | 0.05 | 0.75 |
| real | name | nomic | **0.00** | 1.00 | 0.92 |

Concept truth = the owner-reviewed Phase 3 category, which understates precision: e.g. all 4
"wrong" hits for "streaming service" are a merchant described as a *streaming entertainment
service* that's categorised `entertainment`. Real misses: "pharmacy" → an alcohol retailer,
"taxi ride" → a chicken restaurant, "phone bill" → a "digital billing service".

**mxbai stays the default.** nomic's vectors found none of the name queries on real data.

## Checkpoint (real data)
- **"coffee" + March 2026 filter**: all 50 hits in March; the top 13 (= every March coffee
  transaction) are all coffee. ✅
- **"hardware store purchases"**: vector-only ranks all 4 home-improvement transactions 1–4. ✅
  Hybrid pushes them to 3–6 (see below).
- Amount filter (`min_amount`) is respected on every hit. ✅

## What went wrong first, and why
1. **Hybrid was worse than vector-only** (synthetic concept 0.84 vs 0.94). Cause: the 001
   `search_tsv` indexed the *raw descriptor* with the `simple` config, so generic tokens and
   stopwords matched: "hardware **store**" → "STARBUCKS **STORE**", "filling up **the** car" →
   "**THE** HOME DEPOT". RRF counts any match as a vote. **Fix (migration 005):** full-text over
   the clean merchant name only, `english` config (stopwords out, stemming: "mcdonalds" ≡
   "McDonald's"). Hybrid then matched vector on concepts.
2. **The metric was wrong.** P@10 can't exceed n/10 for a merchant with n < 10 transactions
   (Wawa has 1). Normalising by min(10, n) moved real name precision from 0.62 to 1.00.
3. **The eval hard-coded an LLM-chosen name** ("Chipotle Mexican Grill"; the next run said
   "Chipotle"). Name truth is now resolved by pattern.

## RRF is a compromise, not a free win
- Typos: "starbuks" scores 0 by vector, 1.0 by trigram, 0.5 fused. Wrong vector hits interleave.
- Generic words: "hardware store purchases" ranks a merchant whose *name* contains "store" above
  the hardware stores.
- Weighted RRF (text 0.5 or 0.3) didn't fix it: +0.01 on real concepts, −0.09 on synthetic names.
  Kept equal weights.
- **Conclusion:** concept queries → vector; names and typos → text; mixed or unknown → hybrid.
  That choice belongs to the Phase 7 agent (`search_transactions(mode=...)`), not to one fused
  setting.

## Not done (optional)
HNSW index + `EXPLAIN ANALYZE`: at 912 rows an exact scan is instant and an approximate index
only adds recall risk. Worth revisiting only with much more data.
