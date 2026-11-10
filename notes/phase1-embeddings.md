# Phase 1 — Embeddings and similarity, by hand

**Date:** 2026-11-08
**Script:** `scripts/phase1_similarity.py` (synthetic merchant strings only)
**Code:** `src/ragplay/embeddings.py`: `embed()` via Ollama, `cosine()` and `top_k()` in numpy

## Setup
25 merchants in 9 categories, each written 3 ways:
- **raw**: as printed on a statement: `SQ *BLUE BOTTLE COF 0423 OAKLAND CA`
- **clean**: normalised name: `Blue Bottle Coffee`
- **enriched**: name + short description: `Blue Bottle Coffee: specialty coffee roaster and cafe`

8 queries that avoid the category word itself ("latte", "filling up the car", "prescription
medicine", ...). Metric: precision@3, the share of the top 3 results in the right category.
Random picks score about 0.13.

## 1. Concept search: precision@3

| Model | dim | raw | clean | enriched |
|---|---|---|---|---|
| nomic-embed-text (137M) | 768 | 0.21 | 0.04 | 0.71 (0.88 without prefixes) |
| mxbai-embed-large (335M) | 1024 | **0.79** | 0.79 | **0.92** |

- **nomic on raw descriptors is close to random.** "latte" returned Uber, Luigi's and DoorDash.
- **Enriching the text helps more than anything else** (nomic 0.21 → 0.71–0.88). Phase 3's
  merchant normalisation and description step matters for search quality, not just for display.
- **A bigger model already knows brands:** mxbai gets 0.79 on raw strings.
- **Clean names alone are not enough for nomic** (0.04). "Kroger" carries no meaning for it
  without a description.
- **Task prefixes aren't a sure win.** nomic's documented `search_query:`/`search_document:`
  prefixes *hurt* on the enriched text (0.71 vs 0.88). Measure, don't assume.

## 2. Why raw fails: string shape beats meaning
Similarity to `STARBUCKS STORE 02609 SPRINGFIELD`:

| Compared with | nomic | mxbai |
|---|---|---|
| `PEET'S #1142 BERKELEY CA` | **0.722** | 0.607 |
| `SHELL OIL 00000003144 SPRINGFIELD ST` | **0.721** | 0.592 |
| `Starbucks` | 0.679 | **0.792** |
| `coffee shop` | 0.523 | **0.727** |

nomic sees "uppercase name + store number + city" and groups Starbucks with Shell. mxbai
groups it with "coffee shop".

## 3–4. Number and date blindness (the main lesson)
- **Amounts:** for nomic, `BLUE BOTTLE COFFEE $4.50` is closer to `SHELL OIL $4.50` (0.996) than to
  `BLUE BOTTLE COFFEE $4.75` (0.942). For mxbai, $4.50 vs $450.00 is still 0.924. Neither model
  treats $4 and $400 as different in any useful way.
- **Dates:** for mxbai, `2026-03-14` vs `2026-05-14` (a different month) scores **0.999**, *higher*
  than the next day (0.981). Embeddings can't tell March from May.

## 5. "Purchases over $100" as a vector query
Both models ranked `AMAZON.COM $100.00` (not over $100) first. Of the top 3, nomic got 1 right and
mxbai 1. A vector score has no idea what "over" means.

## Checkpoint: which questions need vectors and which need SQL

| Question shape | Use | Why |
|---|---|---|
| "latte", "taxi ride", "hardware stuff", "pet-related" | **Vectors** (on enriched text) | Meaning-based match across cryptic merchant names |
| "over $100", "in March", "last 30 days" | **SQL `WHERE`** | Embeddings are blind to numbers and dates |
| Totals, counts, averages, trends | **SQL aggregates** | Exact maths |
| "coffee in March over $5" | **Both in one query**: `WHERE` filters + `ORDER BY embedding <=> q` | Why pgvector in the same DB pays off |

## Decisions for later phases
- **Default embedding model → `mxbai-embed-large`.** Much better on raw text and best on
  enriched text. 1024-dim, 670 MB, fine on 16 GB. Phase 4 re-checks on real data.
- **Embed enriched text** (`"<clean merchant>: <description>"`), not raw descriptors. Phase 3
  generates the description once per unique merchant (cached), so it's cheap.
- **Keep dates and amounts out of the embedded text.** They add noise to the vector and belong
  in SQL columns.
