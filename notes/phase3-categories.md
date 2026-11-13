# Phase 3 — Enrichment: merchant normalisation and categories

**Date:** 2026-11-13
**Code:** `src/ragplay/categorize.py`, `sql/003_merchants.sql`, `rag categorize`, `rag categories …`
**Eval:** `scripts/phase3_eval_categories.py` (synthetic, ground truth from the generator)

## Pipeline
1. **Rule normalisation** (`merchant_key`): strip processor prefixes (`SQ *`, `TST*`), store
   numbers, phone numbers, reference codes and trailing province/state codes.
   Real data: **349 raw descriptors → 242 keys**. Synthetic: 198 → 24.
2. **Rules** for a few unambiguous high-volume names (Amazon, Netflix, Uber/Uber Eats, Starbucks,
   Tim Hortons, Costco…): 30 of the 242 real keys.
3. **LLM, once per unique key** (`qwen3:8b`, think off, temperature 0, JSON schema with the
   category as an **enum**): clean name, short description, category. 212 calls, ~4 s each.
4. **Cache** in `merchant_aliases` (key → merchant) and `merchant_categories` (merchant →
   category + description). Re-runs only send new keys. `source='manual'` always wins.
5. `transactions.merchant/category` are refreshed from the cache. Payments, fees and interest
   get their category from the transaction type and never go to the LLM.

Real result: **0 of 912 transactions uncategorised**; 242 keys → 166 merchants.

## What moved the numbers

| Change | Synthetic accuracy | Real txns in `other` |
|---|---|---|
| first prompt | 97.7% (CVS → groceries) | — |
| + "pharmacies = health" | 97.7% (Planet Fitness *Club Fees* → `fees`) | — |
| + `fees`/`payment` removed from the LLM's enum | 100% | **170 (19%)**, 31 merchants |
| + `description` before `category` in the schema, category definitions | 97.7% (gym → subscriptions; name lowercased) | — |
| + "gym memberships = health", drop "lowercase" | **100%** | **35 (4%)**, 11 merchants |

Lessons:
- **Field order in the JSON schema matters.** The model generates keys in schema order. With
  `category` first it committed before "thinking"; 31 real merchants went to `other` although
  their own description said "telecom service provider", "restaurant and bar", "web hosting".
  Writing the description first is cheap chain-of-thought.
- **Structure beats prompting.** Taking `fees`/`payment` out of the enum fixed "CLUB FEES" for
  good, where wording would only have made it less likely. Anything derivable from data
  (transaction type) shouldn't be an LLM choice.
- **Prompt changes have side effects.** Each fix moved another merchant ("lowercase" for the
  description leaked into the merchant name). Re-run the eval after every prompt edit.
- **Synthetic accuracy is necessary but not sufficient**: 100% synthetic still left 11 real
  merchants in `other`, one of them a telecom provider despite the prompt naming "phone,
  mobile" under utilities. That's what the human review step is for.

## Remaining `other` (real, masked)
Mostly genuinely ambiguous: a bank, payment processors, an investment fund, a car dealership, a
farm, a concession stand. One clear miss: a telecom provider (19 txns) → should be `utilities`.

## Human in the loop
- `rag categories review [--limit 50] [--source llm]`: most-used merchants first, shows raw
  descriptors; Enter confirms, a category name changes it, `q` stops. Every reviewed row becomes
  `manual`, so later runs (even `--redo-llm`) never change it.
- `rag categories set "<merchant>" utilities`, `rag categories list --category other`,
  `rag categories summary`.

## Safety
Merchant descriptors are untrusted. They go to the model only inside a constrained-output call
(schema + enum), so a hostile descriptor can at worst produce a wrong name/description, never a
new category or free-form output. The description is model-generated and will be embedded in
Phase 4. Treat it as untrusted text too.

## Checkpoint
Every transaction has a category ✅. **Spot-check of ~50 merchants: to be done by the owner**
with `rag categories review` (the real names were never shown to the assistant).
