# RAG Learning Plan — Credit Card Statement Analyst (Local Ollama)

## Goal
Build a local system that stores my **credit card statements (text PDFs)** and lets a **local Ollama
LLM** answer questions about them:
- **Spending totals & trends** — "How much did I spend on dining in Q2?", "Which month was highest?"
- **Find transactions** — "When did I last pay Netflix?", "Show coffee shop charges in March"
- **Anomalies & subscriptions** — recurring charges, price increases, duplicates, unusual spend

Scope: 1–2 cards, about 1 year of statements (a few dozen PDFs).

This is mainly a learning exercise in RAG design. Approach: **build each piece from scratch first**,
then **rebuild with a framework (LlamaIndex)** and compare. The interface is a **CLI**.

---

## The key design insight: this is not plain RAG
Plain RAG retrieves the top-k chunks that look similar to the question and has the LLM read them.
That works for "what does the document say?" questions, but **fails for most money questions**:

| Question type | Plain RAG | What works |
|---|---|---|
| "Total dining spend in Q2" | Gets ~5 chunks, misses most transactions, LLM adds badly | **SQL `SUM` over a transactions table** |
| "Month with highest spend" | Can't see all months at once | **SQL `GROUP BY`** |
| "Coffee shop charges" | Works OK (semantic match on merchant names) | **Vector search** (pgvector) over transactions, or SQL after categorising |
| "Netflix price increases" | Unreliable | **Code/SQL** comparing amounts over time |

So the architecture is a **hybrid**, which is the main pattern to learn here:

```
             ┌──────────── INGEST (deterministic) ────────────┐
PDF ──► text extraction ──► transaction parser ──► validate vs statement totals
                                     │
                                     ▼
            ┌──────── Remote PostgreSQL 17 (LAN, TLS) ────────┐
            │ relational: statements, transactions, categories│
            │ pgvector:   transaction + merchant embeddings   │
            │ full-text:  tsvector + pg_trgm on merchants     │
            └─────────────────────────────────────────────────┘
                                     ▲
             ┌───────────────── QUERY (LLM agent) ────────────┐
Question ──► qwen3:8b with tools: run_sql | search_transactions | detect_recurring
             ──► answer with exact numbers + the transactions it used
```

Rule: **the LLM never does arithmetic.** It writes SQL or calls tools; the database computes numbers.

One Postgres database holds both the structured data and the vectors. Semantic search, metadata
filters and SQL aggregation can therefore run in **one query** (e.g. `WHERE txn_date >= ... ORDER BY
embedding <=> $1`), with no syncing between two stores.

---

## Privacy and data handling (decide before any data touches the repo)
Statements are sensitive financial data. Models run locally (Ollama); the data is stored in a
**PostgreSQL server on the home LAN**, so it leaves the laptop but not the network.
- `cc_statements/` (real PDFs) and `data/` are **gitignored from Phase 0**, as are `*.pdf` in general.
- DB connection details live in `.env` (`DB_HOST`, `DB`, `DB_USER`, `DB_PASSWORD`), which is gitignored. Load with `python-dotenv`; never log the connection string.
- Connect with `sslmode=require` (the server supports TLS 1.3). Don't expose the DB port outside the LAN.
- Never commit statements, extracted text, DB dumps, eval answers or logs with real transactions.
- **Mask card numbers** during ingest (keep last 4 digits only); drop name and address lines.
- Two roles: `rag_user` (owner; ingest and schema changes) and **`rag_reader`** (read-only: `SELECT` only on the app tables). The SQL tool connects as `rag_reader`, with `default_transaction_read_only = on` and a `statement_timeout`, so a bad or hostile generated query can't change data or hang the server. (Created 2026-11-08: role settings `statement_timeout = 5s`, `search_path = ragplay, public`; `SELECT` on current and future `ragplay` tables via default privileges. Credentials in `.env` as `DB_READER_USER` / `DB_READER_PASSWORD`.)
- Treat statement text as untrusted input (merchant descriptions end up in prompts).
- Eval questions (`eval/questions.jsonl`) contain real amounts, so keep them gitignored too. Commit only a **synthetic sample** for tests.
- Use a **synthetic statement generator** (`scripts/make_fake_statements.py`) for development, tests and anything shared.

---

## Environment (as checked 2026-11-07)
| Item | Value |
|---|---|
| Machine | Apple M3, 16 GB RAM |
| Ollama | `qwen3:8b` (chat + tool calling), `qwen3-vl:8b-instruct` (vision, if a PDF won't parse) |
| Python | 3.14 system; **use 3.12 in the venv** (some ML/LlamaIndex wheels lag new Pythons) |
| Package mgr | `uv` |
| Database | Remote PostgreSQL 17.11 on the LAN (`db01`), TLS 1.3; `rag_user` owns the database. Extensions `vector` 0.8.7 and `pg_trgm` 1.6 enabled (in `public`). Schema `ragplay` created. `pg_hba` allows only the LAN (scram-sha-256) |
| DB client | `psql` (Homebrew libpq) |

### Models to pull
| Role | Model | Why |
|---|---|---|
| Chat / tool calling / SQL | `qwen3:8b` (have it) | Good tool calling and text-to-SQL at 8B |
| Embeddings (default) | `mxbai-embed-large` (have it) | Won Phase 1: P@3 0.79 on raw descriptors, 0.92 on enriched text |
| Embeddings (compare) | `nomic-embed-text` (have it), `qwen3-embedding` | nomic was near-random on raw descriptors (0.21) |
| Small chat (compare) | `llama3.2:3b` | Shows how model size affects SQL and tool-use accuracy |

---

## Project layout
```
rag-play/
  plan.md
  pyproject.toml              # uv-managed, python 3.12
  .env                        # GITIGNORED: DB connection details
  cc_statements/              # GITIGNORED: real statement PDFs (Visa ...2423, Apr 2025 – Oct 2026)
  data/                       # GITIGNORED: data/fake/ synthetic statements + ground-truth JSON
  sql/                        # schema migrations (001_schema.sql, 002_roles.sql, ...)
  eval/questions.sample.jsonl # synthetic eval set (committed)
  eval/questions.jsonl        # real eval set (gitignored)
  scripts/make_fake_statements.py
  src/ragplay/
    extract.py                # PDF -> text/words with positions (pdfplumber)
    parsers/                  # one parser per issuer layout -> Transaction rows
    validate.py               # reconcile parsed rows vs statement summary totals
    db.py                     # psycopg connections from .env (owner + read-only), migrations
    categorize.py             # merchant normalisation + LLM categorisation (cached)
    embeddings.py / store.py  # Ollama embeddings + pgvector
    tools.py                  # run_sql, search_transactions, detect_recurring, ...
    agent.py                  # tool-calling loop
    cli.py                    # rag ingest | ask | chat | eval
  src/ragplay_llamaindex/     # framework rebuild
  notes/                      # per-phase learning notes
```

### Data model (PostgreSQL, schema `ragplay`)
```
statements(id, card_last4, issuer, period_start date, period_end date, closing_balance numeric(12,2),
           total_purchases, total_payments, total_fees, total_interest numeric(12,2),
           source_file, file_hash text UNIQUE)
transactions(id, statement_id, txn_date date, post_date date, description_raw, merchant,
             amount numeric(12,2), type /* purchase|payment|refund|fee|interest */, category, card_last4,
             search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', merchant || ' ' || description_raw)) STORED)
merchant_categories(merchant PRIMARY KEY, category, source /* rule|llm|manual */)
transaction_embeddings(transaction_id, model text, embedding vector, PRIMARY KEY (transaction_id, model))
```
- Money is `numeric(12,2)`, never float, so sums are exact to the cent.
- Embeddings are keyed by `model` so several embedding models can be compared side by side (Phase 4).
  A few thousand rows is small enough for exact (sequential) search; add an HNSW index per model
  (on `embedding::vector(768)` with `WHERE model = ...`) only as an indexing exercise.
- Indexes: `transactions(txn_date)`, `(category)`, GIN on `search_tsv`, GIN `gin_trgm_ops` on `merchant`.
- A separate schema `ragplay_test` holds synthetic data for tests and the sample eval, so tests never touch real data.

---

## Phase 0 — Setup, privacy guardrails, baseline
**Concept:** Why grounding is needed. Safe handling of data.
- `uv init`, `uv venv --python 3.12`, activate, `uv add ollama "psycopg[binary]" pgvector python-dotenv pdfplumber typer rich`.
- Add `data/`, `.env` and `eval/questions.jsonl` to `.gitignore` **before** copying statements in.
- `ollama pull nomic-embed-text`.
- Database: extensions, `ragplay` schema and `rag_reader` role are already set up (2026-11-08). Run `sql/001_schema.sql` as `rag_user` and check `rag db check` connects as both roles over TLS.
- Baseline: paste one statement's text into `qwen3:8b` and ask "total spent on groceries?". Check against a hand calculation. Expect arithmetic and omission errors. This is why the LLM won't do maths.
- **Checkpoint:** guardrails in place; baseline errors recorded.
- ✅ **Done 2026-11-08.** `rag db migrate` / `rag db check`, synthetic generator (`data/fake/`), 14 tests passing. Baseline: **qwen3:8b got 2/5 exact**, with errors up to +$327 and 2–9 min per question → [notes/phase0-baseline.md](notes/phase0-baseline.md).

## Phase 1 — Embeddings and similarity, by hand
**Concept:** Text → vector; where semantic search helps and where it fails.
- Embed merchant strings (`"SQ *BLUE BOTTLE COF"`, `"STARBUCKS #1234"`, `"SHELL OIL 5739"`) and compute cosine similarity with numpy.
- See that "coffee" lands near coffee shops, but embeddings can't tell $4 from $400, or March from May.
- **Checkpoint:** you can explain which questions need vectors and which need SQL.
- ✅ **Done 2026-11-08** → [notes/phase1-embeddings.md](notes/phase1-embeddings.md). Raw descriptors: nomic P@3 0.21 (near random), mxbai 0.79. Enriched `"name: description"` text: up to 0.92. Both models are blind to amounts and dates (`2026-03-14` vs `2026-05-14` = 0.999). Default embedding model is now mxbai; embed enriched text, not raw.

## Phase 2 — Ingestion: PDF → structured transactions
**Concept:** Structured extraction is the most important (and least glamorous) part of the pipeline.
- Extract text with positions using `pdfplumber`. Look at the raw output of each issuer's layout.
- Write a **deterministic parser** per issuer (regex/column positions) → `Transaction` rows. Handle multi-line descriptions, credits shown as negatives or `CR`, and year rollover in Dec/Jan statements.
- **LLM-extraction comparison:** have `qwen3:8b` extract the same page using **structured output** (JSON schema via Ollama `format=`). Compare its accuracy with the regex parser. This shows when to use an LLM for extraction and when not to.
- Fallback for pages that won't parse: `qwen3-vl` on a page image.
- **Validation:** sum parsed purchases, payments and fees, and reconcile against the statement's summary box. Ingest fails loudly if they don't match.
- Idempotent ingest keyed on `file_hash` (`UNIQUE` constraint); each statement loads in **one transaction**, so a failed reconcile leaves nothing behind.
- **Checkpoint:** `rag ingest cc_statements` loads all statements; every statement reconciles to the cent.
- ✅ **Done 2026-11-10** → [notes/phase2-ingestion.md](notes/phase2-ingestion.md). 19/19 real statements (912 txns) reconcile to the cent and chain balance-to-balance; idempotent re-run. Parser written from a masked layout dump (no real content seen). Reconcile caught 2 real parser bugs (Dec→Jan header, merchant named "…FEE…"). LLM extraction reconciled only 2/6 real runs (dropped rows, wrong amounts, vision dated Jan rows as Dec) and is ~2,000× slower → parser is the ingest path, LLM only a reconcile-gated fallback. Migration `002` adds `previous_balance`, `total_cash_advances`, `parser`.

## Phase 3 — Enrichment: merchant normalisation and categories
**Concept:** LLM as a classifier; caching; humans in the loop.
- Normalise merchants (`"SQ *BLUE BOTTLE COF 0423"` → `Blue Bottle Coffee`) with rules first, then the LLM for the rest.
- Categorise each **unique merchant** once (dining, groceries, travel, subscriptions, ...) with an LLM using a fixed category list and structured output. In the same call, generate a **short description** ("specialty coffee roaster and cafe"); Phase 1 showed this is what makes vector search work. Cache both in `merchant_categories` (migration adds a `description` column).
- `rag categories review` lets you correct categories by hand (`source=manual` always wins).
- **Checkpoint:** every transaction has a category; you've spot-checked about 50.
- ✅ **Built 2026-11-13** → [notes/phase3-categories.md](notes/phase3-categories.md). 349 raw descriptors → 242 keys (rules) → 166 merchants; 30 keys by rule, 212 by LLM (~4 s each); 0/912 transactions uncategorised; synthetic accuracy 100%. `other` cut from 19% to 4% of transactions by putting `description` before `category` in the schema and removing `fees`/`payment` from the LLM's enum. Migration `003` (merchant_aliases, description, merchant_key). **Open:** owner spot-check via `rag categories review` (a telecom provider is still `other`).

## Phase 4 — Vector search with pgvector
**Concept:** Dense retrieval with metadata filtering, in the same database as the facts.
- Embed one **enriched** text per transaction, e.g. `"Blue Bottle Coffee: specialty coffee roaster and cafe"` (clean name + Phase 3 description), into `transaction_embeddings`. Date and amount stay as real columns, never in the embedded text (Phase 1: embeddings can't tell them apart). Identical texts can share one embedding call.
- `search_transactions(query, date_from, date_to, min_amount)` is one SQL query: ordinary `WHERE` filters plus `ORDER BY embedding <=> :query_vec LIMIT k` (cosine distance).
- **Hybrid search** for exact merchant names: rank by full-text (`ts_rank` on `search_tsv`) and trigram similarity (`pg_trgm`) as well as vectors, and merge with **Reciprocal Rank Fusion** in a single SQL query with CTEs.
- Compare embedding models (`mxbai-embed-large` vs `nomic-embed-text`/`qwen3-embedding`) on real data by storing both and switching `model` in the query.
- Optional: build an HNSW index and use `EXPLAIN ANALYZE` to compare exact and approximate search.
- **Checkpoint:** "coffee in March" and "hardware store purchases" return the right transactions.
- ✅ **Done 2026-11-13** → [notes/phase4-vector-search.md](notes/phase4-vector-search.md). 912 txns → 167 distinct enriched texts embedded with mxbai and nomic. Real precision (mxbai): concept queries 0.82 vector / 0.81 hybrid, name queries 1.00; nomic found no names by vector, so mxbai stays default. Checkpoint queries pass (vector). Hybrid first scored *below* vector: raw-descriptor full-text matched generic words ("STORE", "THE"); migration `005` moves full-text to the clean merchant name with the `english` config. RRF still trades off concepts against names/typos, so the Phase 7 agent should choose the search mode. HNSW skipped (912 rows).

## Phase 5 — Text-to-SQL
**Concept:** The LLM writes a query and the database computes the answer.
- Prompt = schema + category list + a few example question→SQL pairs + today's date (for "last month"). Say it's **PostgreSQL** so the model uses `date_trunc`, `EXTRACT`, `INTERVAL` correctly.
- Run as **`rag_reader`** in a read-only transaction; allow only a single `SELECT`; `statement_timeout` (e.g. 5s) and a row limit. Restrict `search_path` to `ragplay` so the model can only see app tables.
- **Self-correction loop:** on SQL error or empty result, feed the error back and retry (max 2).
- Have the LLM phrase the result, quoting the exact numbers returned.
- **Checkpoint:** totals and trends questions are correct against hand-checked answers.
- ✅ **Done 2026-11-14** → [notes/phase5-text2sql.md](notes/phase5-text2sql.md). Phase 0 baseline questions **2/5 → 5/5** (2–9 min → ~14 s each). Synthetic 12/12, real 11/11 against hand-written reference SQL, 0 answers with numbers not in the result. Getting there: conflicting prompt rules (statement vs calendar month), "kinds of business = categories", SQL in a code block instead of JSON (qwen3 dropped closing quotes inside JSON), and an example where a rule didn't work. Thinking on: >72 min vs 165 s, no accuracy headroom → off.

## Phase 6 — Evaluation (before tuning anything)
**Concept:** Measure first. Financial data makes this unusually precise, because SQL gives exact ground truth.
- `eval/questions.jsonl`: 25–40 questions across the three analysis types, each with expected answer (number, list of transaction IDs, or merchant list). Include unanswerable ones ("spend in 2019?").
- Metrics:
  - **Numeric exact match** (within $0.01) for totals
  - **Precision/recall** of returned transactions for "find" questions
  - **SQL validity rate** and **tool-selection accuracy**
  - **Faithfulness**: numbers in the answer must appear in tool outputs (checked in code, not by an LLM)
- `rag eval` prints a table and saves runs to `eval/runs/` (gitignored).
- **Checkpoint:** baseline scores. Every later change is judged against them.
- ✅ **Done 2026-11-14** → [notes/phase6-eval.md](notes/phase6-eval.md). Truth = reference SQL run at eval time (no amounts in the question file). Baseline (always text-to-SQL): **real 23/31** (totals 18/19, find 2/7, list 1/2, unanswerable 2/3; SQL valid first try 90%, faithful 97%); sample 12/16. Main gaps: "show me…" questions get aggregated (5/7 finds), zero vs no-data, "regularly" ignored, doesn't decline unanswerable. Three scorer bugs fixed before trusting any number (month in sentence, grouped duplicates, ±$0.01 tolerance accepting wrong answers).

## Phase 7 — The agent: tool-calling router
**Concept:** Agentic RAG. The LLM chooses the retrieval method.
- Tools exposed to `qwen3:8b` via Ollama tool calling:
  - `run_sql(query)` for totals, trends and filters
  - `search_transactions(query, filters)` for fuzzy or semantic lookup
  - `detect_recurring()`, `find_duplicates()`, `spend_outliers()` (Phase 8)
  - `get_schema()`, `list_categories()`
- Agent loop: plan → call tools (possibly several) → answer with numbers and the supporting transactions.
- **Conversational memory:** `rag chat` REPL; follow-ups like "and in April?" are rewritten into standalone questions.
- Compare against Phase 5: does routing beat "always SQL"? Is `llama3.2:3b` good enough?
- **Checkpoint:** one CLI handles all three question types; eval scores recorded.
- ✅ **Done 2026-11-15** → [notes/phase7-agent.md](notes/phase7-agent.md). Tools: `query_database()` (Phase 5 text-to-SQL on the user's own question; no arguments, because the model paraphrased away "on my statement"), `find_transactions(filters)` (rows, newest first), and declining. **Real 23/31 → 30/31** (finds 2/7 → 7/7, unanswerable 2/3 → 3/3, totals 19/19); sample 12/16 → 15/16. Remaining failures are Phase 8 anomaly questions. `rag chat` rewrites follow-ups. Fixes came from failures: deterministic filter relaxation, category definitions in the prompt, unknown tool arguments as errors, newest-first rows, zero vs no data. Infra: autocommit + keepalives + idle-transaction timeout after a sleeping laptop held locks; eval reconnects; sample eval in its own schema. llama3.2:3b comparison not done (optional).

## Phase 8 — Anomalies and subscriptions
**Concept:** Compute deterministically; let the LLM explain.
- `detect_recurring`: group by merchant and look for regular intervals (about 30 days or 1 year) with similar amounts. Flag **price increases**.
- `find_duplicates`: same merchant and amount within N days.
- `spend_outliers`: per-category z-score or IQR against your own history; new merchants with large first charges.
- The LLM produces a "monthly review": what changed, new subscriptions, things worth checking.
- **Checkpoint:** finds the subscriptions you know you have, plus at least one you'd forgotten (that's the fun part).

## Phase 9 — Rebuild with LlamaIndex
**Concept:** Map hand-built parts onto framework abstractions.
- `NLSQLTableQueryEngine` / `SQLTableRetrieverQueryEngine` over the same Postgres (`SQLDatabase` as `rag_reader`), `VectorStoreIndex` on `PGVectorStore` (its own table in the same DB), `RouterQueryEngine`, and a `FunctionAgent` with the same tools; Ollama LLM and embeddings.
- Run the same eval. Write `notes/framework-comparison.md`: code size, what was hidden, customisation pain, debuggability, scores.
- **Checkpoint:** comparable scores and a written comparison.

## Phase 10 — Stretch
- Observability: JSONL trace of each question → tool calls → SQL → result → answer.
- Budgets: "am I over my dining budget this month?" (adds a `budgets` table).
- CSV/OFX import as a second ingest path (should share the same `transactions` table).
- MCP server exposing the read-only tools so other agents can query your spending.
- Charts in the terminal (`plotext`) for trends.

---

## Design patterns covered (cheat sheet)
| Category | Patterns |
|---|---|
| Ingestion | deterministic parsing vs LLM structured extraction, vision fallback, validation/reconciliation, idempotent ingest |
| Enrichment | entity normalisation, LLM classification with fixed labels, caching, human-in-the-loop corrections |
| Retrieval | text-to-SQL, dense search + metadata filters in one query (pgvector), hybrid (full-text + trigram + vectors, RRF), structured vs unstructured routing |
| Orchestration | tool calling, router/agent loop, self-correcting SQL, conversational query rewriting |
| Analytics | compute in code/SQL, LLM explains; recurring detection, duplicates, outliers |
| Safety | local-only models, LAN-only DB over TLS, read-only DB role + timeouts, secrets in `.env`, PII masking, untrusted document text, gitignored data |
| Quality | exact-match numeric eval, retrieval precision/recall, code-checked faithfulness, regression runs |

## Key decisions recorded
| Decision | Choice | Reason |
|---|---|---|
| Data | Own credit card statements (text PDFs), 1–2 cards, ~1 year | The real use case |
| Architecture | Hybrid: PostgreSQL relational tables (structured) + pgvector (semantic) + tool-calling agent | Aggregation questions need exact SQL; plain RAG can't add up |
| Arithmetic | Never done by the LLM | Accuracy |
| Extraction | Deterministic parser per issuer, validated vs statement totals; LLM extraction as a comparison | Correctness of the base data |
| Build approach | Scratch first, then LlamaIndex | Understand internals, then learn the abstractions |
| Database | Remote PostgreSQL 17 on the LAN, one DB for facts + vectors (pgvector) | Single store: filters, joins, aggregation and similarity in one SQL query; real DB roles for read-only access |
| Interface | CLI (typer + rich) | Keeps focus on the RAG design |
| Python | 3.12 via uv | Library compatibility |

## Open items
- ~~Which issuers?~~ One: 19 Visa statements (card ...2423, Apr 2025 – Oct 2026) in `cc_statements/`. One parser to start.
- ~~Category list~~: settled in Phase 3 (`categorize.CATEGORIES`): groceries, dining, coffee, alcohol, gas, transport, travel, shopping, home, health, personal_care, entertainment, subscriptions, utilities, insurance, education, government, charity, fees, payment, other.
