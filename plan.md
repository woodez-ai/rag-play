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
| "Coffee shop charges" | Works OK (semantic match on merchant names) | **Vector search** over transactions, or SQL after categorising |
| "Netflix price increases" | Unreliable | **Code/SQL** comparing amounts over time |

So the architecture is a **hybrid**, which is the main pattern to learn here:

```
             ┌──────────── INGEST (deterministic) ────────────┐
PDF ──► text extraction ──► transaction parser ──► validate vs statement totals
                                     │
                     ┌───────────────┴────────────────┐
                     ▼                                ▼
           SQLite: transactions,              ChromaDB: transaction +
           statements, categories             merchant embeddings
                     ▲                                ▲
             ┌───────┴──────── QUERY (LLM agent) ─────┴───────┐
Question ──► qwen3:8b with tools: run_sql | search_transactions | detect_recurring
             ──► answer with exact numbers + the transactions it used
```

Rule: **the LLM never does arithmetic.** It writes SQL or calls tools; the database computes numbers.

---

## Privacy and data handling (decide before any data touches the repo)
Statements are sensitive financial data. Everything stays local, which is why we use Ollama.
- `data/` (PDFs, SQLite DB, Chroma) is **gitignored from Phase 0**; add `.gitignore` rules before copying any statement in.
- Never commit statements, extracted text, DB files, eval answers or logs with real transactions.
- **Mask card numbers** during ingest (keep last 4 digits only); drop name and address lines.
- The SQL tool uses a **read-only** SQLite connection, so a bad or hostile generated query can't change data.
- Treat statement text as untrusted input (merchant descriptions end up in prompts).
- Eval questions (`eval/questions.jsonl`) contain real amounts, so keep them gitignored too. Commit only a **synthetic sample** for tests.
- Use a **synthetic statement generator** (`scripts/make_fake_statements.py`) for development, tests and anything shared.

---

## Environment (as checked 2026-11-07)
| Item | Value |
|---|---|
| Machine | Apple M3, 16 GB RAM |
| Ollama | `qwen3:8b` (chat + tool calling), `qwen3-vl:8b-instruct` (vision, if a PDF won't parse) |
| Python | 3.14 system; **use 3.12 in the venv** (ChromaDB/torch wheels lag new Pythons) |
| Package mgr | `uv` |

### Models to pull
| Role | Model | Why |
|---|---|---|
| Chat / tool calling / SQL | `qwen3:8b` (have it) | Good tool calling and text-to-SQL at 8B |
| Embeddings | `nomic-embed-text` | Small, fast baseline |
| Embeddings (compare) | `qwen3-embedding` or `mxbai-embed-large` | Embedding comparison exercise |
| Small chat (compare) | `llama3.2:3b` | Shows how model size affects SQL and tool-use accuracy |

---

## Project layout
```
rag-play/
  plan.md
  pyproject.toml              # uv-managed, python 3.12
  data/                       # GITIGNORED: raw PDFs, ledger.db, chroma/
  eval/questions.sample.jsonl # synthetic eval set (committed)
  eval/questions.jsonl        # real eval set (gitignored)
  scripts/make_fake_statements.py
  src/ragplay/
    extract.py                # PDF -> text/words with positions (pdfplumber)
    parsers/                  # one parser per issuer layout -> Transaction rows
    validate.py               # reconcile parsed rows vs statement summary totals
    db.py                     # SQLite schema + read-only connection
    categorize.py             # merchant normalisation + LLM categorisation (cached)
    embeddings.py / store.py  # Ollama embeddings + ChromaDB
    tools.py                  # run_sql, search_transactions, detect_recurring, ...
    agent.py                  # tool-calling loop
    cli.py                    # rag ingest | ask | chat | eval
  src/ragplay_llamaindex/     # framework rebuild
  notes/                      # per-phase learning notes
```

### Data model (SQLite)
```
statements(id, card_last4, issuer, period_start, period_end, closing_balance,
           total_purchases, total_payments, total_fees, total_interest, source_file, file_hash)
transactions(id, statement_id, txn_date, post_date, description_raw, merchant,
             amount, type /* purchase|payment|refund|fee|interest */, category, card_last4)
merchant_categories(merchant, category, source /* rule|llm|manual */)
```

---

## Phase 0 — Setup, privacy guardrails, baseline
**Concept:** Why grounding is needed. Safe handling of data.
- `uv init`, `uv venv --python 3.12`, activate, `uv add ollama chromadb pdfplumber typer rich`.
- Add `data/` and `eval/questions.jsonl` to `.gitignore` **before** copying statements in.
- `ollama pull nomic-embed-text`.
- Baseline: paste one statement's text into `qwen3:8b` and ask "total spent on groceries?". Check against a hand calculation. Expect arithmetic and omission errors. This is why the LLM won't do maths.
- **Checkpoint:** guardrails in place; baseline errors recorded.

## Phase 1 — Embeddings and similarity, by hand
**Concept:** Text → vector; where semantic search helps and where it fails.
- Embed merchant strings (`"SQ *BLUE BOTTLE COF"`, `"STARBUCKS #1234"`, `"SHELL OIL 5739"`) and compute cosine similarity with numpy.
- See that "coffee" lands near coffee shops, but embeddings can't tell $4 from $400, or March from May.
- **Checkpoint:** you can explain which questions need vectors and which need SQL.

## Phase 2 — Ingestion: PDF → structured transactions
**Concept:** Structured extraction is the most important (and least glamorous) part of the pipeline.
- Extract text with positions using `pdfplumber`. Look at the raw output of each issuer's layout.
- Write a **deterministic parser** per issuer (regex/column positions) → `Transaction` rows. Handle multi-line descriptions, credits shown as negatives or `CR`, and year rollover in Dec/Jan statements.
- **LLM-extraction comparison:** have `qwen3:8b` extract the same page using **structured output** (JSON schema via Ollama `format=`). Compare its accuracy with the regex parser. This shows when to use an LLM for extraction and when not to.
- Fallback for pages that won't parse: `qwen3-vl` on a page image.
- **Validation:** sum parsed purchases, payments and fees, and reconcile against the statement's summary box. Ingest fails loudly if they don't match.
- Idempotent ingest keyed on `file_hash`.
- **Checkpoint:** `rag ingest data/raw` loads all statements; every statement reconciles to the cent.

## Phase 3 — Enrichment: merchant normalisation and categories
**Concept:** LLM as a classifier; caching; humans in the loop.
- Normalise merchants (`"SQ *BLUE BOTTLE COF 0423"` → `Blue Bottle Coffee`) with rules first, then the LLM for the rest.
- Categorise each **unique merchant** once (dining, groceries, travel, subscriptions, ...) with an LLM using a fixed category list and structured output. Cache in `merchant_categories`.
- `rag categories review` lets you correct categories by hand (`source=manual` always wins).
- **Checkpoint:** every transaction has a category; you've spot-checked about 50.

## Phase 4 — Vector store for semantic transaction search
**Concept:** Dense retrieval with metadata filtering.
- One ChromaDB document per transaction: `"2026-03-14 Blue Bottle Coffee $6.50 dining"` with metadata `date`, `amount`, `category`, `card_last4`.
- `search_transactions(query, date_from, date_to, min_amount)` combines semantic match with metadata `where` filters.
- Also try **hybrid search** (BM25 + vectors merged with Reciprocal Rank Fusion) for exact merchant names.
- **Checkpoint:** "coffee in March" and "hardware store purchases" return the right transactions.

## Phase 5 — Text-to-SQL
**Concept:** The LLM writes a query and the database computes the answer.
- Prompt = schema + category list + a few example question→SQL pairs + today's date (for "last month").
- Run on the **read-only** connection; allow only `SELECT`; add a row limit and timeout.
- **Self-correction loop:** on SQL error or empty result, feed the error back and retry (max 2).
- Have the LLM phrase the result, quoting the exact numbers returned.
- **Checkpoint:** totals and trends questions are correct against hand-checked answers.

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

## Phase 8 — Anomalies and subscriptions
**Concept:** Compute deterministically; let the LLM explain.
- `detect_recurring`: group by merchant and look for regular intervals (about 30 days or 1 year) with similar amounts. Flag **price increases**.
- `find_duplicates`: same merchant and amount within N days.
- `spend_outliers`: per-category z-score or IQR against your own history; new merchants with large first charges.
- The LLM produces a "monthly review": what changed, new subscriptions, things worth checking.
- **Checkpoint:** finds the subscriptions you know you have, plus at least one you'd forgotten (that's the fun part).

## Phase 9 — Rebuild with LlamaIndex
**Concept:** Map hand-built parts onto framework abstractions.
- `NLSQLTableQueryEngine` / `SQLTableRetrieverQueryEngine` (text-to-SQL), `VectorStoreIndex` on Chroma, `RouterQueryEngine`, and a `FunctionAgent` with the same tools; Ollama LLM and embeddings.
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
| Retrieval | text-to-SQL, dense search + metadata filters, hybrid (BM25 + RRF), structured vs unstructured routing |
| Orchestration | tool calling, router/agent loop, self-correcting SQL, conversational query rewriting |
| Analytics | compute in code/SQL, LLM explains; recurring detection, duplicates, outliers |
| Safety | local-only models, read-only DB access, PII masking, untrusted document text, gitignored data |
| Quality | exact-match numeric eval, retrieval precision/recall, code-checked faithfulness, regression runs |

## Key decisions recorded
| Decision | Choice | Reason |
|---|---|---|
| Data | Own credit card statements (text PDFs), 1–2 cards, ~1 year | The real use case |
| Architecture | Hybrid: SQLite (structured) + ChromaDB (semantic) + tool-calling agent | Aggregation questions need exact SQL; plain RAG can't add up |
| Arithmetic | Never done by the LLM | Accuracy |
| Extraction | Deterministic parser per issuer, validated vs statement totals; LLM extraction as a comparison | Correctness of the base data |
| Build approach | Scratch first, then LlamaIndex | Understand internals, then learn the abstractions |
| Vector DB | ChromaDB embedded | No server, metadata filtering |
| Interface | CLI (typer + rich) | Keeps focus on the RAG design |
| Python | 3.12 via uv | Library compatibility |

## Open items
- Which issuers? Get one sample PDF per issuer to design the parsers.
- Category list: start from a standard set (dining, groceries, gas, travel, shopping, subscriptions, utilities, health, entertainment, fees/interest, other) and adjust.
