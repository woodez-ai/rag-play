# rag-play

Ask questions about your own credit card statements in plain English, from a web UI or the
terminal. The whole pipeline runs locally: a small Ollama model (`qwen3:8b`) and PostgreSQL with
pgvector. No statement data leaves your network.

> "How much did I spend on dining in March?" · "Which subscriptions went up in price?" ·
> "Am I over my groceries budget this month?" · "Show my hardware store purchases this year."

## Use cases

- **Spending questions:** totals by category, merchant or period, answered with exact numbers.
- **Find transactions:** search by meaning ("coffee", "flights") or by exact filters.
- **Subscriptions:** list recurring charges, their monthly cost, price increases and the next charge date.
- **Spot problems:** possible duplicate charges, and purchases unusually large for that category.
- **Budgets:** monthly limits per category or overall, with a projection to month end.
- **Monthly review:** a written summary of the month, generated from numbers computed in code.
- **Learning RAG:** each phase in `notes/` documents a design choice and its measured effect.

## Design

```
 PDF statements ──► parse + reconcile to the cent ──► PostgreSQL (statements, transactions)
                                                        │
                    merchant clean-up + categories ◄────┤  rules → LLM (cached; your fixes win)
                    embeddings (pgvector)          ◄────┘  "merchant: description" text
                                                        │
 question ──► router / agent (qwen3:8b, tool calling) ──┼─► text-to-SQL (read-only role) → totals
                                                        ├─► hybrid search (full-text + trigram + vectors)
                                                        ├─► recurring / duplicates / outliers (code)
                                                        └─► budget status (code)
          ◄── answer phrased from tool rows; numbers not in the rows are flagged
```

Principles:
- **The LLM never does arithmetic.** Totals come from SQL; anomalies and budgets are computed in
  code. The model chooses tools and writes the answer, and a check flags any number in the
  answer that isn't in the tool output.
- **Deterministic ingestion.** One parser per statement layout. A statement loads only if its
  transactions reconcile with its own totals. Ingest is idempotent (keyed by file hash).
- **Safety.** Generated SQL runs as a read-only role, as a single SELECT with a timeout and a row
  cap. The web UI binds to 127.0.0.1 only and uses a strict CSP. Statements, traces and eval runs
  are gitignored.
- **Measured.** An eval harness checks answers against reference SQL: 32/32 on real data and
  18/18 on the synthetic set, with 100% of numbers traceable to tool output.

Stack: Python 3.12 · Ollama (`qwen3:8b`, `mxbai-embed-large`) · PostgreSQL 17 + pgvector ·
pdfplumber · Typer · Django + HTMX.

## Install

1. **Ollama:**
   ```bash
   ollama pull qwen3:8b && ollama pull mxbai-embed-large
   ```
2. **PostgreSQL 17 with pgvector.** As an admin, create a database, an owner role and a
   read-only role:
   ```sql
   CREATE ROLE rag_user LOGIN PASSWORD '...';
   CREATE ROLE rag_reader LOGIN PASSWORD '...';
   CREATE DATABASE ragdb OWNER rag_user;
   \c ragdb
   CREATE EXTENSION vector; CREATE EXTENSION pg_trgm;
   ALTER ROLE rag_reader SET default_transaction_read_only = on;
   ALTER ROLE rag_reader SET statement_timeout = '5s';
   ```
   After step 5, grant `rag_reader` USAGE on schema `ragplay` and SELECT on its tables. The
   connection requires TLS (`sslmode=require`).
3. **Python environment:**
   ```bash
   git clone <this repo> && cd rag-play
   uv venv && source .venv/bin/activate
   uv pip install -e '.[web]'
   ```
4. **`.env`** in the repo root (gitignored): `DB_HOST`, `DB`, `DB_USER`, `DB_PASSWORD`,
   `DB_READER_USER`, `DB_READER_PASSWORD`.
5. **Create the tables:**
   ```bash
   rag db migrate && rag db check
   ```

Statement parsers live in `src/ragplay/parsers/`. A bank with a different PDF layout needs its
own parser. To try it without real statements, run `rag eval --set sample` once. This loads
synthetic statements into the `ragplay_sample` schema. Then run `rag web --schema ragplay_sample`.

## Usage

```bash
rag ingest cc_statements/ && rag categorize && rag embed   # load statements
rag web                                                    # web UI at http://127.0.0.1:8000
rag chat                                                   # or chat in the terminal
rag ask "How much did I spend on groceries last month?"
```

The web UI has these pages: Dashboard, Ask, Spending, Insights, Budgets, Categories, Import
(drag-and-drop PDFs) and History.

See **[howto-use.md](howto-use.md)** for every page and command, a monthly routine and
troubleshooting. Design notes and measurements for each phase are in **[notes/](notes/)** and
**[plan.md](plan.md)**.
