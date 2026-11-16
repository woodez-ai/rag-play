# Phase 9 — Hand-built vs LlamaIndex

**Date:** 2026-11-15
**Code:** `src/ragplay_llamaindex/` (optional extra `ragplay[llamaindex]`), `rag li-index`,
`rag li-ask`, `rag eval --system llamaindex | llamaindex-tuned`
**Versions:** llama-index-core 0.14.26, llms-ollama 0.12, embeddings-ollama 0.11,
vector-stores-postgres 0.10. Same models (qwen3:8b, mxbai-embed-large), same database, same eval.

## Mapping
| Hand-built | LlamaIndex |
|---|---|
| `text2sql.ask` (prompt, guard, read-only run, retry, phrasing) | `NLSQLTableQueryEngine` over `SQLDatabase` (read-only engine) |
| `store.embed_transactions` / `store.find` | `VectorStoreIndex` on `PGVectorStore` (own schema `ragplay_li`) + `MetadataFilters` |
| `agent.ask` (tool loop, router, relaxation, arg checks) | `FunctionAgent` (workflow) + `FunctionTool`s |
| `anomalies.*` | the same functions wrapped as `FunctionTool`s (deterministic; not reimplemented) |

## Scores

| System | Real (32) | totals | find (P/R) | list | unanswerable | Sample (16) | Faithful (real) |
|---|---|---|---|---|---|---|---|
| **Hand-built (Phases 5–8)** | **32** | 19/19 | 7/7 (1.00/1.00) | 3/3 | 3/3 | **16** | 100% |
| LlamaIndex, defaults | 14 | 7/19 | 4/7 (0.69/1.00) | 0/3 | 3/3 | 8 | 91% |
| LlamaIndex + our SQL conventions | 22 | 14/19 | 4/7 (0.69/1.00) | 1/3 | 3/3 | 10 | 94% |

Time for 32 real questions: hand-built 718 s, LlamaIndex 1011 s (defaults) / 822 s (tuned).

### Where the 18 points go
- **Domain knowledge: +8.** The framework's text-to-SQL prompt is generic (schema + 3 sample rows
  per table). Without our conventions qwen3 invented values (`type = 'charge'`, `category =
  'Coffee'`, `'Restaurants'`), wrote invalid SQL (unescaped `'McDonald's'`), used `CURRENT_DATE`
  instead of the given date, grouped by month number across years, and mixed statement payments
  into "spending". Passing our rules and examples through LlamaIndex's own hook
  (`context_query_kwargs`) took real totals from 7/19 to 14/19 with no other change.
- **Orchestration: the other 10**, which are lessons from Phases 7–8 the framework doesn't have:
  - The agent paraphrases the question for the SQL tool ("on my March 2026 statement" → "in March
    2026"): all 4 statement questions on the sample fail. The hand-built tool takes no argument.
  - "Show me…" uses a top-k similarity retriever. It always returns k rows (top 30), so precision
    falls as soon as fewer than 30 match: Amazon over $100 P=0.13, McDonald's in September P=0.40.
    The hand-built `find` returns exactly the rows that match the filters.
  - With no calculation router, the model answered "how many / largest / how much" with the
    retriever and did the arithmetic itself (unfaithful answers).
  - It declined two answerable questions (closing balance, interest): nothing told it statements
    hold them, and there's no "zero is an answer" rule.
  - Subscription questions: extra names, and a claimed price increase the tool didn't report.

## Code size (non-blank, non-comment lines)
| | Hand-built | LlamaIndex |
|---|---|---|
| text-to-SQL + search + agent + embeddings | **657** | **169** |
| shared anomaly detectors | 244 | (reused) |
| added dependencies | — | ~68 MB (llama-index-*, workflows, sqlalchemy, asyncpg, psycopg2, nltk, tiktoken) |

~4× less code to a working first version. The hand-built extra lines are mostly the fixes that
made the difference: read-only guard + retry, router, relaxation, argument checks, no-LLM
phrasing of empty results, faithfulness checks.

## What the framework hides (and what it cost to find out)
- **Defaults to OpenAI** for LLM and embeddings. Any unconfigured component silently targets
  `api.openai.com`. Set `Settings.llm`/`Settings.embed_model` at import (tested).
- **Sample rows in the SQL prompt** (`sample_rows_in_table_info=3`): real transactions go into
  every text-to-SQL prompt by default (local model here, so acceptable, but you have to know).
- **`PGVectorStore` and passwords:** it `str()`s the SQLAlchemy URL, and SQLAlchemy 2 masks the
  password as `***` → "password authentication failed". Fix: `render_as_string(hide_password=False)`.
- **Async lifecycle:** `FunctionAgent.run()` schedules a task, so it must be called inside a
  running loop; and the Ollama client binds to the first loop, so `asyncio.run()` per question
  dies with "Event loop is closed". One persistent loop per assistant.
- **`str(ChatMessage)`** prefixes the role ("assistant: …"); use `.content`.
- **Default request timeout 30 s** for Ollama: too short for an 8B model on a laptop.
- **One embedding per node:** 912 embedding calls vs 167 distinct texts in the hand-built store.
- **Setup noise:** `PGVectorStore` prints "PG Setup: Error …" lines when it can't create things
  it doesn't need.
- **Retrieval semantics:** a retriever answers "the k most similar", never "all that match". For
  money questions that's the wrong primitive for listing.

## Customisation and debuggability
- **Easy:** swapping models, adding tools (`FunctionTool.from_defaults(fn)` reads the docstring
  and signature), adding table context (`context_query_kwargs`), getting the SQL back
  (`response.metadata["sql_query"]`, `["result"]`).
- **Hard:** the behaviours that mattered most live *inside* the framework's loop: forcing a tool
  for a class of questions, passing the user's question through unchanged, refusing unknown tool
  arguments, never phrasing empty results. Each means subclassing, a custom workflow, or wrapping
  tools, which gives back much of the code saved.
- **Debugging:** stack traces run through workflows/instrumentation/httpx layers (the event-loop
  bug was 60 frames deep). The hand-built loop is ~100 lines you can step through.

## Verdict
- **Prototype with the framework:** a working assistant in ~170 lines and an afternoon, and a good
  map of the components.
- **Ship the hand-built pipeline here:** for money questions with exact answers, every point from
  14 to 32 came from domain conventions and small, deterministic orchestration choices the eval
  forced us to make. The framework neither has them nor makes them easy to add.
- The eval was the deciding tool: without it, the LlamaIndex version's fluent answers ("You spent
  $603.84 on groceries in March 2026") would have looked fine.
