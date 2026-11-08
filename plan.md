# RAG Learning Plan — Local Ollama + ChromaDB

## Goal
Build a Retrieval-Augmented Generation (RAG) system over **my own documents (md / pdf / txt)**,
answered by **local Ollama models**, to learn RAG solution design and the main design patterns.

Approach: **build each piece from scratch first**, then **rebuild with a framework (LlamaIndex)**
and compare. Interface is a **CLI**. Vector store is **ChromaDB** (embedded, no server).

Each phase introduces one concept, ends with a working checkpoint, and records what was learned
in `notes/phase-N.md`. Don't move on until the checkpoint works and you can explain why.

---

## Environment (as checked 2026-11-07)
| Item | Value |
|---|---|
| Machine | Apple M3, 16 GB RAM |
| Ollama | installed; `qwen3:8b` present (chat model) |
| Python | 3.14 system — **use 3.12 in the venv** (ChromaDB/torch wheels lag new Pythons) |
| Package mgr | `uv` |

16 GB means: one 8B chat model + one embedding model loaded at once is fine; avoid >14B models.

### Models to pull
| Role | Model | Why |
|---|---|---|
| Chat / generation | `qwen3:8b` (have it) | Good quality at 8B; supports thinking mode |
| Embeddings (primary) | `nomic-embed-text` | Small, fast, 8k context, the common baseline |
| Embeddings (compare) | `mxbai-embed-large` or `qwen3-embedding` | For the embedding-comparison exercise |
| Small chat (compare) | `llama3.2:3b` | Shows how much model size matters vs. retrieval quality |

---

## Project layout
```
rag-play/
  plan.md
  pyproject.toml          # uv-managed, python 3.12
  data/raw/               # your md/pdf/txt (gitignored if private)
  data/chroma/            # persisted vector DB (gitignored)
  eval/questions.jsonl    # hand-written Q&A + expected source docs
  src/ragplay/
    loaders.py            # file -> text + metadata
    chunking.py           # text -> chunks
    embeddings.py         # ollama embed wrapper
    store.py              # ChromaDB wrapper
    retrieve.py           # retrieval strategies
    generate.py           # prompt building + ollama chat
    pipeline.py           # wires it together
    cli.py                # `rag ingest`, `rag ask`, `rag eval`
  src/ragplay_llamaindex/ # framework rebuild (Phase 8)
  notes/                  # per-phase learning notes
```

---

## Phase 0 — Setup and a baseline without RAG
**Concept:** Why RAG exists. See what the LLM does *without* your data.
- `uv init`, `uv venv --python 3.12`, activate, then `uv add ollama chromadb pypdf typer rich`.
- `ollama pull nomic-embed-text`.
- Script: ask `qwen3:8b` 5 questions only answerable from your docs. Record the answers (expect hallucination or "I don't know").
- **Checkpoint:** baseline answers saved — this is what RAG must beat.

## Phase 1 — Embeddings and similarity, by hand
**Concept:** Text → vector; meaning ≈ geometry.
- Call `ollama.embed()` on a handful of sentences; compute cosine similarity with numpy.
- Experiment: paraphrases vs. unrelated sentences vs. keyword-overlap-but-different-meaning.
- Brute-force search: embed 20 paragraphs, find top-k for a query with no vector DB.
- **Checkpoint:** you can explain why "car" and "automobile" are close, and where embeddings fail (exact IDs, numbers, negation).

## Phase 2 — Ingestion pipeline (loading + chunking)
**Concept:** Retrieval quality is mostly decided at ingest time.
- Loaders: `.md`/`.txt` direct; `.pdf` via `pypdf` (keep page numbers in metadata).
- Metadata per chunk: `source`, `page`, `heading`, `chunk_index`, `doc_hash`.
- Implement and compare chunking strategies:
  1. Fixed size (e.g. 500 tokens) with overlap (e.g. 50)
  2. Recursive / separator-aware (paragraph → sentence → word)
  3. Markdown-heading-aware (chunk per section, heading kept as context)
- **Checkpoint:** `rag ingest data/raw` prints chunk counts and sample chunks per strategy.

## Phase 3 — Vector store (ChromaDB)
**Concept:** Indexing, persistence, metadata filtering, idempotent ingest.
- Persistent client at `data/chroma/`; one collection per (chunking strategy × embedding model) so they can be compared.
- Supply your own embeddings (from Ollama) rather than Chroma's default, so you control the model.
- Idempotency: skip unchanged files using `doc_hash`; delete + re-add changed ones.
- Try metadata filters (`where={"source": ...}`).
- **Checkpoint:** re-running ingest is a no-op; queries return top-k chunks with sources.

## Phase 4 — Naive RAG end to end (the core pattern)
**Concept:** Retrieve → augment prompt → generate.
- Prompt template: system instructions + numbered context chunks + question; require citations like `[1]` and "say you don't know if the context doesn't contain it".
- `rag ask "question"` prints answer + cited sources; `--show-context` flag prints retrieved chunks.
- Re-run the Phase 0 questions and compare.
- Things to observe: context window limits, top-k tradeoff, "lost in the middle", the model ignoring the context.
- **Checkpoint:** grounded, cited answers to the baseline questions.

## Phase 5 — Evaluation (do this before optimising)
**Concept:** You can't improve what you don't measure.
- Write `eval/questions.jsonl`: 20–30 questions with expected answer and expected source file(s). Include some unanswerable ones.
- Retrieval metrics: **hit rate@k**, **MRR** (did the right source come back, how high).
- Generation metrics: **faithfulness** (answer supported by context?) and **correctness**, scored with LLM-as-judge using `qwen3:8b` plus manual spot checks.
- `rag eval --collection X` prints a results table; store results per run in `eval/runs/`.
- **Checkpoint:** baseline scores for naive RAG. Every later phase is judged against these.

## Phase 6 — Advanced retrieval patterns
Implement each as a switchable option, and evaluate each against the Phase 5 baseline.
| Pattern | What it fixes |
|---|---|
| **Hybrid search** (BM25 via `rank_bm25` + vectors, merged with Reciprocal Rank Fusion) | Exact terms, names, IDs that embeddings miss |
| **Re-ranking** (cross-encoder, e.g. `bge-reranker-base` via sentence-transformers, or LLM rerank) | Top-k contains the right chunk but in the wrong order |
| **Query rewriting / multi-query** | Vague or badly phrased questions |
| **HyDE** (embed a hypothetical answer) | Question and answer phrased very differently |
| **Parent-document / small-to-big** (retrieve small chunks, send their parent section) | Precise retrieval but too little context |
| **Contextual chunk headers** (prepend doc title/heading to chunk before embedding) | Chunks that make no sense alone |
| **MMR** (max marginal relevance) | Near-duplicate chunks wasting context |
- **Checkpoint:** an eval table comparing patterns; notes on which helped *your* data and why.

## Phase 7 — Agentic and conversational patterns
**Concept:** The LLM decides when and how to retrieve.
- **Conversational RAG:** chat history + question condensing ("what about the second one?" → standalone query). `rag chat` REPL.
- **Routing:** classify a question → which collection/filter to use, or no retrieval needed.
- **Tool-calling RAG:** expose `search_docs(query, filter)` as an Ollama tool; let `qwen3:8b` call it, possibly several times.
- **Corrective / self-reflective RAG (CRAG / Self-RAG lite):** grade retrieved chunks for relevance; if poor, rewrite the query and retry; check the answer for groundedness before returning.
- **Checkpoint:** multi-turn chat works; agentic version handles a multi-hop question naive RAG fails.

## Phase 8 — Rebuild with a framework (LlamaIndex)
**Concept:** Map what you built by hand onto framework abstractions.
- Same data, Ollama models (`llama-index-llms-ollama`, `llama-index-embeddings-ollama`), and Chroma (`llama-index-vector-stores-chroma`).
- Rebuild: ingestion pipeline, query engine, reranker, chat engine, agent with tools.
- Run the same Phase 5 eval.
- Write `notes/framework-comparison.md`: lines of code, what was hidden, what was hard to customize, debuggability, results.
- Optional: repeat a slice in LangChain/LangGraph for a second comparison.
- **Checkpoint:** comparable eval scores; written comparison.

## Phase 9 — Production-minded concerns (stretch)
Pick what interests you:
- Observability: log each query's retrieved chunks, prompts, latency, tokens (simple JSONL trace, or Arize Phoenix locally).
- Caching embeddings and answers.
- Incremental ingest / watch a folder for changes.
- Prompt-injection risk from documents (content inside docs is untrusted input).
- Structured output (JSON answers with citations schema).
- Expose retrieval as an MCP server so other agents can use it.
- GraphRAG / knowledge-graph extraction (a deeper topic for later).

---

## Design patterns covered (cheat sheet)
| Category | Patterns |
|---|---|
| Ingestion | loaders, chunking strategies, metadata enrichment, contextual headers, idempotent/incremental indexing |
| Retrieval | dense, sparse (BM25), hybrid + RRF, metadata filtering, MMR, parent-document, HyDE, multi-query |
| Post-retrieval | re-ranking, context compression, deduplication, ordering |
| Generation | grounded prompting, citations, refusal when unsupported, structured output |
| Orchestration | naive RAG, conversational RAG, routing, tool-calling/agentic RAG, corrective/self-RAG |
| Quality | retrieval metrics (hit@k, MRR), LLM-as-judge faithfulness/correctness, regression eval runs |

## Key decisions recorded
| Decision | Choice | Reason |
|---|---|---|
| Data | Own md/pdf/txt | Easy to judge whether answers are right |
| Build approach | Scratch first, then LlamaIndex | Understand internals, then learn the abstractions |
| Vector DB | ChromaDB embedded | No server, minimal setup, metadata filtering |
| Interface | CLI (typer + rich) | Keeps focus on RAG, not UI |
| Python | 3.12 via uv | Library compatibility |

## Open items
- Pick the document set to put in `data/raw/` (ideally 20–200 docs on topics you know well).
- Decide whether `data/raw/` is private (gitignore it) or committed.
