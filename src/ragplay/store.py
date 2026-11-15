"""Transaction embeddings in pgvector, and search: vector, text, and hybrid (RRF), all in SQL.

Embedded text is the enriched merchant text, e.g. "Blue Bottle Coffee: specialty coffee roaster
and cafe". Dates and amounts are *never* in it (Phase 1: embeddings can't tell March from May or
$4 from $400); they stay real columns and become ordinary WHERE filters in the same query.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import numpy as np
import psycopg

from ragplay.embeddings import DEFAULT_MODEL, embed

RRF_K = 60  # standard Reciprocal Rank Fusion constant: score = Σ 1 / (RRF_K + rank)
CANDIDATES = 50  # per-method candidates fed into the fusion
TEXT_WEIGHT = 1.0  # weight of full-text and trigram ranks relative to the vector rank


def content_for(merchant: str | None, description: str | None, raw: str) -> str:
    if merchant and description:
        return f"{merchant}: {description}"
    return merchant or raw


def embed_transactions(conn: psycopg.Connection, model: str = DEFAULT_MODEL, batch: int = 64) -> tuple[int, int]:
    """Embed transactions that have no embedding for `model`, or whose text changed.
    Identical texts share one embedding call. Returns (transactions written, texts embedded)."""
    rows = conn.execute(
        """SELECT t.id, t.merchant, m.description, t.description_raw
           FROM transactions t
           LEFT JOIN merchant_categories m ON m.merchant = t.merchant
           LEFT JOIN transaction_embeddings e ON e.transaction_id = t.id AND e.model = %s
           WHERE e.transaction_id IS NULL
              OR e.content IS DISTINCT FROM
                 CASE WHEN t.merchant IS NOT NULL AND m.description IS NOT NULL
                      THEN t.merchant || ': ' || m.description
                      ELSE coalesce(t.merchant, t.description_raw) END""",
        (model,)).fetchall()
    by_text: dict[str, list[int]] = {}
    for tid, merchant, desc, raw in rows:
        by_text.setdefault(content_for(merchant, desc, raw), []).append(tid)
    texts = sorted(by_text)
    written = 0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        vecs = embed(chunk, model=model, kind="document")
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO transaction_embeddings (transaction_id, model, embedding, content)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (transaction_id, model)
                   DO UPDATE SET embedding = EXCLUDED.embedding, content = EXCLUDED.content""",
                [(tid, model, vec, text) for text, vec in zip(chunk, vecs) for tid in by_text[text]])
            written += sum(len(by_text[t]) for t in chunk)
    return written, len(texts)


@dataclass
class Hit:
    id: int
    txn_date: date
    merchant: str | None
    description_raw: str
    amount: Decimal
    category: str | None
    score: float


FILTERS = """
    (%(date_from)s::date IS NULL OR t.txn_date >= %(date_from)s)
    AND (%(date_to)s::date IS NULL OR t.txn_date <= %(date_to)s)
    AND (%(min_amount)s::numeric IS NULL OR t.amount >= %(min_amount)s)
    AND (%(max_amount)s::numeric IS NULL OR t.amount <= %(max_amount)s)
    AND (%(category)s::text IS NULL OR t.category = %(category)s)
    AND (%(merchant)s::text IS NULL OR t.merchant ILIKE '%%' || %(merchant)s || '%%')
    AND (%(category)s::text IS NOT NULL OR t.type <> 'payment')
"""

# Each method ranks the *filtered* transactions; RRF fuses the ranks. One round trip.
#   vec: cosine distance on the enriched-text embedding (meaning: "latte" -> coffee shops)
#   fts: full-text on the clean merchant name ('english'), terms OR-ed (exact words: "rogers")
#   trg: trigram word similarity to the merchant name (typos/spacing: "mcdonalds")
HYBRID_SQL = f"""
WITH f AS (
    SELECT t.* FROM transactions t WHERE {FILTERS}
),
vec AS (
    SELECT f.id, row_number() OVER (ORDER BY e.embedding <=> %(qvec)s) AS r
    FROM f JOIN transaction_embeddings e ON e.transaction_id = f.id AND e.model = %(model)s
    ORDER BY e.embedding <=> %(qvec)s LIMIT {CANDIDATES}
),
q AS (
    SELECT nullif(replace(plainto_tsquery('english', %(text)s)::text, '&', '|'), '')::tsquery AS tsq
),
fts AS (
    SELECT f.id, row_number() OVER (ORDER BY ts_rank(f.search_tsv, q.tsq) DESC) AS r
    FROM f, q WHERE f.search_tsv @@ q.tsq
    ORDER BY ts_rank(f.search_tsv, q.tsq) DESC LIMIT {CANDIDATES}
),
trg AS (
    SELECT f.id, row_number() OVER (ORDER BY word_similarity(%(text)s, f.merchant) DESC) AS r
    FROM f WHERE word_similarity(%(text)s, f.merchant) >= 0.5
    ORDER BY word_similarity(%(text)s, f.merchant) DESC LIMIT {CANDIDATES}
),
fused AS (
    SELECT id, sum(w / ({RRF_K} + r)) AS score
    FROM (SELECT id, r, 1.0 AS w FROM vec WHERE %(use_vec)s
          UNION ALL SELECT id, r, %(text_weight)s FROM fts WHERE %(use_text)s
          UNION ALL SELECT id, r, %(text_weight)s FROM trg WHERE %(use_text)s) ranks
    GROUP BY id
)
SELECT t.id, t.txn_date, t.merchant, t.description_raw, t.amount, t.category, fused.score
FROM fused JOIN transactions t USING (id)
ORDER BY fused.score DESC, t.txn_date DESC
LIMIT %(k)s
"""


def search(conn: psycopg.Connection, query: str, *, mode: str = "hybrid",
           model: str = DEFAULT_MODEL, k: int = 10, date_from: date | None = None,
           date_to: date | None = None, min_amount: Decimal | None = None,
           max_amount: Decimal | None = None, category: str | None = None,
           merchant: str | None = None, text_weight: float = TEXT_WEIGHT) -> list[Hit]:
    """mode: 'vector' (embeddings only), 'text' (full-text + trigram), 'hybrid' (all, RRF).
    Payments are excluded unless category='payment'."""
    if mode not in ("vector", "text", "hybrid"):
        raise ValueError(f"unknown mode {mode!r}")
    qvec = embed([query], model=model, kind="query")[0] if mode != "text" else np.zeros(1, np.float32)
    params = dict(qvec=qvec, model=model, text=query, k=k, date_from=date_from, date_to=date_to,
                  min_amount=min_amount, max_amount=max_amount, category=category,
                  merchant=merchant, use_vec=mode != "text", use_text=mode != "vector",
                  text_weight=text_weight)
    return [Hit(*r) for r in conn.execute(HYBRID_SQL, params).fetchall()]


FIND_SQL = f"""
SELECT t.id, t.txn_date, t.merchant, t.description_raw, t.amount, t.category, 0.0
FROM transactions t WHERE {FILTERS}
ORDER BY t.txn_date DESC, t.id DESC LIMIT %(k)s  -- newest first: "most recent" must be visible
"""


def find(conn: psycopg.Connection, *, text: str | None = None, merchant: str | None = None,
         category: str | None = None, date_from: date | None = None, date_to: date | None = None,
         min_amount: Decimal | None = None, max_amount: Decimal | None = None,
         k: int = 100, model: str = DEFAULT_MODEL) -> list[Hit]:
    """Transactions matching structured filters (all rows, newest first). With `text`, rank by
    hybrid search instead and return the top k. Structured filters are exact; text ranking
    always returns k rows, so prefer filters when a category or merchant fits."""
    if text:
        return search(conn, text, model=model, k=k, date_from=date_from, date_to=date_to,
                      min_amount=min_amount, max_amount=max_amount, category=category,
                      merchant=merchant)
    params = dict(k=k, date_from=date_from, date_to=date_to, min_amount=min_amount,
                  max_amount=max_amount, category=category, merchant=merchant)
    return [Hit(*r) for r in conn.execute(FIND_SQL, params).fetchall()]
