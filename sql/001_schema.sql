-- 001: core ledger schema. Runs with search_path = ragplay, public (set by `rag db migrate`).
-- Sign convention for amounts: purchases/fees/interest are positive (money owed goes up);
-- payments and refunds are negative. Money is numeric(12,2), never float.

CREATE TABLE statements (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    card_last4       char(4)       NOT NULL CHECK (card_last4 ~ '^[0-9]{4}$'),
    issuer           text          NOT NULL,
    period_start     date          NOT NULL,
    period_end       date          NOT NULL,
    closing_balance  numeric(12,2) NOT NULL,
    total_purchases  numeric(12,2) NOT NULL DEFAULT 0,
    total_payments   numeric(12,2) NOT NULL DEFAULT 0,
    total_fees       numeric(12,2) NOT NULL DEFAULT 0,
    total_interest   numeric(12,2) NOT NULL DEFAULT 0,
    source_file      text          NOT NULL,
    file_hash        text          NOT NULL UNIQUE,   -- idempotent ingest
    ingested_at      timestamptz   NOT NULL DEFAULT now(),
    CHECK (period_end >= period_start)
);

CREATE TABLE transactions (
    id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    statement_id     bigint        NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    txn_date         date          NOT NULL,
    post_date        date,
    description_raw  text          NOT NULL,
    merchant         text,                            -- normalised in Phase 3
    amount           numeric(12,2) NOT NULL,
    type             text          NOT NULL
                     CHECK (type IN ('purchase', 'payment', 'refund', 'fee', 'interest')),
    category         text,                            -- filled in Phase 3
    card_last4       char(4)       NOT NULL CHECK (card_last4 ~ '^[0-9]{4}$'),
    search_tsv       tsvector GENERATED ALWAYS AS
                     (to_tsvector('simple', coalesce(merchant, '') || ' ' || description_raw)) STORED
);

CREATE INDEX transactions_statement_idx ON transactions (statement_id);
CREATE INDEX transactions_txn_date_idx  ON transactions (txn_date);
CREATE INDEX transactions_category_idx  ON transactions (category);
CREATE INDEX transactions_search_idx    ON transactions USING gin (search_tsv);
CREATE INDEX transactions_merchant_trgm ON transactions USING gin (merchant gin_trgm_ops);

CREATE TABLE merchant_categories (
    merchant    text        PRIMARY KEY,
    category    text        NOT NULL,
    source      text        NOT NULL CHECK (source IN ('rule', 'llm', 'manual')),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per (transaction, embedding model) so models can be compared side by side.
-- Dimension is left open because models differ; exact search is fine at this data size.
CREATE TABLE transaction_embeddings (
    transaction_id  bigint NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
    model           text   NOT NULL,
    embedding       vector NOT NULL,
    PRIMARY KEY (transaction_id, model)
);
