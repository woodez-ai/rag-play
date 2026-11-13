-- 003: merchant normalisation + categories with descriptions.
--   transactions.merchant_key  -> rule-normalised descriptor ("SQ *BLUE BOTTLE COF 0423 OAKLAND CA"
--                                 -> "BLUE BOTTLE COF OAKLAND")
--   merchant_aliases            -> key -> clean merchant name ("Blue Bottle Coffee")
--   merchant_categories         -> clean merchant -> category + short description (embedded in Phase 4)
-- source: rule | llm | manual. Manual rows are never overwritten by rule or llm.

ALTER TABLE merchant_categories ADD COLUMN description text;

CREATE TABLE merchant_aliases (
    merchant_key text        PRIMARY KEY,
    merchant     text        NOT NULL REFERENCES merchant_categories(merchant) ON UPDATE CASCADE,
    source       text        NOT NULL CHECK (source IN ('rule', 'llm', 'manual')),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE transactions ADD COLUMN merchant_key text;
CREATE INDEX transactions_merchant_key_idx ON transactions (merchant_key);
