-- 005: full-text search over the clean merchant name only, with English stemming + stopwords.
-- The 001 version indexed merchant || raw descriptor with the 'simple' config. Measured in Phase 4:
-- generic raw tokens and stopwords matched the wrong merchants ("hardware STORE" -> "STARBUCKS
-- STORE", "filling up THE car" -> "THE HOME DEPOT") and dragged hybrid search below vector-only.
-- Phase 3 already moved the useful words into `merchant`; 'english' also makes "mcdonalds"
-- and "McDonald's" the same lexeme.
ALTER TABLE transactions DROP COLUMN search_tsv;  -- drops transactions_search_idx with it
ALTER TABLE transactions ADD COLUMN search_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(merchant, ''))) STORED;
CREATE INDEX transactions_search_idx ON transactions USING gin (search_tsv);
