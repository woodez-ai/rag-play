-- 004: remember the exact text each embedding was made from, so a changed merchant name or
-- description (e.g. a manual fix) is detected and re-embedded instead of going stale.
ALTER TABLE transaction_embeddings ADD COLUMN content text NOT NULL DEFAULT '';
