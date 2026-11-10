-- 002: keep every summary-box figure so a statement's reconciliation can be re-checked in SQL,
-- and so consecutive statements can be chained (previous_balance = prior closing_balance).
ALTER TABLE statements
    ADD COLUMN previous_balance    numeric(12,2) NOT NULL DEFAULT 0,
    ADD COLUMN total_cash_advances numeric(12,2) NOT NULL DEFAULT 0,
    ADD COLUMN parser              text          NOT NULL DEFAULT 'unknown';
