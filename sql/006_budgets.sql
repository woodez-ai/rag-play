-- 006: monthly budgets per category ('all' = total spending). Spend is measured like everywhere
-- else: purchases + refunds by txn_date in the calendar month.
CREATE TABLE budgets (
    category        text          PRIMARY KEY,
    monthly_amount  numeric(12,2) NOT NULL CHECK (monthly_amount > 0),
    updated_at      timestamptz   NOT NULL DEFAULT now()
);
