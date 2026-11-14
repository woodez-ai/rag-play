"""Scorers must be right before any score means anything."""

from datetime import date
from decimal import Decimal

from ragplay.evaluate import declined, score_find, score_list, score_total
from ragplay.text2sql import Attempt


def final(cols, rows):
    return Attempt("SELECT", columns=cols, rows=rows)


D = Decimal


def test_total_number_in_answer_and_result():
    f = final(["total"], [(D("673.99"),)])
    assert score_total([D("673.99")], f, "You spent $673.99 on groceries.") == (True, True)
    assert score_total([D("673.99")], f, "You spent $674.00.") == (False, True)
    assert score_total([D("1234.50")], final(["t"], [(D("1234.50"),)]), "Total: $1,234.50") == (True, True)


def test_total_sign_insensitive():
    assert score_total([D("-120.00")], final(["t"], [(D("-120.00"),)]), "Refunds came to $120.00.")[0]


def test_total_month_in_sentence_and_words():
    f = final(["month", "total"], [("2026-09", D("2825.60"))])
    assert score_total(["2026-09"], f, "The highest month was 2026-09, with $2825.60.") == (True, True)
    assert score_total(["2026-09"], f, "September 2026 was the highest.")[0]
    assert not score_total(["2026-09"], f, "August 2026 was the highest.")[0]


def test_total_text_value():
    f = final(["category"], [("dining",)])
    assert score_total(["dining"], f, "You spent the most on Dining.") == (True, True)


E = [(1, date(2026, 3, 2), D("4.50")), (2, date(2026, 3, 9), D("6.25"))]


def test_find_by_id_and_by_date_amount():
    assert score_find(E, final(["id"], [(1,), (2,)])) == (1.0, 1.0)
    assert score_find(E, final(["txn_date", "merchant", "amount"],
                                [(date(2026, 3, 2), "Starbucks", D("4.50"))])) == (1.0, 0.5)
    assert score_find(E, final(["txn_date", "amount"], [(date(2026, 3, 2), D("4.50")),
                                                          (date(2026, 3, 3), D("9.99"))])) == (0.5, 0.5)


def test_find_aggregate_instead_of_rows_scores_zero():
    assert score_find(E, final(["total"], [(D("10.75"),)])) == (0.0, 0.0)


def test_find_grouped_duplicates_count():
    dup = [(7, date(2026, 3, 14), D("91.12")), (8, date(2026, 3, 14), D("91.12"))]
    grouped = final(["merchant", "txn_date", "amount", "n"], [("Target", date(2026, 3, 14), D("91.12"), 2)])
    per_txn = final(["txn_date", "amount"], [(date(2026, 3, 14), D("91.12"))] * 2)
    assert score_find(dup, grouped) == (1.0, 1.0)
    assert score_find(dup, per_txn) == (1.0, 1.0)


def test_list():
    assert score_list(["Netflix", "Spotify"], final(["merchant"], [("Netflix",), ("Spotify",)])) == (1.0, 1.0)
    assert score_list(["Netflix", "Spotify"], final(["merchant"], [("Netflix",), ("Uber",)])) == (0.5, 0.5)


def test_declined():
    assert declined(None, "")
    assert declined(final(["t"], [(None,)]), "There is no data for 2019.")
    assert declined(final(["t"], [(D("100"),)]), "I don't have data on credit scores.")
    assert not declined(final(["credit_score"], [("Unknown",)]), "Your credit score is Unknown.")
