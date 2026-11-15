"""Scorers must be right before any score means anything."""

from datetime import date
from decimal import Decimal

from ragplay.evaluate import declined, score_find, score_list, score_list_answer, score_total
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


def test_total_null_truth_fails_instead_of_crashing():
    assert score_total([None], final(["t"], [(D("1"),)]), "1") == (False, False)


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


def test_list_from_transaction_rows():
    rows = final(["id", "txn_date", "merchant", "amount"],
                 [(1, date(2026, 1, 3), "Netflix", D("15.49")), (2, date(2026, 2, 3), "Netflix", D("15.49")),
                  (3, date(2026, 1, 9), "Spotify", D("11.99"))])
    assert score_list(["Netflix", "Spotify"], rows) == (1.0, 1.0)


KNOWN = ["Netflix", "Spotify", "Apple", "Uber", "Uber Eats", "dining", "groceries"]


def test_list_answer_scoring():
    assert score_list_answer(["Netflix"], "Only Netflix went up, from 15.49 to 17.99.", KNOWN) == (1.0, 1.0)
    assert score_list_answer(["Netflix"], "Netflix and Spotify went up.", KNOWN) == (0.5, 1.0)
    assert score_list_answer(["Netflix", "Spotify"], "Netflix.", KNOWN) == (1.0, 0.5)


def test_list_answer_empty_truth():
    q = "Which subscriptions went up in price?"
    assert score_list_answer([], "None of your subscriptions went up in price.", KNOWN, q) == (1.0, 1.0)
    assert score_list_answer([], "Spotify went up.", KNOWN, q) == (0.0, 1.0)


def test_list_answer_negated_lines_are_not_claims():
    answer = ("The following subscriptions had price increases:\n1. **Netflix**: Increased from $15.49 "
              "to $17.99.\n2. **Apple**: No price change noted.\n3. **Spotify**: No price change noted.")
    assert score_list_answer(["Netflix"], answer, KNOWN, "Which subscriptions went up in price?") == (1.0, 1.0)


def test_list_answer_negation_only_matters_for_change_questions():
    answer = ("You pay for:\n1. **Apple** - Monthly, $2.99, no price change.\n"
              "2. **Netflix** - Monthly, $17.99, increased from $15.49.\n3. **Spotify** - Monthly, no price change.")
    q = "Which subscriptions do I pay for regularly?"
    assert score_list_answer(["Apple", "Netflix", "Spotify"], answer, KNOWN, q) == (1.0, 1.0)


def test_list_answer_longest_name_wins():
    assert score_list_answer(["Uber Eats"], "You use Uber Eats every month.", KNOWN) == (1.0, 1.0)


def test_declined():
    assert declined(None, "")
    assert declined(final(["t"], [(None,)]), "There is no data for 2019.")
    assert declined(final(["t"], [(D("100"),)]), "I don't have data on credit scores.")
    assert not declined(final(["credit_score"], [("Unknown",)]), "Your credit score is Unknown.")
    # empty result but the answer invents an amount: not a decline
    assert not declined(final(["year", "total"], []), "The total spent in 2019 was $2543.75.")
    assert declined(final(["total"], [(None,)]), "You spent $0 in 2019: there is no data.")
