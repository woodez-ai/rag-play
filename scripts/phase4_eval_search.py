"""Phase 4 eval: precision@10 of search modes (vector / text / hybrid) x embedding models.

Precision is over min(10, relevant transactions), so a 1-transaction merchant found first = 1.0.

Truth:
- concept queries ("latte") -> expected *category* (Phase 3 categories; real ones owner-reviewed)
- name queries ("mcdonalds", incl. typos) -> expected *merchant*
Prints scores only (safe on real data).

Usage: python scripts/phase4_eval_search.py --schema ragplay_test|ragplay
"""

import argparse

from ragplay import db, store

CONCEPT = {
    "ragplay_test": {
        "latte": "coffee", "hardware store": "home", "filling up the car": "gas",
        "supermarket": "groceries", "movie night": "entertainment", "taxi ride": "transport",
        "prescription medicine": "health", "streaming service": "subscriptions",
        "internet bill": "utilities", "place to stay abroad": "travel",
    },
    "ragplay": {
        "latte": "coffee", "filling up the car": "gas", "supermarket": "groceries",
        "phone bill": "utilities", "pharmacy": "health", "burgers and fries": "dining",
        "streaming service": "subscriptions", "beer and wine": "alcohol", "taxi ride": "transport",
        "online shopping": "shopping",
    },
}
# query -> ILIKE pattern that finds the expected merchant (the LLM picks the exact clean name,
# e.g. "Chipotle" one run and "Chipotle Mexican Grill" the next). Typos are deliberate.
NAMES = {
    "ragplay_test": {
        "trader joes": "trader joe%", "chipotle": "chipotle%", "comcast": "comcast%",
        "home depot": "%home depot%", "safeway": "safeway%", "starbuks": "starbucks%",
    },
    # names the owner shared in conversation
    "ragplay": {
        "mcdonalds": "mcdonald%", "mcdonnalds": "mcdonald%", "rogers": "rogers%",
        "wawa": "wawa%", "kia": "kia%", "network solutions": "network solutions%",
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--schema", default=db.TEST_SCHEMA)
    ap.add_argument("--models", default="mxbai-embed-large,nomic-embed-text")
    args = ap.parse_args()
    models = args.models.split(",")
    connect = db.connect_reader if args.schema == db.SCHEMA else (lambda: db.connect_owner(args.schema))
    with connect() as conn:
        names = dict(NAMES[args.schema])
        for q, pattern in names.items():
            rows = conn.execute("SELECT merchant FROM merchant_categories WHERE merchant ILIKE %s",
                                (pattern,)).fetchall()
            assert len(rows) == 1, f"{q!r}: expected one merchant, found {len(rows)}"
            names[q] = rows[0][0]
        print(f"{'query set':10} {'model':18} {'vector':>7} {'text':>7} {'hybrid':>7}")
        for model in models:
            for label, queries, field in (("concept", CONCEPT[args.schema], "category"),
                                          ("name", names, "merchant")):
                scores = {}
                for mode in ("vector", "text", "hybrid"):
                    if mode == "text" and model != models[0]:
                        scores[mode] = scores_text[label]  # text mode doesn't use the model
                        continue
                    p = []
                    for q, want in queries.items():
                        hits = store.search(conn, q, mode=mode, model=model, k=10)
                        # P@10 can't exceed n/10 for a merchant with n < 10 txns; normalise so
                        # a perfect result scores 1.0 (R-precision when n < 10).
                        n = conn.execute(f"SELECT count(*) FROM transactions WHERE {field} = %s",
                                         (want,)).fetchone()[0]
                        p.append(sum(getattr(h, field) == want for h in hits) / min(10, n))
                    scores[mode] = sum(p) / len(p)
                if model == models[0]:
                    scores_text = locals().get("scores_text", {})
                    scores_text[label] = scores["text"]
                print(f"{label:10} {model:18} {scores['vector']:7.2f} {scores['text']:7.2f} {scores['hybrid']:7.2f}")


if __name__ == "__main__":
    main()
