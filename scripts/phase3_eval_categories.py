"""Phase 3 eval: category accuracy on synthetic statements (ground truth from the generator).

Loads data/fake/*.pdf into the ragplay_test schema (fresh), runs `categorize` with the real LLM,
then compares every purchase/refund's category with the generator's JSON sidecar.

Usage: python scripts/phase3_eval_categories.py [--fake data/fake] [--model qwen3:8b]
"""

import argparse
import json
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

from ragplay import categorize, db, ingest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fake", type=Path, default=Path("data/fake"))
    ap.add_argument("--model", default=categorize.MODEL)
    args = ap.parse_args()

    db.migrate(db.TEST_SCHEMA)
    with db.connect_owner(db.TEST_SCHEMA) as conn:
        conn.autocommit = True
        conn.execute("TRUNCATE statements, merchant_aliases, merchant_categories CASCADE")
        for pdf in sorted(args.fake.glob("*.pdf")):
            r = ingest.ingest_file(conn, pdf)
            assert r.status == "loaded", (pdf.name, r.errors)
        t0 = time.time()
        st = categorize.categorize(conn, args.model)
        secs = time.time() - t0

        truth = Counter()
        for js in sorted(args.fake.glob("*.json")):
            for t in json.loads(js.read_text())["transactions"]:
                if t["type"] in ("purchase", "refund"):
                    truth[(t["txn_date"], t["description"], Decimal(t["amount"]), t["category"])] += 1
        got = Counter(
            (str(d), desc, amt, cat) for d, desc, amt, cat in conn.execute(
                "SELECT txn_date, description_raw, amount, category FROM transactions"
                " WHERE type IN ('purchase', 'refund')"))
        merchants = conn.execute(
            "SELECT a.merchant_key, m.merchant, m.category, m.source, m.description"
            " FROM merchant_aliases a JOIN merchant_categories m USING (merchant)"
            " ORDER BY 1").fetchall()

    total = sum(truth.values())
    right = sum((truth & got).values())
    print(f"{st.keys} keys: {st.rule} rule, {st.llm} LLM in {secs:.0f}s")
    print(f"transaction category accuracy: {right}/{total} = {right / total:.1%}")
    by_key = {}
    for (d, desc, amt, cat), n in truth.items():
        k = categorize.merchant_key(desc.removeprefix("REFUND "))
        by_key.setdefault(k, cat)
    print(f"\n{'key':42} {'merchant':24} {'got':14} {'truth':14} src   description")
    for key, name, cat, src, desc in merchants:
        want = by_key.get(key, "?")
        mark = "  " if cat == want else "✗ "
        print(f"{mark}{key[:40]:40} {name[:24]:24} {cat:14} {want:14} {src:5} {desc}")


if __name__ == "__main__":
    main()
