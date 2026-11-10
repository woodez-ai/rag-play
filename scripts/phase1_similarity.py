"""Phase 1: where semantic search helps, and where it fails.

Experiments (synthetic merchant strings only, no real data):
  1. Concept search on RAW statement descriptors vs CLEAN names vs ENRICHED (name + description),
     with and without nomic task prefixes, across embedding models
  2. Why raw fails: similarity is driven by string shape, not meaning
  3. Number blindness: can embeddings tell $4 from $400?
  4. Date blindness: can they tell March from May?
  5. Filter questions: "purchases over $100" as a vector query

Usage: python scripts/phase1_similarity.py [--models nomic-embed-text mxbai-embed-large]
"""

import argparse

import numpy as np
from rich.console import Console
from rich.table import Table

from ragplay.embeddings import cosine, embed, top_k

console = Console()

# (raw descriptor as printed on a statement, clean name, short description, true category)
# Clean names/descriptions are what Phase 3 (normalisation + LLM enrichment) will produce.
MERCHANTS = [
    ("SQ *BLUE BOTTLE COF 0423 OAKLAND CA", "Blue Bottle Coffee", "specialty coffee roaster and cafe", "coffee"),
    ("STARBUCKS STORE 02609 SPRINGFIELD", "Starbucks", "coffee shop chain", "coffee"),
    ("PEET'S #1142 BERKELEY CA", "Peet's Coffee", "coffee and tea shop", "coffee"),
    ("DUNKIN #350112 Q35 BOSTON MA", "Dunkin'", "coffee and donut shop", "coffee"),
    ("SAFEWAY #4003 SPRINGFIELD ST", "Safeway", "supermarket", "groceries"),
    ("TRADER JOE'S #4326 SPRINGFIELD ST", "Trader Joe's", "grocery store chain", "groceries"),
    ("WHOLE FOODS MKT #00591 SPRINGFIELD", "Whole Foods Market", "natural foods supermarket", "groceries"),
    ("KROGER #0617 COLUMBUS OH", "Kroger", "supermarket chain", "groceries"),
    ("SHELL OIL 00000003144 SPRINGFIELD ST", "Shell", "gas station", "gas"),
    ("CHEVRON 0007239 SPRINGFIELD ST", "Chevron", "gas station", "gas"),
    ("EXXONMOBIL 4471 HOUSTON TX", "ExxonMobil", "gas station", "gas"),
    ("TST* LUIGI'S TRATTORIA SPRINGFIELD", "Luigi's Trattoria", "Italian restaurant", "dining"),
    ("DOORDASH*THAI BASIL SAN FRANCISCO CA", "DoorDash - Thai Basil", "food delivery from a Thai restaurant", "dining"),
    ("CHIPOTLE 2251 SPRINGFIELD ST", "Chipotle", "fast casual Mexican restaurant", "dining"),
    ("THE HOME DEPOT #4012 SPRINGFIELD", "The Home Depot", "home improvement and hardware store", "hardware"),
    ("LOWES #01734 SPRINGFIELD ST", "Lowe's", "home improvement store", "hardware"),
    ("ACE HARDWARE 8812 SPRINGFIELD", "Ace Hardware", "hardware store", "hardware"),
    ("NETFLIX.COM LOS GATOS CA", "Netflix", "video streaming subscription", "streaming"),
    ("SPOTIFY USA 877-778-1161 NY", "Spotify", "music streaming subscription", "streaming"),
    ("HULU 877-8244858 CA", "Hulu", "TV streaming subscription", "streaming"),
    ("UBER *TRIP HELP.UBER.COM CA", "Uber", "ride-hailing trip", "transport"),
    ("LYFT *RIDE SUN 8PM SAN FRANCISCO", "Lyft", "ride-hailing trip", "transport"),
    ("AMAZON.COM*640BE0F AMZN.COM/BILL WA", "Amazon", "online retailer", "shopping"),
    ("CVS/PHARMACY #05512 SPRINGFIELD ST", "CVS Pharmacy", "pharmacy and drugstore", "pharmacy"),
    ("WALGREENS #11902 SPRINGFIELD", "Walgreens", "pharmacy and drugstore", "pharmacy"),
]

# Queries deliberately avoid the exact category words where possible, so a hit means the
# model understood meaning, not that it matched a word.
QUERIES = [
    ("latte", "coffee"),
    ("buying food for the week", "groceries"),
    ("filling up the car", "gas"),
    ("power drill and lumber", "hardware"),
    ("watching TV shows", "streaming"),
    ("taxi ride", "transport"),
    ("prescription medicine", "pharmacy"),
    ("dinner out", "dining"),
]

VARIANTS = {
    "raw": lambda m: m[0],
    "clean": lambda m: m[1],
    "enriched": lambda m: f"{m[1]}: {m[2]}",
}


def mean_p_at_k(model: str, variant: str, prefixed: bool, k: int = 3):
    docs = embed([VARIANTS[variant](m) for m in MERCHANTS], model,
                 "document" if prefixed else "raw")
    qs = embed([q for q, _ in QUERIES], model, "query" if prefixed else "raw")
    per_query = []
    for (q, cat), qv in zip(QUERIES, qs):
        hits = top_k(qv, docs, k)
        per_query.append((q, cat, sum(MERCHANTS[i][3] == cat for i, _ in hits) / k, hits))
    return float(np.mean([p for _, _, p, _ in per_query])), per_query


def exp1(models: list[str]):
    console.rule("[bold]1. Concept search: mean precision@3 over 8 queries (random ≈ 0.13)")
    t = Table("model", *[f"{v}" for v in VARIANTS], *[f"{v} (no prefix)" for v in VARIANTS])
    detail = None
    for model in models:
        cells = [f"{mean_p_at_k(model, v, True)[0]:.2f}" for v in VARIANTS]
        cells += [f"{mean_p_at_k(model, v, False)[0]:.2f}" for v in VARIANTS]
        t.add_row(model, *cells)
        detail = detail or (model, mean_p_at_k(model, "enriched", True)[1])
    console.print(t)
    console.print("(prefixes only differ for models that define them, e.g. nomic-embed-text)")

    for variant in ("raw", "enriched"):
        model = models[0]
        _, rows = mean_p_at_k(model, variant, True)
        console.rule(f"[dim]{model}, {variant}: top 3 per query")
        t = Table("query", "top 3 (cosine)", "P@3")
        for q, cat, p, hits in rows:
            t.add_row(q, "\n".join(
                f"{'✓' if MERCHANTS[i][3] == cat else '✗'} {s:.3f}  {VARIANTS[variant](MERCHANTS[i])}"
                for i, s in hits), f"{p:.2f}")
        console.print(t)


def similarity_table(title: str, anchor: str, others: list[str], model: str):
    console.rule(f"[bold]{title}")
    vecs = embed([anchor, *others], model, "document")
    sims = cosine(vecs[:1], vecs[1:])[0]
    t = Table("compared with", "cosine")
    t.add_row(f"[dim]anchor:[/] {anchor}", "")
    for text, s in sorted(zip(others, sims), key=lambda x: -x[1]):
        t.add_row(text, f"{s:.4f}")
    console.print(t)


def exp5_filter_query(model: str):
    console.rule("[bold]5. 'purchases over $100' as a vector query")
    docs = [
        "2026-03-02 STARBUCKS STORE 02609 $5.40",
        "2026-03-04 THE HOME DEPOT #4012 $243.18",
        "2026-03-09 SAFEWAY #4003 $101.50",
        "2026-03-11 NETFLIX.COM $17.99",
        "2026-03-14 DELTA AIR LINES $612.30",
        "2026-03-20 CHIPOTLE 2251 $13.85",
        "2026-03-22 SHELL OIL $64.10",
        "2026-03-27 AMAZON.COM $100.00",
    ]
    vecs = embed(docs, model, "document")
    hits = top_k(embed(["purchases over $100"], model, "query"), vecs, len(docs))
    t = Table("rank", "cosine", "transaction", "really > $100?")
    for r, (i, s) in enumerate(hits, 1):
        amt = float(docs[i].rsplit("$", 1)[1])
        t.add_row(str(r), f"{s:.3f}", docs[i], "yes" if amt > 100 else "[red]no[/]")
    console.print(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nomic-embed-text", "mxbai-embed-large"])
    args = ap.parse_args()
    for m in args.models:
        console.print(f"Model: [bold]{m}[/], dim = {embed(['x'], m).shape[1]}")

    exp1(args.models)
    m = args.models[0]
    similarity_table("2. Why raw fails: string shape beats meaning", "STARBUCKS STORE 02609 SPRINGFIELD",
                     ["SHELL OIL 00000003144 SPRINGFIELD ST", "Starbucks", "coffee", "coffee shop",
                      "PEET'S #1142 BERKELEY CA"], m)
    similarity_table("3. Number blindness", "BLUE BOTTLE COFFEE $4.50",
                     ["BLUE BOTTLE COFFEE $4.75", "BLUE BOTTLE COFFEE $450.00",
                      "BLUE BOTTLE COFFEE $45,000.00", "STARBUCKS $4.50", "SHELL OIL $4.50"], m)
    similarity_table("4. Date blindness", "2026-03-14 BLUE BOTTLE COFFEE $6.50",
                     ["2026-03-15 BLUE BOTTLE COFFEE $6.50", "2026-05-14 BLUE BOTTLE COFFEE $6.50",
                      "2019-03-14 BLUE BOTTLE COFFEE $6.50", "2026-03-14 STARBUCKS $6.50",
                      "2026-03-14 SHELL OIL $6.50"], m)
    exp5_filter_query(m)


if __name__ == "__main__":
    main()
