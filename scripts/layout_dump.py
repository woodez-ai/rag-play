"""Print a statement PDF's layout with its contents masked, so the layout can be shared safely.

Every word is reduced to its shape (uppercase -> X, lowercase -> x, digit -> 9; punctuation kept)
unless it is on a short allowlist of generic statement labels. Names, addresses, merchants, card
numbers and amounts never appear, but column positions, date/amount formats, signs and "CR"
markers do. That is enough to write a parser.

Usage: python scripts/layout_dump.py "cc_statements/Visa Statement-2423 2026-10-21.pdf" [--pages 1-3]
"""

import argparse
import re

import pdfplumber

ALLOW = {
    w.lower()
    for w in """
    account accounts activity adjustments advance advances amount annual apr available balance
    balances billing card cash charge charged charges closing cr credit credits date dates days
    debit debits description detail details due end ending fee fees from interest last limit
    minimum new number of opening other page past payment payments percentage period post posted
    posting previous purchase purchases rate reference start statement summary total totals
    transaction transactions trans transfer transfers visa your and the to for on in at by
    jan feb mar apr may jun jul aug sep oct nov dec
    january february march april june july august september october november december
    """.split()
}


def shape(word: str) -> str:
    core = re.sub(r"[^A-Za-z]", "", word).lower()
    if core and core in ALLOW and not re.search(r"\d", word):
        return word
    return re.sub(r"\d", "9", re.sub(r"[a-z]", "x", re.sub(r"[A-Z]", "X", word)))


def page_range(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    a, _, b = spec.partition("-")
    return list(range(int(a) - 1, min(int(b or a), n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf")
    ap.add_argument("--pages")
    args = ap.parse_args()
    with pdfplumber.open(args.pdf) as pdf:
        print(f"pages={len(pdf.pages)} size={pdf.pages[0].width:.0f}x{pdf.pages[0].height:.0f}")
        for i in page_range(args.pages, len(pdf.pages)):
            page = pdf.pages[i]
            print(f"\n=== page {i + 1} ===")
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
            lines: dict[int, list] = {}
            for w in words:
                lines.setdefault(round(w["top"] / 2) * 2, []).append(w)
            for top in sorted(lines):
                ws = sorted(lines[top], key=lambda w: w["x0"])
                print(f"y{top:4d} " + "  ".join(f"[{w['x0']:.0f}]{shape(w['text'])}" for w in ws))


if __name__ == "__main__":
    main()
