"""Phase 2 comparison: LLM structured extraction vs the deterministic parser.

For each statement, every page goes to a local Ollama model, which must return transactions
matching a JSON schema (Ollama `format=`). Two modes:
- text:   the page's extracted text -> qwen3:8b
- vision: the page rendered as an image -> qwen3-vl:8b-instruct (the fallback for pages with no
          usable text layer)

Reference = the deterministic parser's rows, which reconcile to the cent against the statement
summary. The LLM's rows are scored by exact-row precision/recall and run through the same
reconcile checks.

Prints counts and rates only (safe for real statements). Row-level diffs are written to
data/phase2_llm/ (gitignored).

Usage: python scripts/phase2_llm_extract.py data/fake/acme_4821_2026-01.pdf ... [--mode text|vision|both]
"""

import argparse
import dataclasses
import io
import json
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import ollama
import pdfplumber

from ragplay.parsers import ParsedTxn, classify, parse_file
from ragplay.validate import reconcile

NUM_CTX = 8192
MODELS = {"text": "qwen3:8b", "vision": "qwen3-vl:8b-instruct"}
OUT = Path("data/phase2_llm")

SCHEMA = {
    "type": "object",
    "properties": {
        "transactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # `pattern` is enforced by Ollama's constrained decoding; a `description`
                    # alone is not (qwen3 copied "01/07" despite "YYYY-MM-DD"). Ollama 0.20's
                    # grammar rejects `\d` ("failed to load model vocabulary"), so use [0-9].
                    "txn_date": {"type": "string", "pattern": "^20[0-9]{2}-[0-9]{2}-[0-9]{2}$"},
                    "post_date": {"type": "string", "pattern": "^20[0-9]{2}-[0-9]{2}-[0-9]{2}$"},
                    "description": {"type": "string"},
                    "amount": {"type": "string", "pattern": "^-?[0-9]{1,6}[.][0-9]{2}$"},
                },
                "required": ["txn_date", "post_date", "description", "amount"],
            },
        }
    },
    "required": ["transactions"],
}

INSTRUCTIONS = (
    "Extract every transaction row from this credit card statement page. "
    "The statement period is {start} to {end}; dates print without a year, so use that period "
    "to choose the year. Copy the description exactly as printed, including any wrapped second "
    "line, but not reference numbers or foreign-currency/exchange-rate lines. Amounts: charges "
    "positive, payments and credits negative, no $ sign or commas. Ignore summary boxes, "
    "totals, interest-rate tables and the payment slip. If the page has no transaction rows, "
    "return an empty list."
)


def extract_page(mode: str, page, start, end) -> tuple[list[dict], float]:
    prompt = INSTRUCTIONS.format(start=start, end=end)
    msg = {"role": "user", "content": prompt}
    if mode == "text":
        msg["content"] = f"{prompt}\n\n--- PAGE TEXT ---\n{page.extract_text()}"
    else:
        buf = io.BytesIO()
        page.to_image(resolution=150).original.save(buf, format="PNG")
        msg["images"] = [buf.getvalue()]
    t0 = time.time()
    kwargs = {"think": False} if mode == "text" else {}
    resp = ollama.chat(model=MODELS[mode], messages=[msg], format=SCHEMA,
                       options={"temperature": 0, "num_ctx": NUM_CTX}, **kwargs)
    if resp.prompt_eval_count and resp.prompt_eval_count >= NUM_CTX:
        raise RuntimeError("prompt filled the context window; it was truncated")
    return json.loads(resp.message.content)["transactions"], time.time() - t0


def to_txn(row: dict, card: str) -> ParsedTxn | None:
    try:
        amount = Decimal(row["amount"].replace("$", "").replace(",", "")).quantize(Decimal("0.01"))
        from datetime import date
        return ParsedTxn(date.fromisoformat(row["txn_date"]), date.fromisoformat(row["post_date"]),
                         " ".join(row["description"].split()), amount,
                         classify(row["description"], amount), card)
    except (ValueError, ArithmeticError, KeyError, AttributeError):
        return None


def key(t: ParsedTxn, strict: bool):
    return (t.txn_date, t.post_date, t.description, t.amount) if strict else (t.txn_date, t.amount)


def score(ref: list[ParsedTxn], got: list[ParsedTxn], strict: bool) -> tuple[float, float]:
    r, g = Counter(key(t, strict) for t in ref), Counter(key(t, strict) for t in got)
    hit = sum((r & g).values())
    return hit / max(len(got), 1), hit / max(len(ref), 1)


def run(pdf: Path, mode: str) -> dict:
    ref = parse_file(pdf)
    rows, secs, bad = [], 0.0, 0
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            out, s = extract_page(mode, page, ref.period_start, ref.period_end)
            secs += s
            for r in out:
                t = to_txn(r, ref.card_last4)
                rows.append(t) if t else None
                bad += t is None
    llm = dataclasses.replace(ref, transactions=rows)
    p, r = score(ref.transactions, rows, strict=True)
    lp, lr = score(ref.transactions, rows, strict=False)
    OUT.mkdir(parents=True, exist_ok=True)
    diff = {
        "missed": sorted(map(str, (Counter(key(t, True) for t in ref.transactions)
                                   - Counter(key(t, True) for t in rows)).elements())),
        "extra": sorted(map(str, (Counter(key(t, True) for t in rows)
                                  - Counter(key(t, True) for t in ref.transactions)).elements())),
    }
    (OUT / f"{pdf.stem}.{mode}.json").write_text(json.dumps(diff, indent=2))
    return {
        "file": pdf.name[-14:], "mode": mode, "ref": len(ref.transactions), "llm": len(rows),
        "unparseable": bad, "exact_P": p, "exact_R": r, "date+amt_P": lp, "date+amt_R": lr,
        "reconciles": not reconcile(llm), "secs": secs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--mode", choices=["text", "vision", "both"], default="both")
    args = ap.parse_args()
    modes = ["text", "vision"] if args.mode == "both" else [args.mode]
    print("file            mode    ref  llm  bad  exactP exactR  d+aP  d+aR  recon  secs")
    for pdf in args.pdfs:
        for mode in modes:
            r = run(pdf, mode)
            print(f"{r['file']:15} {r['mode']:6} {r['ref']:4} {r['llm']:4} {r['unparseable']:4}"
                  f"  {r['exact_P']:.2f}   {r['exact_R']:.2f}  {r['date+amt_P']:.2f}  "
                  f"{r['date+amt_R']:.2f}  {'yes' if r['reconciles'] else 'NO':5} {r['secs']:5.0f}",
                  flush=True)


if __name__ == "__main__":
    main()
