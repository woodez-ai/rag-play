"""Phase 0 baseline: paste one statement's text into the LLM and ask money questions.

Shows why the LLM must not do arithmetic: answers are checked against ground truth from the
synthetic generator's JSON sidecar.

Usage: python scripts/baseline.py data/fake/acme_4821_2026-03.pdf [--model qwen3:8b] [--runs 3]
"""

import argparse
import json
import re
import time
from decimal import Decimal
from pathlib import Path

import ollama
import pdfplumber

NUM_CTX = 16384


def truths(t: dict) -> list[tuple[str, Decimal]]:
    tx = t["transactions"]
    total = lambda cats: sum(
        (Decimal(x["amount"]) for x in tx if x["type"] == "purchase" and x["category"] in cats),
        Decimal(0),
    )
    purchases = [Decimal(x["amount"]) for x in tx if x["type"] == "purchase"]
    return [
        ("What is the total amount spent on groceries (supermarkets)?", total({"groceries"})),
        ("What is the total spent on dining, i.e. restaurants, food delivery and coffee shops?",
         total({"dining", "coffee"})),
        ("How many separate coffee shop purchases are there?",
         Decimal(sum(1 for x in tx if x["category"] == "coffee" and x["type"] == "purchase"))),
        ("What is the total spent on gas stations?", total({"gas"})),
        ("What was the single largest purchase amount?", max(purchases)),
    ]


def ask(model: str, text: str, question: str) -> tuple[Decimal | None, str, float]:
    prompt = (
        f"Here is my credit card statement:\n\n{text}\n\n"
        f"Question: {question}\n"
        "Answer with a short explanation, then a final line of the form `ANSWER: <number>`."
    )
    t0 = time.time()
    # Ollama's default context (4096) is smaller than statement + qwen3 thinking, and it
    # truncates silently; the window must hold the whole statement for a fair test.
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0, "num_ctx": NUM_CTX})
    if resp.prompt_eval_count and resp.prompt_eval_count >= NUM_CTX:
        raise RuntimeError(f"prompt filled the {NUM_CTX}-token context; it was truncated")
    content = resp.message.content
    m = re.findall(r"ANSWER:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", content)
    value = Decimal(m[-1].replace(",", "")) if m else None
    return value, content, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--runs", type=int, default=1, help="repeat each question to see variance")
    args = ap.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    truth = json.loads(args.pdf.with_suffix(".json").read_text())

    correct = total = 0
    for question, expected in truths(truth):
        for run in range(args.runs):
            got, _, secs = ask(args.model, text, question)
            ok = got is not None and abs(got - expected) < Decimal("0.01")
            correct += ok
            total += 1
            err = "" if got is None else f"  (off by {got - expected:+,.2f})"
            print(f"[{'OK ' if ok else 'BAD'}] {question}\n"
                  f"      expected {expected:,.2f}  got {got}{'' if ok else err}  {secs:.0f}s",
                  flush=True)
    print(f"\n{args.model}: {correct}/{total} exact on {args.pdf.name}")


if __name__ == "__main__":
    main()
