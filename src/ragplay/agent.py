"""Tool-calling agent: the LLM picks how to answer, the tools do the work.

Tools:
  query_database(question)    -> Phase 5 text-to-SQL (totals, counts, averages, trends, rankings)
  find_transactions(filters…) -> the matching transaction ROWS (show/list/which… questions)
Declining is a first-class option: if the data can't answer, say so (no tool needed).

Loop: system prompt + question -> model either calls tools (results are fed back) or answers.
Max MAX_STEPS tool rounds. The final answer is checked in code: every number in it must appear
in some tool output or in the question.

Conversation memory (`rag chat`): a follow-up like "and in April?" is first rewritten into a
standalone question using the previous turns, then answered like any other question.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

import ollama
import psycopg

from ragplay import anomalies, budgets, store, text2sql
from ragplay.text2sql import Attempt

MODEL = "qwen3:8b"
MAX_STEPS = 4
NUM_CTX = 8192
ROWS_TO_MODEL = 40

TOOLS = [
    {"type": "function", "function": {
        "name": "query_database",
        # No arguments: the tool always gets the user's own question. When it took a question,
        # qwen3 paraphrased "on my March 2026 statement" to "in March 2026" (a different
        # period) even when told to pass it word for word.
        "description": "Answer the user's question with a database query when it needs "
                       "CALCULATION over the transactions: totals, sums, counts, averages, "
                       "maximums, comparisons, rankings, per-month/category breakdowns, balances. "
                       "The database computes the numbers exactly.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "find_transactions",
        "description": "Return the individual transactions (date, merchant, amount) that match "
                       "filters (all filters must match). Use for 'show me', 'list', 'which "
                       "purchases', 'when did I' questions. Not for largest/smallest/most/top/"
                       "totals: those are calculations. Use ONE of category / merchant / text "
                       "for what was bought: category for a kind of business, merchant only when "
                       "the user names a specific business, text only when neither fits.",
        "parameters": {"type": "object", "properties": {
            "merchant": {"type": "string", "description": "a business the user named, e.g. 'amazon'"},
            "category": {"type": "string", "description": "one of the categories listed in the system prompt"},
            "date_from": {"type": "string", "description": "YYYY-MM-DD, inclusive"},
            "date_to": {"type": "string", "description": "YYYY-MM-DD, inclusive"},
            "min_amount": {"type": "number", "description": "e.g. 100 for 'over $100'"},
            "max_amount": {"type": "number"},
            "text": {"type": "string", "description": "free-text description, e.g. 'baseball game'"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "detect_recurring",
        "description": "Recurring charges (subscriptions, bills, memberships) found from regular "
                       "intervals and similar amounts: merchant, cadence, current amount, whether "
                       "still active, PRICE CHANGES, next expected charge. Use for 'subscriptions', "
                       "'recurring', 'regularly', 'price went up', 'forgotten subscriptions'.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string", "description": "only this category, e.g. 'subscriptions'"},
            "active_only": {"type": "boolean", "description": "only series still being charged"},
            "price_increased_only": {"type": "boolean",
                                     "description": "only series whose price went UP; use for "
                                                    "'went up', 'price increase' questions"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "find_duplicates",
        "description": "Possible duplicate charges: same merchant and same amount within `days` "
                       "days (0 = same day).",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "max days apart, default 0"},
            "date_from": {"type": "string"}, "date_to": {"type": "string"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "spend_outliers",
        "description": "Unusually large purchases compared with the user's own history: more than "
                       "3x the usual amount for their category and among their largest overall.",
        "parameters": {"type": "object", "properties": {
            "date_from": {"type": "string"}, "date_to": {"type": "string"}},
            "required": []}}},
]
TOOLS.append({"type": "function", "function": {
    "name": "budget_status",
    # Returns EVERY budget for the month. It used to take a category, and the model passed
    # category='all' for "my budgets" ('all' = the total budget), so it missed dining going over.
    "description": "Every monthly budget vs actual spending, computed in code: budget, spent, "
                   "remaining, % used, projected month-end (if the data stops mid-month) and status "
                   "(over / at risk / on track). Use for any question about budgets. The row with "
                   "category 'all' is the budget for TOTAL spending.",
    "parameters": {"type": "object", "properties": {
        "month": {"type": "string", "description": "YYYY-MM; omit for the current month"}},
        "required": []}}})
ANOMALY_TOOLS = {"detect_recurring", "find_duplicates", "spend_outliers", "budget_status"}

# What each category covers (same meanings Phase 3 used to assign them). Without this the model
# mapped "hardware store" to 'shopping' instead of 'home'.
CATEGORY_GUIDE = """coffee = coffee shops; dining = restaurants, fast food, bars, cafes, food delivery;
groceries = supermarkets; gas = gas stations; transport = transit, taxis, parking, tolls, car
dealers and repair; travel = hotels, airlines, resorts; shopping = general retail and online
stores; home = hardware stores, furniture, home improvement; health = pharmacies, clinics, gyms;
subscriptions = software, apps, streaming, web hosting; utilities = phone, mobile, internet,
electricity; entertainment = events, museums, sports, lottery; alcohol = liquor/beer/wine stores;
payment = card payments."""

SYSTEM = """You answer questions about the user's credit card transactions ({first} to {last}).
Today is {today}. Categories in the data: {categories}.
What they cover: {guide}

Rules:
- Never calculate anything yourself. Use query_database for any total, count, average, maximum,
  comparison or ranking; use find_transactions to show or list individual transactions.
- Subscriptions, recurring charges: detect_recurring. Price increases: detect_recurring with
  price_increased_only=true (if it returns no rows, nothing went up). Duplicate charges:
  find_duplicates. Unusual or unexpectedly large purchases: spend_outliers. Budgets:
  budget_status. These are computed by code; report what they return and don't add your own
  judgement. If budget_status returns no rows, no budget is set for that.
- Quote numbers exactly as the tools return them.
- The data contains this card's transactions and its monthly statements (balances and totals
  of purchases, payments, fees and interest charged). Questions about those are answerable,
  even if the answer is zero. If a question can't be answered from that (e.g. credit score,
  other cards, bank accounts, dates outside the range above), say you don't have that data.
  Do not call a tool for it.
- When listing transactions, give date, merchant and amount for each; if there are many, give
  the count and the first several.
- Answer briefly."""

REWRITE = """Rewrite the user's last message as a single standalone question, using the
conversation for missing details (time period, merchant, category). Do not answer it.
If it is already standalone, return it unchanged. Reply with the question only.

Conversation:
{history}
Last message: {question}"""


@dataclass
class ToolCall:
    name: str
    arguments: dict
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    sql: str = ""
    error: str | None = None
    seconds: float = 0.0


@dataclass
class AgentAnswer:
    question: str
    standalone: str
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    seconds: float = 0.0
    sql_first_try: bool | None = None

    @property
    def tools(self) -> list[str]:
        return [c.name for c in self.calls]

    @property
    def final(self) -> Attempt | None:
        """The last successful tool result, in the shape the evaluator scores."""
        ok = [c for c in self.calls if c.error is None]
        if not ok:
            return None
        c = ok[-1]
        return Attempt(c.sql or c.name, columns=c.columns, rows=c.rows)


def _chat(messages: list[dict], model: str, tools: list | None = None):
    resp = ollama.chat(model=model, messages=messages, tools=tools, think=False,
                       options={"temperature": 0, "num_ctx": NUM_CTX})
    if resp.prompt_eval_count and resp.prompt_eval_count >= NUM_CTX:
        raise RuntimeError("prompt filled the context window; it was truncated")
    return resp.message


def _date(s):
    return date.fromisoformat(s) if s else None


def _dec(x):
    return None if x in (None, "") else Decimal(str(x))


RELAX = ("text", "merchant")  # dropped in this order when a find with redundant filters is empty


def _find(conn, args: dict, model: str):
    return store.find(conn, text=args.get("text") or None, merchant=args.get("merchant") or None,
                      category=args.get("category") or None, date_from=_date(args.get("date_from")),
                      date_to=_date(args.get("date_to")), min_amount=_dec(args.get("min_amount")),
                      max_amount=_dec(args.get("max_amount")), model=store.DEFAULT_MODEL)


def run_tool(conn: psycopg.Connection, name: str, args: dict, *, model: str, today: date,
             answer: AgentAnswer) -> ToolCall:
    call = ToolCall(name, args)
    try:
        if name == "query_database":
            sub = text2sql.ask(conn, answer.standalone, model=model, today=today, phrase=False)
            if answer.sql_first_try is None and sub.attempts:
                answer.sql_first_try = sub.attempts[0].error is None
            final = sub.final
            if final is None:
                call.error = "could not produce a working query: " + (sub.attempts[-1].error or "")
            else:
                call.sql, call.columns, call.rows = final.sql, final.columns, final.rows
        elif name == "find_transactions":
            unknown = set(args) - set(next(t for t in TOOLS if t["function"]["name"] == name)
                                      ["function"]["parameters"]["properties"])
            if unknown:
                # qwen3 invented sort/limit to answer "largest purchase" by reading rows itself
                call.error = (f"unknown argument(s) {sorted(unknown)}. find_transactions only "
                              "filters; for largest/smallest/most/top/totals use query_database.")
                return call
            hits = _find(conn, args, model)
            # The model often stacks category + merchant ("home" + "hardware store") and gets
            # nothing. Relax deterministically: drop text, then merchant, while a category remains.
            for key in RELAX:
                if hits or not args.get("category") or not args.get(key):
                    continue
                args = {k: v for k, v in args.items() if k != key}
                hits = _find(conn, args, model)
                call.arguments = {**args, "relaxed": f"dropped {key}"}
            call.columns = ["id", "txn_date", "merchant", "amount", "category"]
            call.rows = [(h.id, h.txn_date, h.merchant, h.amount, h.category) for h in hits]
        elif name in ANOMALY_TOOLS:
            known = set(next(t for t in TOOLS if t["function"]["name"] == name)
                        ["function"]["parameters"]["properties"])
            args = {k: v for k, v in args.items() if k in known and v not in (None, "")}
            if name == "budget_status":
                month, items = budgets.status(conn, args.get("month") or None, today=today)
                if items and all(b.data_through is None for b in items):
                    # "this month" with no data yet read as "on track, nothing spent"
                    last = conn.execute("SELECT max(txn_date) FROM transactions").fetchone()[0]
                    call.error = (f"no transactions recorded for {month} (the data ends {last}); "
                                  "budget status can't be judged for that month")
                    return call
                call.columns, call.rows = budgets.rows(items)
                call.columns.insert(0, "month")
                call.rows = [(month, *r) for r in call.rows]
            elif name == "detect_recurring":
                series = anomalies.detect_recurring(conn, category=args.get("category"))
                if args.get("active_only"):
                    series = [s for s in series if s.active]
                if args.get("price_increased_only"):
                    series = [s for s in series if s.increased]
                call.columns, call.rows = anomalies.recurring_rows(series)
            elif name == "find_duplicates":
                pairs = anomalies.find_duplicates(conn, days=int(args.get("days", 0)),
                                                  date_from=_date(args.get("date_from")),
                                                  date_to=_date(args.get("date_to")))
                call.columns, call.rows = anomalies.duplicate_rows(pairs, conn)
            else:
                call.columns, call.rows = anomalies.outlier_rows(anomalies.spend_outliers(
                    conn, date_from=_date(args.get("date_from")), date_to=_date(args.get("date_to"))))
        else:
            call.error = f"unknown tool {name!r}"
    except (ValueError, TypeError, psycopg.Error) as e:
        call.error = f"{type(e).__name__}: {e}"
        if isinstance(e, psycopg.Error) and not conn.autocommit:
            conn.rollback()
    return call


def _tool_text(call: ToolCall) -> str:
    if call.error:
        return f"ERROR: {call.error}"
    if not call.rows:
        if call.name == "find_transactions" and len(call.arguments) > 1:
            return ("No rows. All filters must match; if you combined category, merchant or text, "
                    "try again with fewer (merchant names rarely contain words like 'store').")
        return "No rows."
    shown = [r for r in call.rows[:ROWS_TO_MODEL]]
    cols = [c for c in call.columns if c != "id"]  # ids are for scoring, not for the model
    idx = [i for i, c in enumerate(call.columns) if c != "id"]
    body = text2sql.result_text(cols, [tuple(r[i] for i in idx) for r in shown])
    more = f"\n({len(call.rows)} rows in total; first {ROWS_TO_MODEL} shown)" if len(call.rows) > ROWS_TO_MODEL else \
        f"\n({len(call.rows)} rows)"
    return body + more


REVIEW = """Write a short monthly credit card review for {month} from the FACTS below (JSON,
computed by code). 4 to 8 bullet points, most important first: how total spending compares
with the usual, the biggest category changes, price changes, new recurring charges, recurring
charges that didn't appear, possible duplicates, unusual purchases, notable new merchants,
budgets that are over or at risk.
Rules: use ONLY these facts; copy numbers exactly; do not add, subtract or estimate anything;
skip empty sections; end with at most two things worth checking.

FACTS:
{facts}"""


def review(conn: psycopg.Connection, month: str | None = None, *, model: str = MODEL) -> tuple[dict, str, list[str]]:
    """Deterministic facts for the month -> LLM narrative -> numbers in the narrative that are
    not in the facts. Returns (facts, text, unsupported)."""
    facts = anomalies.monthly_review(conn, month)
    blob = json.dumps(facts, default=str, indent=1)
    text = (_chat([{"role": "user", "content": REVIEW.format(month=facts["month"], facts=blob)}],
                  model).content or "").strip()
    return facts, text, text2sql.unsupported_numbers(text, blob, [], [])


def rewrite(question: str, history: list[tuple[str, str]], model: str) -> str:
    if not history:
        return question
    convo = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history[-3:])
    msg = _chat([{"role": "user", "content": REWRITE.format(history=convo, question=question)}], model)
    return msg.content.strip().strip('"') or question


# Deterministic routing for calculation questions. With 2 tools qwen3 routed these well; with 5
# it started answering "how much / how many / largest" with find_transactions and doing the
# arithmetic itself (sample totals 9/9 -> 6/9, one total off by $30). A rule decides reliably,
# so the LLM doesn't get the choice. Anything else (show/list, subscriptions, duplicates,
# unusual purchases, unanswerable) still goes to the agent.
CALC_RE = re.compile(
    r"\b(how much|how many|total|sum|average|avg|mean|largest|smallest|biggest|highest|lowest|"
    r"most|least|more than .* than|less than .* than|compare|per month|which (month|category|"
    r"merchant|year)|balance|interest|count)\b", re.I)
# Questions for the agent's own tools, even if they contain calculation words
# ("how much of my dining budget is left" -> budget_status, not text-to-SQL)
ANOMALY_RE = re.compile(r"\b(subscriptions?|recurring|regularly|duplicates?|unusual|went up|"
                        r"price (increase|change)|show|list|budgets?)\b", re.I)


def _answer_with_sql(conn: psycopg.Connection, ans: AgentAnswer, *, model: str,
                     today: date) -> AgentAnswer | None:
    """Text-to-SQL straight away; None if no working query (then the agent gets a turn)."""
    sub = text2sql.ask(conn, ans.standalone, model=model, today=today)
    if sub.attempts:
        ans.sql_first_try = sub.attempts[0].error is None
    final = sub.final
    if final is None:
        return None
    ans.calls.append(ToolCall("query_database", {"routed": "calculation"}, final.columns,
                              final.rows, final.sql, seconds=sub.seconds))
    ans.text, ans.unsupported = sub.text, sub.unsupported
    return ans


def ask(conn: psycopg.Connection, question: str, *, model: str = MODEL, today: date | None = None,
        history: list[tuple[str, str]] | None = None) -> AgentAnswer:
    t0 = time.time()
    today = today or date.today()
    standalone = rewrite(question, history or [], model)
    ans = AgentAnswer(question, standalone)
    if CALC_RE.search(standalone) and not ANOMALY_RE.search(standalone):
        routed = _answer_with_sql(conn, ans, model=model, today=today)
        if routed is not None:
            routed.seconds = time.time() - t0
            return routed
    first, last = conn.execute("SELECT min(txn_date), max(txn_date) FROM transactions").fetchone()
    system = SYSTEM.format(first=first, last=last, today=today, guide=CATEGORY_GUIDE,
                           categories=", ".join(text2sql.categories(conn)))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": standalone}]
    for _ in range(MAX_STEPS):
        msg = _chat(messages, model, TOOLS)
        if not msg.tool_calls:
            ans.text = (msg.content or "").strip()
            break
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            t_call = time.time()
            call = run_tool(conn, tc.function.name, dict(tc.function.arguments or {}),
                            model=model, today=today, answer=ans)
            call.seconds = time.time() - t_call
            ans.calls.append(call)
            messages.append({"role": "tool", "content": _tool_text(call), "tool_name": call.name})
    else:
        messages.append({"role": "user", "content": "Answer now from the tool results above."})
        ans.text = (_chat(messages, model).content or "").strip()
    if ans.text:
        # a number is supported if any tool output, the question, or the facts given in the system
        # prompt (data range, today) contain it; row counts count too
        context = f"{standalone} {first} {last} {today}"
        per_call = [set(text2sql.unsupported_numbers(ans.text, context, c.columns, c.rows))
                    for c in ans.calls] or [set(text2sql.unsupported_numbers(ans.text, context, [], []))]
        ans.unsupported = sorted(set.intersection(*per_call))
    ans.seconds = time.time() - t0
    return ans
