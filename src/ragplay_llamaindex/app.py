"""LlamaIndex rebuild: SQL engine, vector index, FunctionAgent. See package docstring."""

import asyncio
import os
import time
from datetime import date

from llama_index.core import Settings, SQLDatabase, StorageContext, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.schema import TextNode
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import URL, create_engine

import ragplay_llamaindex  # noqa: F401  (sets local Ollama defaults)
from ragplay import anomalies, db, text2sql
from ragplay.agent import CATEGORY_GUIDE, AgentAnswer, ToolCall

LI_SCHEMA = "ragplay_li"  # PGVectorStore's own table lives here, not in the app schema
TABLES = ["statements", "transactions", "merchant_categories"]


def _env(name: str) -> str:
    db._conninfo("DB_USER", "DB_PASSWORD")  # loads .env and checks it's complete
    return os.environ[name]


def _url(driver: str, user_var: str, password_var: str, **query) -> URL:
    return URL.create(driver, username=_env(user_var), password=_env(password_var),
                      host=_env("DB_HOST"), database=_env("DB"), query=query)


def sql_engine(schema: str):
    """Read-only SQLAlchemy engine: rag_reader for the real data, the owner in a read-only
    session for the synthetic schema (rag_reader has no grants there)."""
    opts = (f"-c search_path={schema},public -c default_transaction_read_only=on "
            f"-c statement_timeout={db.READER_TIMEOUT_MS}")
    user, pw = ("DB_READER_USER", "DB_READER_PASSWORD") if schema == db.SCHEMA else ("DB_USER", "DB_PASSWORD")
    return create_engine(_url("postgresql+psycopg", user, pw, sslmode="require", options=opts))


def table_name(schema: str) -> str:
    return f"txn_{schema}"  # PGVectorStore adds a "data_" prefix


def vector_store(schema: str, setup: bool = False) -> PGVectorStore:
    """As the owner (it needs CREATE on its schema to set up); queries only read.
    PGVectorStore str()s the URL, and SQLAlchemy 2's URL.__str__ masks the password as '***'
    (auth failed), so render it explicitly. In memory only; never printed."""
    plain = lambda u: u.render_as_string(hide_password=False)  # noqa: E731
    return PGVectorStore.from_params(
        connection_string=plain(_url("postgresql+psycopg2", "DB_USER", "DB_PASSWORD", sslmode="require")),
        async_connection_string=plain(_url("postgresql+asyncpg", "DB_USER", "DB_PASSWORD", ssl="require")),
        schema_name=LI_SCHEMA, table_name=table_name(schema), embed_dim=1024, perform_setup=setup)


def build_index(schema: str = db.SCHEMA) -> int:
    """One TextNode per transaction: the same enriched text the hand-built store embeds, with
    date/amount/category/merchant as metadata for filters. Rebuilt from scratch each time."""
    with db.connect_owner() as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {LI_SCHEMA}")
        conn.execute(f"DROP TABLE IF EXISTS {LI_SCHEMA}.data_{table_name(schema)}")
    with db.connect_owner(schema) as conn:
        rows = conn.execute(
            """SELECT t.id, t.txn_date, t.merchant, t.amount, t.category, t.type,
                      coalesce(t.merchant || ': ' || m.description, t.merchant, t.description_raw)
               FROM transactions t LEFT JOIN merchant_categories m USING (merchant)""").fetchall()
    nodes = [TextNode(text=text, id_=f"{schema}-{tid}", metadata={
        "transaction_id": tid, "txn_date": d.isoformat(), "date_num": int(d.strftime("%Y%m%d")),
        "merchant": merchant or "", "amount": float(amount), "category": category or "",
        "type": ttype}) for tid, d, merchant, amount, category, ttype, text in rows]
    store = vector_store(schema, setup=True)
    VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store),
                     show_progress=False)
    return len(nodes)


class LlamaIndexAssistant:
    """Tools are thin wrappers that also record what they returned, so the shared evaluator
    can score rows exactly as it does for the hand-built agent."""

    def __init__(self, schema: str = db.SCHEMA, tuned: bool = False):
        """tuned=False: the framework's idiomatic defaults. tuned=True: the hand-built system's
        domain knowledge (conventions, categories, examples, today) passed through LlamaIndex's
        own hook, context_query_kwargs, to separate 'framework' from 'domain knowledge'."""
        self.schema = schema
        self.tuned = tuned
        self.sql_db = SQLDatabase(sql_engine(schema), schema=schema, include_tables=TABLES)
        self.sql = self._sql_engine(None)
        self.index = VectorStoreIndex.from_vector_store(vector_store(schema))
        self.calls: list[ToolCall] = []
        self.loop = asyncio.new_event_loop()

    def _sql_engine(self, today: date | None) -> NLSQLTableQueryEngine:
        ctx = None
        if self.tuned and today:
            with db.connect_owner(self.schema) if self.schema != db.SCHEMA else db.connect_reader() as conn:
                cats = text2sql.categories(conn)
            ctx = {"transactions": (
                f"category is one of: {', '.join(cats)} (lowercase). type is one of: purchase, refund, "
                "payment, fee, interest.\n" + text2sql.RULES.format(today=today) + "\n"
                + text2sql.EXAMPLES.format(today=today))}
        return NLSQLTableQueryEngine(self.sql_db, tables=TABLES, llm=Settings.llm,
                                     context_query_kwargs=ctx)

    # --- tools -------------------------------------------------------------------------------
    def query_database(self, question: str) -> str:
        """Answer a question that needs calculation over the credit card data (totals, counts,
        averages, maximums, comparisons, rankings, balances, interest)."""
        call = ToolCall("query_database", {"question": question})
        try:
            r = self.sql.query(question)
            call.sql = r.metadata.get("sql_query", "")
            call.rows = [tuple(x) for x in r.metadata.get("result", []) or []]
            call.columns = list(r.metadata.get("col_keys", []) or [f"c{i}" for i in range(len(call.rows[0]))] if call.rows else [])
            out = str(r)
        except Exception as e:  # the framework raises on bad SQL; report it to the agent
            call.error = f"{type(e).__name__}: {e}"
            out = f"ERROR: {call.error}"
        self.calls.append(call)
        return out

    def find_transactions(self, query: str, category: str | None = None, merchant: str | None = None,
                          date_from: str | None = None, date_to: str | None = None,
                          min_amount: float | None = None, top_k: int = 30) -> str:
        """Find individual transactions similar to `query` (a kind of purchase or a merchant),
        optionally filtered by exact category, merchant name, date range (YYYY-MM-DD) and
        minimum amount. Returns date, merchant and amount for each."""
        f = [MetadataFilter(key="type", value="payment", operator=FilterOperator.NE)]
        if category:
            f.append(MetadataFilter(key="category", value=category, operator=FilterOperator.EQ))
        if merchant:
            f.append(MetadataFilter(key="merchant", value=merchant, operator=FilterOperator.TEXT_MATCH_INSENSITIVE))
        if date_from:
            f.append(MetadataFilter(key="date_num", value=int(date_from.replace("-", "")), operator=FilterOperator.GTE))
        if date_to:
            f.append(MetadataFilter(key="date_num", value=int(date_to.replace("-", "")), operator=FilterOperator.LTE))
        if min_amount is not None:
            f.append(MetadataFilter(key="amount", value=float(min_amount), operator=FilterOperator.GTE))
        nodes = self.index.as_retriever(similarity_top_k=top_k, filters=MetadataFilters(filters=f)).retrieve(query)
        call = ToolCall("find_transactions", {"query": query, "category": category, "merchant": merchant,
                                              "date_from": date_from, "date_to": date_to, "min_amount": min_amount},
                        ["id", "txn_date", "merchant", "amount", "category"],
                        [(n.metadata["transaction_id"], date.fromisoformat(n.metadata["txn_date"]),
                          n.metadata["merchant"], n.metadata["amount"], n.metadata["category"]) for n in nodes])
        self.calls.append(call)
        return text2sql.result_text(call.columns[1:], [r[1:] for r in call.rows]) if call.rows else "No rows."

    def detect_recurring(self, category: str | None = None, price_increased_only: bool = False) -> str:
        """Recurring charges (subscriptions, bills): cadence, amount, price changes, active.
        Use price_increased_only=true for 'went up in price' questions."""
        with db.connect_owner(self.schema) if self.schema != db.SCHEMA else db.connect_reader() as conn:
            series = anomalies.detect_recurring(conn, category=category)
        if price_increased_only:
            series = [s for s in series if s.increased]
        return self._record("detect_recurring", {"category": category}, *anomalies.recurring_rows(series))

    def find_duplicates(self, days: int = 0) -> str:
        """Possible duplicate charges: same merchant and amount within `days` days (0 = same day)."""
        with db.connect_owner(self.schema) if self.schema != db.SCHEMA else db.connect_reader() as conn:
            cols, rows = anomalies.duplicate_rows(anomalies.find_duplicates(conn, days=days), conn)
        return self._record("find_duplicates", {"days": days}, cols, rows)

    def _record(self, name, args, cols, rows) -> str:
        self.calls.append(ToolCall(name, args, cols, rows))
        return text2sql.result_text(cols, rows) if rows else "No rows."

    # --- agent -------------------------------------------------------------------------------
    def agent(self, first: date, last: date, today: date) -> FunctionAgent:
        return FunctionAgent(
            tools=[FunctionTool.from_defaults(fn=f) for f in (
                self.query_database, self.find_transactions, self.detect_recurring, self.find_duplicates)],
            llm=Settings.llm,
            system_prompt=(
                f"You answer questions about the user's credit card transactions ({first} to {last}). "
                f"Today is {today}. Categories: {CATEGORY_GUIDE}\n"
                "Never calculate yourself: use query_database for totals, counts, averages, maxima and "
                "rankings; find_transactions to list transactions; detect_recurring for subscriptions and "
                "price increases; find_duplicates for duplicate charges. Quote numbers exactly. If the data "
                "can't answer (credit score, other cards, dates outside the range), say you don't have that "
                "data. Answer briefly."))

    def ask(self, question: str, today: date | None = None) -> AgentAnswer:
        t0 = time.time()
        today = today or date.today()
        self.calls = []
        if self.tuned:
            self.sql = self._sql_engine(today)  # conventions mention today's date
        with db.connect_owner(self.schema) if self.schema != db.SCHEMA else db.connect_reader() as conn:
            first, last = conn.execute("SELECT min(txn_date), max(txn_date) FROM transactions").fetchone()
        agent = self.agent(first, last, today)

        async def run():  # .run() must be called inside a running loop (it schedules a task)
            return await agent.run(user_msg=question)

        # One loop for the assistant's life: the Ollama LLM's async HTTP client binds to the first
        # loop it runs on, so asyncio.run() per question fails with "Event loop is closed".
        out = self.loop.run_until_complete(run())
        # .content, not str(): str(ChatMessage) prefixes the role ("assistant: ...")
        ans = AgentAnswer(question, question, text=(out.response.content or "").strip(),
                          calls=list(self.calls))
        sql_calls = [c for c in self.calls if c.name == "query_database"]
        ans.sql_first_try = (sql_calls[0].error is None) if sql_calls else None
        context = f"{question} {first} {last} {today}"
        per_call = [set(text2sql.unsupported_numbers(ans.text, context, c.columns, c.rows)) for c in self.calls] \
            or [set(text2sql.unsupported_numbers(ans.text, context, [], []))]
        ans.unsupported = sorted(set.intersection(*per_call)) if ans.text else []
        ans.seconds = time.time() - t0
        return ans
