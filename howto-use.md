# How to use ragplay

Ask questions about your credit card statements, in a browser or in the terminal. Everything runs
on this Mac: the models run in Ollama, the data lives in PostgreSQL on your LAN, and nothing is
sent to an outside service.

- [Before you start](#before-you-start)
- [The web UI](#the-web-ui)
- [The command line](#the-command-line)
- [A monthly routine](#a-monthly-routine)
- [Troubleshooting](#troubleshooting)

---

## Before you start

You need these once:

| What | How to check |
|---|---|
| The Python environment, with the web extra | `source .venv/bin/activate`, then `uv pip install -e '.[web]'` (only once) |
| Ollama running, with both models | `ollama list` shows `qwen3:8b` and `mxbai-embed-large` |
| Database settings in `.env` (repo root, gitignored) | `DB_HOST`, `DB`, `DB_USER`, `DB_PASSWORD`, `DB_READER_USER`, `DB_READER_PASSWORD` |
| The database tables | `rag db migrate`, then `rag db check` |

Every command below assumes the environment is active:

```bash
cd ~/projects/coding-repos/rag-play
source .venv/bin/activate
```

> Answers take 10–60 seconds because the model runs locally. Keep the laptop lid open during
> long jobs (imports, evals): closing it puts the Mac to sleep and pauses the work.

---

## The web UI

```bash
rag web
```

Your browser opens **http://127.0.0.1:8000**. The UI is only reachable from this Mac and has no
login. Stop it with `Ctrl-C` in the terminal.

| Option | What it does |
|---|---|
| `--port 8001` | Use another port |
| `--no-open` | Don't open the browser |
| `--schema ragplay_sample` | Look around with synthetic data instead of yours (a yellow badge in the sidebar shows this) |
| `--dev` | Debug pages and auto-reload when the code changes |

The dot at the bottom of the sidebar shows whether the database and Ollama are reachable.

### Dashboard

The month at a glance. Pick a month in the top-right corner.

- **Tiles:** total spent, the change from last month, the top category and how many budgets are
  over. The change is left blank for a month that isn't complete yet.
- **Monthly spending chart:** click a bar to open that month.
- **By category:** click a category to see its transactions for the month.
- **Budgets:** progress bars with a projection when the month isn't over. "Manage →" opens Budgets.
- **Needs attention:** price increases, possible duplicate charges and unusually large purchases
  this month. Click one to open the matching Insights tab.

### Ask

Chat about your spending in plain English.

- Type a question and press **Enter** (**Shift+Enter** adds a new line), or click a starter question.
- Follow-ups work: ask "How much did I spend on groceries in March?", then "and in April?".
  When the question was reworded, the answer shows what it was understood as.
- **Agent** (default) picks the right tool: totals, transaction search, subscriptions, budgets.
  **SQL only** answers with a single database query and no follow-up context.
- Open **How I got this** under an answer to see the tools used, the SQL and the rows it came
  from. A ⚠ warning means a number in the answer isn't in any tool result; treat that answer
  with care.
- **＋ New chat** clears the conversation.

Good questions to try:
- How much did I spend on dining last month?
- What was my largest purchase in 2026?
- Show my Amazon purchases this year
- Which subscriptions went up in price?
- Were there any duplicate charges?
- Am I over my dining budget in October 2026?

### Spending

Search and browse transactions.

- **Search box:** finds by meaning ("coffee", "flights", "hardware store") and shows the top 50
  matches ranked by relevance.
- **Filters:** category, merchant, date range and min/max amount. Use them on their own for an
  exact list: newest first, 50 per page, with a total.
- Click a merchant or category in the table to see all of its transactions. **Clear** resets.

### Insights

| Tab | What it shows |
|---|---|
| **Subscriptions** | Recurring charges, their monthly cost (yearly ones spread over 12 months), price changes and the next expected date. Toggle **Include stopped** or **Price changed**. |
| **Duplicates** | The same merchant and amount on the same day, or within 1, 3 or 7 days. Wider windows mostly catch repeat purchases, like coffee two days running. |
| **Unusual charges** | Purchases more than 3× your usual amount for that category and in your top 2% overall. Optional date range. |
| **Monthly review** | Pick a month and click **Write review**. The numbers are computed in code and the model writes them up (about 20–30 s). Open **The facts behind this** to check them. |

### Budgets

Monthly limits for a category, or for your total spending ("Total spending").

- **Add a budget:** choose a category, enter an amount and click **Add budget**.
- **Change one:** edit the amount on its card and click **Save**. **Remove** asks first.
- Each card shows spent, % used and what's left (or how much over). If the data stops partway
  through a month, it also shows a straight-line projection to month end.
- Status: **on track**, **at risk** (90% or more used, actual or projected) or **over**.
- Pick another month at the top to see how past months did.

Budget changes save to the database, the same as `rag budget set`.

### Categories

See and fix which category each merchant is in.

- **Tiles:** spend, transactions and merchants per category. Click one to filter the table.
- **Merchant table:** change the dropdown and it saves straight away (✓ saved). Your choice is
  marked **You** and is never overridden when categorisation runs again.
- Filter by merchant name, or by who set the category: **AI**, **Rule** or **You**.
- **Review AI choices** (top right, with a count) walks through every merchant the model
  categorised, busiest first. It shows how the merchant appears on your statement. Click
  **Looks right**, **Skip for now**, or the correct category. Each answer locks the category
  in as yours.

### Import

Load new statements.

1. Drag statement PDFs onto the dashed box, or click **choose files**. You can add several at once.
2. Watch the three steps: **Read & check statements**, **Categorise new merchants**, **Embed for search**.
3. Each file ends up as:
   - **✓ loaded:** the number of transactions;
   - **– already loaded:** the same file was imported before, so it was skipped;
   - **✕ not loaded:** the reason is shown, e.g. the totals didn't reconcile.

Each statement is checked against its own totals to the cent and loads only if everything
balances. Uploaded files are saved in `cc_statements/` (gitignored). If an import stops partway
(for example, Ollama wasn't running), fix the problem and upload again: work already done is
kept and the rest carries on. The table at the bottom lists every loaded statement; click a
period to see its transactions.

### History

Every question asked from the web, `rag ask`, `rag chat` or `rag li-ask`, newest first. Open
one to see how it was answered. Useful when an answer looks wrong.

---

## The command line

Run `rag --help`, or `rag <command> --help`, for the full options. Most commands take
`--schema ragplay_sample` to use synthetic data instead of yours.

### Load statements

```bash
rag ingest cc_statements/               # a folder or individual PDFs; already-loaded files are skipped
rag ingest new.pdf --dry-run            # parse and reconcile only, load nothing
rag categorize                          # categorise new merchants (rules first, then the model)
rag embed                               # make new transactions searchable by meaning
```

Run all three after adding statements, in that order (the web Import page does all three).
`rag categorize --redo-llm` re-asks the model about every merchant it categorised before. Your
own corrections and rule matches stay as they are.

### Ask questions

```bash
rag chat                                # a conversation; empty line or Ctrl-D quits
rag chat -v                             # also show tool calls and SQL
rag ask "How much did I spend on gas in 2026?"
rag ask "..." --sql                     # show the tool calls / SQL used
rag ask "..." --sql-only                # one SQL query only, no agent
```

### Search transactions

```bash
rag search "coffee"                                     # hybrid search, top 10
rag search "flights" --from 2026-01-01 --to 2026-06-30 -k 20
rag search "lunch" --category dining --min 20 --max 100
rag search "rogers" --mode text                         # hybrid | vector | text
```

### Subscriptions, duplicates, unusual charges, monthly review

```bash
rag recurring                           # active recurring charges
rag recurring --all                     # include ones that stopped
rag recurring --category subscriptions
rag duplicates                          # same merchant and amount, same day
rag duplicates --days 3
rag outliers --from 2026-01-01 --to 2026-12-31
rag review                              # last full month, written up by the model
rag review --month 2026-09 --facts      # also print the computed facts
```

### Budgets

```bash
rag budget set dining 400               # a category
rag budget set all 3000                 # total spending
rag budget list
rag budget status                       # current month (or the latest month with data)
rag budget status --month 2026-09
rag budget remove dining
```

### Categories

```bash
rag categories summary                  # transactions, merchants and spend per category
rag categories list --category other    # merchants in one category
rag categories list --source llm        # merchants the model categorised (rule | llm | manual)
rag categories set "Blue Bottle Coffee" coffee
rag categories set "Luigi's Trattoria" dining --description "family restaurant"
rag categories review                   # walk through model choices: Enter keeps, a name changes, q stops
```

Categories: groceries, dining, coffee, alcohol, gas, transport, travel, shopping, home, health,
personal_care, entertainment, subscriptions, utilities, insurance, education, government,
charity, fees, payment, other.

### Question history (trace log)

```bash
rag trace                               # the last 15 questions: route, tools, time, warnings
rag trace -n 50
rag trace --show 3                      # full detail of entry 3 from the list
```

Traces are kept in `data/traces/` (gitignored, one file per day).

### Database and quality checks

```bash
rag db check                            # connections, TLS, read-only role, schema
rag db migrate                          # apply new sql/ migrations
rag eval --set sample                   # the 18 synthetic questions (about 7 min)
rag eval --set real                     # your 32 questions (about 11 min); prints pass/fail only
rag eval --set real --ids t01,l03       # just some questions
rag eval --system sql                   # compare against the text-to-SQL-only baseline
```

Eval runs are saved in `eval/runs/` (gitignored). Rerun the evals after changing prompts or
tools to make sure nothing got worse.

### LlamaIndex comparison (optional)

```bash
rag li-index                            # build its index (schema ragplay_li)
rag li-ask "How much did I spend on groceries in March 2026?"
```

This is the Phase 9 framework rebuild, kept for comparison; `rag chat` is the better answerer.

---

## A monthly routine

1. **Import** the new statement (web Import page, or `rag ingest` → `rag categorize` → `rag embed`).
2. **Categories → Review AI choices** for any new merchants the model categorised.
3. **Dashboard:** check the month, and anything under **Needs attention**.
4. **Insights → Monthly review** for a write-up of the month.
5. **Budgets:** adjust if needed.

---

## Troubleshooting

| Symptom | What to do |
|---|---|
| Sidebar says **Ollama not ready** | Start Ollama, then check `ollama list` shows `qwen3:8b` and `mxbai-embed-large` |
| Sidebar says **Database unreachable** | Check you're on the home network, then run `rag db check` |
| `rag web` says Django is not installed | `source .venv/bin/activate`, then `uv pip install -e '.[web]'` |
| Port 8000 is in use | `rag web --port 8001` |
| An answer looks wrong | Open **How I got this** (or **History**, or `rag trace --show N`) and check the SQL and rows |
| Answer has a ⚠ "numbers aren't in any tool result" warning | The model added a number of its own; trust the rows under **How I got this** instead |
| A merchant is in the wrong category | Change it on **Categories** (or `rag categories set`); it updates every transaction and stays fixed |
| A statement shows **✕ not loaded** | The reason is listed under the file; the statement didn't reconcile, so nothing from it was loaded |
| An import stopped with an error | Fix the cause (usually Ollama) and upload the same files again; it resumes |
| Search misses recent transactions | Run `rag embed` (the web Import page does this automatically) |
