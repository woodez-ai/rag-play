# Phase 11 — Web UI (Django + HTMX)

## Goal
The CLI's functionality in a browser, simple to navigate, for one person on this Mac.

## Decisions
| Question | Choice | Why |
|---|---|---|
| Who / where | Only this Mac: binds 127.0.0.1, `ALLOWED_HOSTS` localhost only, no login | Real statements; nothing on the network should reach it |
| Scope | Everyday use + data management (ingest/categorise/embed). Eval, `db`, LlamaIndex stay CLI | Those are developer tools |
| Front end | Django templates + HTMX, hand-written CSS, no build step | One codebase; server-rendered pages; small live updates |
| Home | Dashboard with charts | Covers the "terminal charts" stretch item better |

## Architecture
- `src/ragplay_web/` is a thin layer: views call the same `ragplay` functions as the CLI
  (`agent.ask`, `text2sql.ask`, `budgets.status`, `anomalies.*`, `trace.*`). Dashboard queries live
  in `ragplay/stats.py` so the numbers are computed in SQL, never in templates or JS.
- **No Django database.** Pages use the psycopg connections: the read-only role on the real
  schema, the owner on synthetic schemas (same rule as the CLI). Sessions are files in `data/web/`.
- **Secret key**: `DJANGO_SECRET_KEY` if set (e.g. in `.env`), else generated once into
  `data/web/secret_key` (mode 600). It only signs session cookies on this machine.
- **CSP** (Django 6 built-in): scripts only from this site, so no inline scripts. htmx 2.0.4 and
  Chart.js 4.4.7 are vendored in `static/vendor/`: the app works offline and makes no third-party
  requests. Chart data is passed with `json_script` (not executed).
- CSRF on every POST (htmx sends the token as a header); `X-Frame-Options: DENY`.
- `rag web [--schema ragplay_sample] [--port] [--dev]` runs Django's `runserver` on 127.0.0.1,
  threaded. Static files are served by a URL route because runserver only does that in DEBUG.

## Pages (PR 1)
- **Dashboard**: month picker; tiles (spent, change vs previous month, top category, budgets
  over); trend chart (click a bar to open that month); category bars; budget bars with
  projection; "needs attention" (price changes, same-day duplicates, outliers in the month), loaded
  after the page because recurring detection scans every purchase. Partial months are flagged,
  and the "vs last month" change is not shown for them (it would compare half a month to a whole).
- **Ask**: chat with follow-ups (history kept in the session, last 30 turns); Agent or SQL-only
  mode; starter questions; each answer has "How I got this" (route, tool calls with args, SQL,
  first 10 rows, timings) built from the trace entry, so the UI and `rag trace` show the same thing.
- **History**: recent traced questions, filtered to the schema being viewed.

## Found while building
- History read every trace, so `rag web --schema ragplay_sample` would have listed real questions.
  It now filters by the trace's `schema`.
- The trend started 12 months back even when data started later (six empty bars); it now starts
  at the first month with data.

## Verification
- `tests/test_web.py` (9 in PR 1, 16 after PR 2): dashboard numbers equal ground truth, partial month flagged, chat keeps
  history and passes it to the agent, SQL-only mode, traces carry schema and session, History
  filters by schema, CSRF and Host enforcement, CSP header. 143 tests pass.
- End to end on `ragplay_sample` through the HTTP endpoint: "groceries in March 2026" → $603.84
  (matches the dashboard), "and in February?" rewritten correctly, budget question used
  `budget_status`. Screenshots checked in dark and light mode (synthetic data only).

## Pages (PR 2)
- **Spending**: hybrid search box ("coffee", "flights") plus exact filters (category, merchant,
  dates, min/max). Filtered lists are exact, newest first, paged 50 at a time, with a total; a
  text search shows the top 50 by relevance and no total (a ranked top-50 isn't a complete set).
  Merchant and category cells link to their own filtered lists.
- **Insights**, four tabs: Subscriptions (active / include stopped / price changed; active count,
  cost per month with yearly charges spread over 12, price increases), Duplicates (same day, 1,
  3 or 7 days), Unusual charges (date range), Monthly review (written by the model on request,
  with "the facts behind this").
- **Budgets**: a card per budget (spent, % used, left or over, projection), amount edited in
  place, remove with confirmation, add-a-budget card. Edits use the owner role; everything else
  stays read-only.
- Dashboard now links through: category bars → that category's transactions for the month,
  "Needs attention" items → the right Insights tab, budget card → Budgets.

## Found while building (PR 2)
- The local model writes Markdown (`**bold**`, `- ` bullets), which showed as raw symbols. The
  `md` filter escapes all HTML first, then renders only bold and bullet lists, so model output
  can't inject markup. Used for chat answers, history and the review.
- The review's faithfulness check flagged "$359.77 less than average": the model had subtracted
  the 3-month average from the total itself. Root-cause fix: `monthly_review` now includes
  `difference_from_average`, so there's nothing left to compute.

## Next PR
3. Categories (edit, review queue), Import (drag-and-drop PDFs with progress).
