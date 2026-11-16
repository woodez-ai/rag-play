"""JSONL trace of each question: route, tool calls (args, SQL, rows, timing, errors), answer.

One file per day in data/traces/ (gitignored: traces contain real amounts). Used by
`rag ask`, `rag chat` and `rag li-ask`; `rag trace` reads them back. The eval keeps its own
run files, so eval questions are not traced here.
"""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "data/traces"
MAX_ROWS = 50  # rows kept per tool call (the full count is recorded too)


def _plain(v):
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return v


def _rows(rows: list[tuple]) -> list[list]:
    return [[_plain(v) for v in r] for r in rows[:MAX_ROWS]]


def record(ans, *, system: str, model: str, schema: str, session: str | None = None,
           path: Path | None = None) -> dict:
    """Append one trace line. `ans` is an agent.AgentAnswer or a text2sql.Answer."""
    if hasattr(ans, "calls"):  # agent / LlamaIndex
        steps = [{"tool": c.name, "args": {k: _plain(v) for k, v in c.arguments.items()},
                  "sql": c.sql or None, "error": c.error, "seconds": round(getattr(c, "seconds", 0.0), 2),
                  "columns": c.columns, "n_rows": len(c.rows), "rows": _rows(c.rows)} for c in ans.calls]
        standalone = ans.standalone
        route = ("router: sql" if any(c.arguments.get("routed") for c in ans.calls)
                 else "agent" if ans.calls else "agent: no tool (declined or answered directly)")
    else:  # text2sql.Answer
        steps = [{"tool": "text2sql", "sql": a.sql, "error": a.error, "empty": a.empty,
                  "columns": a.columns, "n_rows": len(a.rows), "rows": _rows(a.rows)} for a in ans.attempts]
        standalone, route = ans.question, "text2sql"
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"), "session": session, "system": system,
        "model": model, "schema": schema, "question": ans.question, "standalone": standalone,
        "route": route, "steps": steps, "answer": ans.text,
        "unsupported": list(ans.unsupported), "seconds": round(ans.seconds, 2),
    }
    path = path or TRACES / f"{date.today().isoformat()}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def recent(n: int = 20, directory: Path | None = None) -> list[dict]:
    """The last n traces across days, oldest first."""
    files = sorted((directory or TRACES).glob("*.jsonl"))
    out: list[dict] = []
    for f in reversed(files):
        lines = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        out = lines + out
        if len(out) >= n:
            break
    return out[-n:]
