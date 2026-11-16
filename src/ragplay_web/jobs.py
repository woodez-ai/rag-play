"""Import job: ingest uploaded PDFs, then categorise and embed, in a background thread.

One job at a time (categorise and embed work over the whole table). State lives in memory: a
restart forgets the progress display, but the work itself is in the database (each statement
loads in its own transaction, each merchant is cached as it's classified, embedding only does
what's missing), so running the import again picks up where it stopped.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ragplay import categorize, db, ingest, store


@dataclass
class FileResult:
    name: str
    status: str = "waiting"  # waiting | loading | loaded | skipped | failed
    transactions: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class Job:
    id: str
    schema: str
    files: list[FileResult]
    stage: str = "queued"     # queued | ingest | categorize | embed | done | error
    detail: str = ""
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: float | None = None

    @property
    def done(self) -> bool:
        return self.stage in ("done", "error")

    @property
    def loaded(self) -> int:
        return sum(f.status == "loaded" for f in self.files)

    @property
    def steps(self) -> list[tuple[str, str]]:
        """(label, state) for the progress list; state is ok | now | skip | todo | failed."""
        order = ["ingest", "categorize", "embed"]
        labels = ["Read & check statements", "Categorise new merchants", "Embed for search"]
        reached = order.index(self.stage) if self.stage in order else (3 if self.stage == "done" else -1)
        out = []
        for i, label in enumerate(labels):
            if self.stage == "done" and i > 0 and not self.loaded:
                state = "skip"  # nothing new loaded, so nothing to categorise or embed
            elif self.stage == "error" and i == self._failed_at:
                state = "failed"
            elif i < reached or (self.stage == "error" and i < self._failed_at):
                state = "ok"
            elif i == reached:
                state = "now"
            else:
                state = "todo"
            out.append((label, state))
        return out

    _failed_at: int = -1


_lock = threading.Lock()
_jobs: dict[str, Job] = {}
_current: str | None = None


def current() -> Job | None:
    with _lock:
        return _jobs.get(_current) if _current else None


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


RUN_ASYNC = True  # tests run the job inline


def start(schema: str, paths: list[Path]) -> Job:
    """Start an import. Raises RuntimeError if one is already running."""
    global _current
    with _lock:
        running = _jobs.get(_current) if _current else None
        if running and not running.done:
            raise RuntimeError("an import is already running")
        job = Job(uuid.uuid4().hex[:12], schema, [FileResult(p.name) for p in paths])
        _jobs[job.id] = job
        _current = job.id
    if RUN_ASYNC:
        threading.Thread(target=_run, args=(job, paths), daemon=True, name=f"import-{job.id}").start()
    else:
        _run(job, paths)
    return job


def _run(job: Job, paths: list[Path]) -> None:
    try:
        with db.connect_owner(job.schema) as conn:
            job.stage = "ingest"
            for fr, path in zip(job.files, paths):
                fr.status = "loading"
                r = ingest.ingest_file(conn, path)
                fr.status, fr.transactions, fr.errors = r.status, r.transactions, r.errors
            if job.loaded:
                job.stage = "categorize"
                st = categorize.categorize(conn, progress=lambda i, n: setattr(job, "detail", f"new merchant {i} of {n}"))
                job.detail = f"{st.rule + st.llm} new merchant{'s' if st.rule + st.llm != 1 else ''} categorised"
                job.stage = "embed"
                written, _ = store.embed_transactions(conn)
                job.detail += f"; {written} transaction{'s' if written != 1 else ''} embedded for search"
            job.stage = "done"
    except Exception as e:  # shown on the page; nothing secret in psycopg/ollama messages
        stages = ["ingest", "categorize", "embed"]
        job._failed_at = stages.index(job.stage) if job.stage in stages else 0
        job.stage, job.error = "error", f"{type(e).__name__}: {e}"
    finally:
        job.finished = time.time()
