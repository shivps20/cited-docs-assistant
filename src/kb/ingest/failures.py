"""The ingestion failure log: one CSV row per document that failed a step of `kb ingest`.

The file is append-only history (`data/logs/ingest_failures.csv` by default, `KB_FAILURE_LOG`): each run has an
id (its start time), so `kb ingest --retry-failed` re-runs the documents that failed in the latest run, and
`kb status` reports them. A document that later succeeds simply does not appear in the next run's rows.
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path

FAILURE_COLUMNS = ("run", "time", "step", "doc_id", "error")
STEPS = ("parse", "chunk", "index")


@dataclass(frozen=True)
class Failure:
    """One document that failed one ingestion step."""

    step: str           # parse | chunk | index
    doc_id: str
    error: str


def new_run_id() -> str:
    """An id for this run: its local start time, sortable (20261010-153000)."""
    return time.strftime("%Y%m%d-%H%M%S")


def append_failures(path: Path, run: str, failures: list[Failure]) -> None:
    """Append this run's failures to the log (header written when the file is new); nothing when there are none.
    A run without failures is recorded too, as one row without a document, so the latest run is always known."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = [{"run": run, "time": now, "step": f.step, "doc_id": f.doc_id, "error": f.error} for f in failures] \
        or [{"run": run, "time": now, "step": "", "doc_id": "", "error": ""}]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def latest_run(path: Path) -> tuple[str | None, list[Failure]]:
    """The latest run's id and its failures (empty when it had none); (None, []) when no run is logged."""
    if not path.exists():
        return None, []
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, []
    run = max(r["run"] for r in rows)
    return run, [Failure(r["step"], r["doc_id"], r["error"]) for r in rows if r["run"] == run and r["doc_id"]]
