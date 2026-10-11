"""Per-query tracing into the SQLite `traces` / `trace_stages` tables.

Every pipeline stage (condense, route, embed, search, rerank, gate, generate, ...)
is timed and its inputs/outputs recorded, so a bad answer can be traced back to
the stage that went wrong, and the evaluation can calibrate the confidence gate.

    with Tracer(conn, query="what is the default port?", user_id="u1") as trace:
        with trace.stage("search", top_k=30) as s:
            hits = search(...)
            s["hits"] = [h.id for h in hits]
        trace.set(route="single", answer=answer)

Stages and fields are buffered and written in one transaction when the trace
finishes. An exception inside the `with` block is recorded (stage and trace
`error`) and then re-raised, so failed queries are traced too.
"""

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Self

# Columns of `traces` that callers may set; anything else is rejected.
TRACE_FIELDS = {
    "standalone_query", "route", "release_filter", "top_rerank_score", "gate_decision",
    "llm_provider", "llm_model", "answer", "sources", "cache_hit", "index_version",
}
_JSON_FIELDS = {"sources"}


def _to_json(value: Any) -> str:
    """JSON text for a trace value (non-ASCII kept, unknown types via str())."""
    return json.dumps(value, ensure_ascii=False, default=str)


class Tracer:
    """Collects one query's stages and fields in memory and writes them to SQLite when finished."""

    def __init__(self, conn: sqlite3.Connection, query: str, *,
                 session_id: str | None = None, user_id: str | None = None):
        """Start a trace for `query`; the clock for total_ms starts now."""
        self.conn = conn
        self.trace_id = uuid.uuid4().hex
        self.query = query
        self.session_id = session_id
        self.user_id = user_id
        self.fields: dict[str, Any] = {}
        self.stages: list[dict[str, Any]] = []
        self.error: str | None = None
        self.finished = False
        self._start = time.perf_counter()

    def set(self, **fields: Any) -> None:
        """Set trace columns (only names in TRACE_FIELDS); later calls overwrite earlier values."""
        unknown = fields.keys() - TRACE_FIELDS
        if unknown:
            raise ValueError(f"unknown trace field(s): {', '.join(sorted(unknown))}")
        self.fields.update(fields)

    @contextmanager
    def stage(self, name: str, **data: Any) -> Iterator[dict[str, Any]]:
        """Time a pipeline stage; the yielded dict is stored as the stage's JSON data."""
        start = time.perf_counter()
        try:
            yield data
        except Exception as e:
            data["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            self.stages.append({"stage": name, "duration_ms": (time.perf_counter() - start) * 1000, "data": data})

    def add_stage(self, name: str, duration_ms: float, **data: Any) -> None:
        """Record a stage that was timed elsewhere (e.g. condensing, which runs before the trace starts)."""
        self.stages.append({"stage": name, "duration_ms": duration_ms, "data": data})

    def finish(self, error: str | None = None) -> None:
        """Write the trace and its stages. Called automatically when used as a context manager."""
        if self.finished:
            return
        self.finished = True
        self.error = error or self.error
        row = {
            "trace_id": self.trace_id, "session_id": self.session_id, "user_id": self.user_id,
            "query": self.query, "total_ms": (time.perf_counter() - self._start) * 1000, "error": self.error,
        }
        for key, value in self.fields.items():
            row[key] = _to_json(value) if key in _JSON_FIELDS else value
        if "cache_hit" in row:
            row["cache_hit"] = int(bool(row["cache_hit"]))

        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        with self.conn:  # one transaction for the trace and all its stages
            self.conn.execute(f"INSERT INTO traces ({columns}) VALUES ({placeholders})", row)
            self.conn.executemany(
                "INSERT INTO trace_stages (trace_id, seq, stage, duration_ms, data) VALUES (?, ?, ?, ?, ?)",
                [(self.trace_id, seq, s["stage"], s["duration_ms"], _to_json(s["data"]))
                 for seq, s in enumerate(self.stages, start=1)],
            )

    def __enter__(self) -> Self:
        """Use the tracer as a context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Write the trace, recording the exception if the block raised (which is then re-raised)."""
        self.finish(f"{exc_type.__name__}: {exc}" if exc_type else None)
        # returning None re-raises any exception


def append_stage(conn: sqlite3.Connection, trace_id: str, name: str, duration_ms: float, data: dict[str, Any]) -> None:
    """Add a stage to a trace that is already written (e.g. a faithfulness check run later, on demand),
    numbered after its last stage."""
    with conn:
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM trace_stages WHERE trace_id = ?",
                           (trace_id,)).fetchone()[0]
        conn.execute("INSERT INTO trace_stages (trace_id, seq, stage, duration_ms, data) VALUES (?, ?, ?, ?, ?)",
                     (trace_id, seq, name, duration_ms, _to_json(data)))
