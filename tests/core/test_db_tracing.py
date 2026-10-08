import json
import sqlite3

import pytest

from kb.core.db import SCHEMA_VERSION, SchemaError, connect, migrate, schema_version
from kb.core.tracing import Tracer


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "kb.core.db"
    migrate(connect(path, check_schema=False))
    c = connect(path)
    yield c
    c.close()


def test_migrate_fresh_db(tmp_path):
    c = connect(tmp_path / "kb.core.db", check_schema=False)
    assert migrate(c) == list(range(1, SCHEMA_VERSION + 1))
    assert migrate(c) == []  # idempotent
    assert schema_version(c) == SCHEMA_VERSION
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    cols = {r["name"] for r in c.execute("PRAGMA table_info(documents)")}
    assert {"family", "release_min", "release_max"} <= cols


def test_connect_rejects_missing_or_stale_db(tmp_path):
    with pytest.raises(SchemaError, match="does not exist"):
        connect(tmp_path / "missing.db")
    stale = tmp_path / "stale.db"
    sqlite3.connect(stale).close()  # schema v0
    with pytest.raises(SchemaError, match="code expects"):
        connect(stale)


def test_connect_enables_foreign_keys(conn):
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO trace_stages (trace_id, seq, stage) VALUES ('nope', 1, 'x')")


def test_trace_records_stages_and_fields(conn):
    with Tracer(conn, "default launcher port?", user_id="u1") as t:
        with t.stage("search", top_k=30) as s:
            s["hits"] = ["c1", "c2"]
        with t.stage("rerank"):
            pass
        t.set(route="single", top_rerank_score=0.91, sources=[{"doc_id": "install-guide", "pages": [4]}],
              cache_hit=False)

    row = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (t.trace_id,)).fetchone()
    assert (row["query"], row["user_id"], row["route"], row["cache_hit"], row["error"]) == \
        ("default launcher port?", "u1", "single", 0, None)
    assert json.loads(row["sources"])[0]["doc_id"] == "install-guide"
    assert row["total_ms"] >= 0

    stages = conn.execute("SELECT * FROM trace_stages WHERE trace_id = ? ORDER BY seq", (t.trace_id,)).fetchall()
    assert [(s["seq"], s["stage"]) for s in stages] == [(1, "search"), (2, "rerank")]
    assert json.loads(stages[0]["data"]) == {"top_k": 30, "hits": ["c1", "c2"]}


def test_trace_records_failure_and_reraises(conn):
    with pytest.raises(RuntimeError), Tracer(conn, "q") as t, t.stage("embed"):
        raise RuntimeError("model not loaded")
    row = conn.execute("SELECT error FROM traces WHERE trace_id = ?", (t.trace_id,)).fetchone()
    assert row["error"] == "RuntimeError: model not loaded"
    data = conn.execute("SELECT data FROM trace_stages WHERE trace_id = ?", (t.trace_id,)).fetchone()["data"]
    assert json.loads(data)["error"] == "RuntimeError: model not loaded"


def test_trace_rejects_unknown_fields(conn):
    t = Tracer(conn, "q")
    with pytest.raises(ValueError, match="unknown trace field"):
        t.set(answr="typo")
