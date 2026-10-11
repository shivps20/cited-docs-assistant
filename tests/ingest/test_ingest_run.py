from types import SimpleNamespace

import pytest

from kb.cli import ingest as cli
from kb.core import db as core_db
from kb.core.db import connect, migrate
from kb.ingest import chunk, parse
from kb.ingest.failures import Failure, append_failures, latest_run
from kb.ingest.manifest import Document


def test_failure_log_keeps_runs_and_reads_the_latest(tmp_path):
    """Runs append to one file; the latest run's failures are read back, and a clean run counts as latest."""
    log = tmp_path / "logs" / "ingest_failures.csv"
    assert latest_run(log) == (None, [])
    append_failures(log, "20261010-100000", [Failure("parse", "doc-a", "ValueError: bad"),
                                             Failure("index", "doc-b", "timeout")])
    run, failures = latest_run(log)
    assert run == "20261010-100000"
    assert [(f.step, f.doc_id) for f in failures] == [("parse", "doc-a"), ("index", "doc-b")]

    append_failures(log, "20261010-110000", [])
    assert latest_run(log) == ("20261010-110000", [])
    assert log.read_text(encoding="utf-8").count("run,time,step,doc_id,error") == 1


@pytest.fixture
def parse_env(tmp_path, monkeypatch):
    """A database, a documents folder with one PDF and its manifest Document."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.pdf").write_bytes(b"%PDF-1.4 test")
    monkeypatch.setattr(parse, "get_settings",
                        lambda: SimpleNamespace(docs_dir=docs_dir, parsed_dir=tmp_path / "parsed"))
    db = tmp_path / "kb.db"
    migrate(connect(db, check_schema=False))
    conn = connect(db)
    doc = Document(doc_id="doc-a", path=(docs_dir / "a.pdf").resolve(), title="Doc A", family="doc-a",
                   version="1.0", release_min=2015, release_max=9999, allowed_groups=("all",),
                   external_ok=False, category="installation", doc_type="pdf", is_latest=True)
    yield conn, doc
    conn.close()


def test_parse_up_to_date_needs_same_file_and_cached_parse(parse_env):
    """Skipped only when the row records this file's hash and the parse cache holds it."""
    conn, doc = parse_env
    assert not parse.parse_up_to_date(conn, doc)                     # no row yet
    file_hash = parse.file_sha256(doc.path)
    parse.record_document(conn, doc, status="parsed", file_hash=file_hash)
    assert not parse.parse_up_to_date(conn, doc)                     # no cached parse
    target = parse.cache_path(doc.doc_id, file_hash)
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    assert parse.parse_up_to_date(conn, doc)
    doc.path.write_bytes(b"%PDF-1.4 changed")
    assert not parse.parse_up_to_date(conn, doc)                     # file changed
    parse.record_document(conn, doc, status="failed", error="boom")
    assert not parse.parse_up_to_date(conn, doc)                     # last parse failed


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    """`kb ingest` with three documents and the three steps replaced by recorders; returns (calls, log)."""
    docs = [SimpleNamespace(doc_id=i, path=tmp_path / f"{i}.pdf") for i in ("doc-a", "doc-b", "doc-c")]
    log = tmp_path / "ingest_failures.csv"
    db = tmp_path / "kb.db"
    migrate(connect(db, check_schema=False))
    monkeypatch.setattr(core_db, "connect", lambda: connect(db))
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(failure_log=log))
    monkeypatch.setattr(cli, "_manifest_docs",
                        lambda ids=None: [d for d in docs if ids is None or d.doc_id in ids])
    monkeypatch.setattr(parse, "parse_up_to_date", lambda conn, d: d.doc_id == "doc-c")
    monkeypatch.setattr(chunk, "chunks_up_to_date", lambda conn, d: d.doc_id == "doc-c")
    monkeypatch.setattr(cli, "_free_gpu", lambda: None)
    calls: dict[str, list[str]] = {}

    def step(name, fail):
        """A replacement step that records its documents and fails `fail`."""
        def run(conn, step_docs, *args, **kwargs):
            """Record the documents and return the failures."""
            calls[name] = [d.doc_id for d in step_docs]
            failures = [Failure(name, d.doc_id, "boom") for d in step_docs if d.doc_id == fail]
            return (failures, 0, 0, 0) if name == "chunk" else failures
        return run

    monkeypatch.setattr(cli, "_parse_step", step("parse", "doc-a"))
    monkeypatch.setattr(cli, "_chunk_step", step("chunk", None))
    monkeypatch.setattr(cli, "_index_step", step("index", None))
    return calls, log


def test_ingest_skips_up_to_date_and_drops_failed_documents_from_later_steps(run_env):
    """Only doc-a and doc-b are parsed; doc-a fails and is neither chunked nor indexed; the failure is logged."""
    calls, log = run_env
    assert cli.ingest_documents(None, None, retry_failed=False, dry_run=False) == 1
    assert calls["parse"] == ["doc-a", "doc-b"]
    assert calls["chunk"] == ["doc-b", "doc-c"]          # doc-c: the chunk step itself skips it if up to date
    assert calls["index"] == ["doc-b", "doc-c"]
    assert [(f.step, f.doc_id) for f in latest_run(log)[1]] == [("parse", "doc-a")]


def test_retry_failed_runs_only_the_latest_failures(run_env):
    """--retry-failed takes the documents of the latest run's failures, nothing else."""
    calls, log = run_env
    append_failures(log, "20261010-100000", [Failure("index", "doc-b", "timeout")])
    assert cli.ingest_documents(None, None, retry_failed=True, dry_run=False) == 0
    assert calls["parse"] == ["doc-b"]
    assert calls["index"] == ["doc-b"]
    assert latest_run(log)[1] == []


def test_dry_run_changes_nothing(run_env, capsys):
    """--dry-run prints the plan and runs no step."""
    calls, log = run_env
    assert cli.ingest_documents(None, None, retry_failed=False, dry_run=True) == 0
    assert calls == {}
    assert not log.exists()
    out = capsys.readouterr().out
    assert "3 documents: 2 to parse, 2 to chunk, 3 to embed" in out
    assert "doc-a" in out and "parse + chunk + embed" in out
