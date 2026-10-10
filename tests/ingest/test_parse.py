from pathlib import Path
from types import SimpleNamespace

import pytest

from kb.core.db import connect, migrate
from kb.ingest import parse
from kb.ingest.manifest import Document

PARSED = parse.ParseStats("doc-a", pages=3, seconds=4.5, cached=False, text_items=40, headings=5, tables=1,
                          pictures=2, empty_pages=0, furniture=6)
CACHED = parse.ParseStats("doc-a", pages=3, seconds=0.0, cached=True, text_items=40, headings=5, tables=1,
                          pictures=2, empty_pages=0, furniture=6)


@pytest.fixture
def env(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.pdf").write_bytes(b"%PDF-1.4 test")
    monkeypatch.setattr(parse, "get_settings",
                        lambda: SimpleNamespace(docs_dir=docs_dir, parsed_dir=tmp_path / "parsed"))
    db = tmp_path / "kb.core.db"
    migrate(connect(db, check_schema=False))
    conn = connect(db)
    doc = Document(doc_id="doc-a", path=(docs_dir / "a.pdf").resolve(), title="Doc A", family="doc-a",
                   version="1.0", release_min=2015, release_max=9999, allowed_groups=("all", "eng"),
                   external_ok=False, category="installation", doc_type="pdf", is_latest=True)
    yield conn, doc
    conn.close()


def status(conn) -> str:
    return conn.execute("SELECT status FROM documents WHERE doc_id = 'doc-a'").fetchone()["status"]


def test_record_document_inserts_manifest_fields(env):
    conn, doc = env
    parse.record_document(conn, doc, status="parsed", file_hash="h1", stats=PARSED)
    row = conn.execute("SELECT * FROM documents WHERE doc_id = 'doc-a'").fetchone()
    assert (row["source_path"], row["release_version"], row["release_min"], row["page_count"]) == \
        ("a.pdf", "R2015x+", 2015, 3)
    assert row["allowed_groups"] == '["all", "eng"]'
    assert row["parsed_at"] is not None


def test_cache_hit_keeps_later_status_but_file_change_resets(env):
    conn, doc = env
    parse.record_document(conn, doc, status="parsed", file_hash="h1", stats=PARSED)
    conn.execute("UPDATE documents SET status = 'indexed' WHERE doc_id = 'doc-a'")
    conn.commit()

    parse.record_document(conn, doc, status="parsed", file_hash="h1", stats=PARSED)  # same file
    assert status(conn) == "indexed"

    parse.record_document(conn, doc, status="parsed", file_hash="h2", stats=PARSED)  # file changed
    assert status(conn) == "parsed"


def test_stats_saved_and_cache_hit_keeps_parse_time(env):
    conn, doc = env
    parse.record_document(conn, doc, status="parsed", file_hash="h1", stats=PARSED)
    first = conn.execute("SELECT * FROM documents WHERE doc_id = 'doc-a'").fetchone()
    assert (first["parse_seconds"], first["headings"], first["tables"], first["furniture"]) == (4.5, 5, 1, 6)

    parse.record_document(conn, doc, status="parsed", file_hash="h1", stats=CACHED)
    again = conn.execute("SELECT parse_seconds, parsed_at FROM documents WHERE doc_id = 'doc-a'").fetchone()
    assert (again["parse_seconds"], again["parsed_at"]) == (4.5, first["parsed_at"])


def test_failure_is_recorded_without_losing_hash(env):
    conn, doc = env
    parse.record_document(conn, doc, status="parsed", file_hash="h1", stats=PARSED)
    parse.record_document(conn, doc, status="failed", error="ConversionError: boom")
    row = conn.execute("SELECT status, error, file_hash FROM documents WHERE doc_id = 'doc-a'").fetchone()
    assert (row["status"], row["error"], row["file_hash"]) == ("failed", "ConversionError: boom", "h1")


def test_cache_path_includes_hash_and_version(env):
    path = parse.cache_path("doc-a", "abcdef1234567890")
    assert path.name == f"doc-a.abcdef123456.v{parse.PARSER_VERSION}.json"
    assert isinstance(path, Path)


def test_only_the_bbox_warning_is_silenced():
    import re
    import warnings

    from kb.ingest.parse import BBOX_WARNING

    noise = "Provenance bbox coordinate l on page 23 is outside page bounds: value=-1441985.0 < lo=0.0; clamping to 0.0"
    assert re.match(BBOX_WARNING, noise)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=BBOX_WARNING, category=UserWarning)
        warnings.warn(noise, UserWarning, stacklevel=1)
        warnings.warn("something else", UserWarning, stacklevel=1)
    assert [str(w.message) for w in seen] == ["something else"]
