from pathlib import Path

import pytest

from kb.answer.cache import CachedAnswer
from kb.answer.pipeline import _source
from kb.api.chat import context_payload
from kb.core.config import get_settings
from kb.core.db import connect, migrate
from kb.ingest.manifest import display_path
from kb.retrieve.assemble import ContextUnit, assemble
from kb.retrieve.search import Candidate


@pytest.fixture
def source_path_mode(monkeypatch):
    """Set KB_SOURCE_PATH (and a known KB_DOCS_DIR) for one test; settings are cached per process."""
    def set_mode(mode, docs_dir):
        monkeypatch.setenv("KB_SOURCE_PATH", mode)
        monkeypatch.setenv("KB_DOCS_DIR", str(docs_dir))
        get_settings.cache_clear()
    yield set_mode
    for name in ("KB_SOURCE_PATH", "KB_DOCS_DIR"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()


def unit(**overrides):
    """A context unit."""
    base = {"doc_id": "guide", "title": "Guide", "section_id": "guide#2.1", "section_number": "2.1",
            "heading_path": "2 X > 2.1 Y", "header": "Guide > 2.1", "page_start": 4, "page_end": 6,
            "text": "Port 20300.", "kind": "section", "score": 0.9, "tokens": 5}
    return ContextUnit(**(base | overrides))


def test_display_path_full_and_relative(tmp_path, source_path_mode):
    docs = tmp_path / "documents"
    outside = (tmp_path / "Shared" / "New" / "Deck.pptx").resolve()
    source_path_mode("full", docs)
    assert display_path("Install/Guide.pdf") == str((docs / "Install" / "Guide.pdf").resolve())
    assert display_path(outside.as_posix()) == str(outside)
    assert display_path("") == ""
    source_path_mode("relative", docs)
    assert display_path("Install/Guide.pdf") == "Install/Guide.pdf"
    assert display_path(outside.as_posix()) == "Deck.pptx"                 # no disk layout on a server


def test_assembled_units_and_sources_carry_the_manifest_path(tmp_path, source_path_mode):
    db = tmp_path / "kb.db"
    migrate(connect(db, check_schema=False))
    conn = connect(db)
    with conn:
        conn.execute("INSERT INTO documents (doc_id, source_path, file_hash) VALUES ('guide', 'Install/Guide.pdf', 'h')")
        conn.execute("INSERT INTO sections (section_id, doc_id, heading_path, ordinal, page_start, page_end, text, "
                     "token_count) VALUES ('guide#2.1', 'guide', '2 X > 2.1 Y', 1, 4, 6, 'Port 20300.', 5)")
    cand = Candidate(point_id="p", chunk_id="guide#2.1#0", doc_id="guide", section_id="guide#2.1", section_number="2.1",
                     chunk_index=0, title="Guide", header="Guide > 2.1", text="Port 20300.", page_start=4, page_end=6,
                     score=0.5, rerank_score=0.9)
    (u,) = assemble(conn, [cand])
    assert u.source_path == "Install/Guide.pdf"
    src = _source(1, u)
    assert src.path == "Install/Guide.pdf" and "Install" not in src.line          # the line itself is unchanged
    source_path_mode("relative", tmp_path / "documents")
    assert context_payload([u])["sources"][0]["path"] == "Install/Guide.pdf"
    conn.close()


def test_cached_answers_from_before_the_path_field_still_load():
    old = {"trace_id": "t", "text": "x", "context": [{k: v for k, v in unit().__dict__.items() if k != "source_path"}],
           "gate": {"decision": "pass", "top_score": 0.9, "threshold": 0.1},
           "generation": {"provider": "ollama", "model": "qwen"}, "sources": []}
    (u,) = CachedAnswer(old, "2026-10-10T00:00:00Z", 1).context()
    assert u.source_path == "" and display_path(u.source_path) == ""
    assert Path(unit(source_path="E:/x/y.pdf").source_path).name == "y.pdf"
