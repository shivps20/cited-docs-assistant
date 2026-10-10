from pathlib import Path

import pytest

from kb.core.db import connect, migrate
from kb.evaluation.coverage import check_coverage
from kb.ingest import chunk as ch
from kb.ingest.manifest import Document
from kb.ingest.structure import Block, Element, build_structure


def words(text: str) -> int:
    return len(text.split())


def B(kind, text, page=1):
    return Block(kind, text, page, page)


def doc(doc_id="doc-a", doc_type="pdf", release=(2015, 9999)):
    return Document(doc_id=doc_id, path=Path(f"/tmp/{doc_id}.{doc_type}"), title="Doc A", family=doc_id,
                    version="1.0", release_min=release[0], release_max=release[1], allowed_groups=("all",),
                    external_ok=False, category="installation", doc_type=doc_type, is_latest=True)


def test_command_detection():
    assert ch.is_command("ACME> set index param depth 5;")
    assert ch.is_command("<ACME> validate index;")
    assert ch.is_command('<Valve className="org.apache.catalina.valves.RemoteIpValve"')
    assert ch.is_command("C:\\Apache24\\bin> httpd.exe -k install")
    assert ch.is_command("impdp system/system@ORCL REMAP_SCHEMA=app:app")
    assert not ch.is_command("Define the port for Apache Service.")
    assert not ch.is_command("Set the time zone to UTC for all the servers.")
    assert not ch.is_command("For example:")


def test_merge_commands_keeps_examples_together():
    merged = ch.merge_commands([
        B("text", "For example:"),
        B("text", "ACME> set index param depth 5;"),
        B("text", "ACME> set index param precision 100;", page=2),
        B("text", "where:", page=2),
    ])
    assert [b.kind for b in merged] == ["text", "code", "text"]
    assert merged[1].text == ("ACME> set index param depth 5;\n"
                              "ACME> set index param precision 100;")
    assert (merged[1].page_start, merged[1].page_end) == (1, 2)


def test_split_long_text_by_sentences(monkeypatch):
    monkeypatch.setattr(ch, "TARGET_TOKENS", 5)
    pieces = ch.split_block(B("text", "One two three. Four five six. Seven eight nine ten eleven twelve."), words)
    assert [p.text for p in pieces] == ["One two three.", "Four five six.", "Seven eight nine ten eleven", "twelve."]


def test_split_large_table_repeats_header(monkeypatch):
    monkeypatch.setattr(ch, "TARGET_TOKENS", 8)
    monkeypatch.setattr(ch, "MAX_ATOMIC_TOKENS", 8)
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"
    pieces = ch.split_block(B("table", table), words)
    assert len(pieces) == 3
    assert all(p.text.startswith("| a | b |\n| --- | --- |\n") for p in pieces)
    assert pieces[2].text.endswith("| 5 | 6 |")


def test_small_table_stays_whole_even_above_target(monkeypatch):
    monkeypatch.setattr(ch, "TARGET_TOKENS", 5)
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert ch.split_block(B("table", table), words) == [B("table", table)]


def test_pack_moves_lead_in_with_following_block(monkeypatch):
    monkeypatch.setattr(ch, "TARGET_TOKENS", 10)
    monkeypatch.setattr(ch, "MIN_TAIL_TOKENS", 3)
    eight = B("text", "one two three four five six seven eight")
    lead = B("text", "For example:")
    code = B("code", "ACME> a b c d e")
    groups = ch.pack_blocks([eight, lead, code], words)
    assert groups == [[eight], [lead, code]]


def test_pack_merges_tiny_tail(monkeypatch):
    monkeypatch.setattr(ch, "TARGET_TOKENS", 10)
    monkeypatch.setattr(ch, "MIN_TAIL_TOKENS", 4)
    nine, two = B("text", "a b c d e f g h i"), B("text", "j k")
    assert ch.pack_blocks([nine, two], words) == [[nine, two]]


def sample_structure():
    toc = [["1.", "Introduction ....", "4"], ["3.", "Installing ....", "8"], ["3.1.", "Prerequisites ....", "8"],
           ["", "On Cloud ....", "8"]]
    return build_structure([
        E("heading", "Executive Summary", 2), E("text", "Applies to all releases from R2015x and above.", 2),
        E("toc", "", 3, rows=toc),
        E("heading", "1. Introduction", 4), E("text", "The Launcher is a background service.", 4),
        E("heading", "3. Installing", 8),
        E("heading", "3.1. Prerequisites", 8),
        E("heading", "On Cloud", 8), E("list", "Run the cloud eligibility checker", 8, depth=2),
        E("text", "For example:", 9), E("text", "ACME> print index params all;", 9),
    ], doc_type="pdf", n_pages=10)


def E(kind, text="", page=1, **kw):
    return Element(kind, text, page, **kw)


def test_chunk_document_ids_headers_and_types():
    st = sample_structure()
    chunks = ch.chunk_document(doc(), st, words)
    assert [c.chunk_id for c in chunks] == ["doc-a#0#0", "doc-a#1#0", "doc-a#3.1.1#0"]  # 3 and 3.1 have no text
    assert chunks[0].header == "Doc A [R2015x+] > Executive Summary"
    on_cloud = chunks[2]
    assert on_cloud.header == "Doc A [R2015x+] > 3 Installing > 3.1 Prerequisites > 3.1.1 On Cloud"
    assert on_cloud.content_type == "mixed"
    assert on_cloud.text == ("- Run the cloud eligibility checker\n\nFor example:\n\n"
                             "ACME> print index params all;")
    assert (on_cloud.page_start, on_cloud.page_end) == (8, 9)
    assert on_cloud.embed_text.startswith(on_cloud.header + "\n\n")
    assert on_cloud.token_count == words(on_cloud.embed_text)


def test_pptx_header_uses_slide_range_and_no_release_for_any():
    st = build_structure([E("heading", "Configured product development", 16), E("text", "step one", 16),
                          E("heading", "Configured product development", 17), E("text", "step two", 17)],
                         doc_type="pptx", n_pages=20)
    chunks = ch.chunk_document(doc("deck", "pptx", release=(0, 9999)), st, words)
    assert chunks[0].header == "Doc A > Slides 16-17: Configured product development"


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "kb.db"
    migrate(connect(db, check_schema=False))
    c = connect(db)
    c.execute("INSERT INTO documents (doc_id, source_path, file_hash, status) VALUES ('doc-a', 'a.pdf', 'h', 'parsed')")
    c.commit()
    yield c
    c.close()


def test_store_document_replaces_sections_and_chunks(conn):
    st = sample_structure()
    chunks = ch.chunk_document(doc(), st, words)
    ch.store_document(conn, doc(), st, chunks, words)
    ch.store_document(conn, doc(), st, chunks, words)  # re-run must not duplicate

    sections = conn.execute("SELECT * FROM sections WHERE doc_id = 'doc-a' ORDER BY ordinal").fetchall()
    assert [s["section_id"] for s in sections] == ["doc-a#0", "doc-a#1", "doc-a#3", "doc-a#3.1", "doc-a#3.1.1"]
    on_cloud = sections[-1]
    assert (on_cloud["parent_section_id"], on_cloud["level"]) == ("doc-a#3.1", 3)
    assert on_cloud["heading_path"] == "3 Installing > 3.1 Prerequisites > 3.1.1 On Cloud"
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
    row = conn.execute("SELECT status, chunk_count, chunker_version FROM documents").fetchone()
    assert (row["status"], row["chunk_count"], row["chunker_version"]) == ("chunked", 3, ch.CHUNKER_VERSION)


def test_store_requires_parsed_document(conn):
    with pytest.raises(ValueError, match="run `kb parse` first"):
        ch.store_document(conn, doc("missing"), sample_structure(), [], words)


def test_coverage_reports_missing_facts_and_pages(conn):
    st = sample_structure()
    ch.store_document(conn, doc(), st, ch.chunk_document(doc(), st, words), words)
    results = check_coverage(conn, [
        {"id": "q1", "question": "eligibility?", "must_include": ["eligibility checker", "ACME>"],
         "sources": [{"doc_id": "doc-a", "pages": [8]}]},
        {"id": "q2", "question": "port?", "must_include": ["20300"], "sources": [{"doc_id": "doc-a", "pages": [8]}]},
        {"id": "q3", "question": "elsewhere?", "must_include": ["x"], "sources": [{"doc_id": "doc-a", "pages": [50]}]},
        {"id": "q4", "question": "unanswerable", "must_include": [], "sources": []},
    ])
    by_id = {r.qid: r for r in results}
    assert by_id["q1"].passed
    assert by_id["q2"].missing == ["20300"]
    assert by_id["q3"].no_chunks == ["doc-a p50-50"]
    assert "q4" not in by_id
