import json

import pytest
from qdrant_client import QdrantClient, models

from kb.core.db import connect, migrate
from kb.retrieve.assemble import assemble, near_duplicate
from kb.retrieve.pipeline import Retriever, SearchRequest
from kb.retrieve.rerank import rerank
from kb.retrieve.search import Candidate, build_filter, search
from kb.store.embed import Embedding
from kb.store.vectorstore import DENSE, DENSE_DIM, SPARSE, create_collection

COLLECTION = "test_chunks"


def unit(i: int) -> list[float]:
    v = [0.0] * DENSE_DIM
    v[i] = 1.0
    return v


def emb(i: int, token: int) -> Embedding:
    return Embedding(unit(i), [token], [1.0])


POINTS = [  # id, dense axis, sparse token, payload
    (1, 0, 10, {"doc_id": "launcher", "allowed_groups": ["all"], "release_min": 2015, "release_max": 9999, "is_latest": True}),
    (2, 1, 11, {"doc_id": "tracing", "allowed_groups": ["internal"], "release_min": 0, "release_max": 9999, "is_latest": True}),
    (3, 2, 12, {"doc_id": "mssql", "allowed_groups": ["all"], "release_min": 2026, "release_max": 2026, "is_latest": True}),
    (4, 3, 13, {"doc_id": "old-rev", "allowed_groups": ["all"], "release_min": 2015, "release_max": 9999, "is_latest": False}),
]


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    create_collection(c, COLLECTION)
    c.upsert(COLLECTION, points=[
        models.PointStruct(id=pid, vector={DENSE: unit(axis), SPARSE: models.SparseVector(indices=[tok], values=[1.0])},
                           payload={**p, "chunk_id": f"{p['doc_id']}#1#0", "section_id": f"{p['doc_id']}#1",
                                    "section_number": "1", "chunk_index": 0, "title": p["doc_id"],
                                    "header": f"{p['doc_id']} > 1 Intro", "text": "text", "page_start": 1, "page_end": 1})
        for pid, axis, tok, p in POINTS])
    return c


def docs(results) -> list[str]:
    return [c.doc_id for c in results]


def test_filter_applies_groups_latest_and_release(client):
    query = emb(0, 10)
    everyone = search(client, COLLECTION, query, build_filter(["all"]), mode="dense", limit=10)
    assert sorted(docs(everyone)) == ["launcher", "mssql"]                  # no internal, no old revision
    internal = search(client, COLLECTION, query, build_filter(["internal"]), mode="dense", limit=10)
    assert sorted(docs(internal)) == ["launcher", "mssql", "tracing"]
    r2025 = search(client, COLLECTION, query, build_filter(["all"], release=2025), mode="dense", limit=10)
    assert docs(r2025) == ["launcher"]                                      # R2015x+ matches, R2026x-only does not


def test_dense_sparse_and_hybrid_rank_the_matching_chunk_first(client):
    flt = build_filter(["all"])
    for mode, query in [("dense", emb(2, 99)), ("sparse", emb(5, 12)), ("hybrid", emb(2, 12))]:
        assert docs(search(client, COLLECTION, query, flt, mode=mode, limit=2))[0] == "mssql", mode


def test_unknown_mode_rejected(client):
    with pytest.raises(ValueError):
        search(client, COLLECTION, emb(0, 10), build_filter(["all"]), mode="bm25")


def cand(chunk_id, score=0.5, section=None, index=0):
    doc, num, _ = chunk_id.split("#")
    return Candidate(point_id=chunk_id, chunk_id=chunk_id, doc_id=doc, section_id=section or f"{doc}#{num}",
                     section_number=num, chunk_index=index, title=doc.upper(), header=f"{doc} > {num}",
                     text=f"text of {chunk_id}", page_start=1, page_end=1, score=score)


class ReverseReranker:
    def score(self, query, passages):
        return [float(i) for i in range(len(passages))]       # later passages score higher


def test_rerank_sorts_by_rerank_score():
    ranked = rerank(ReverseReranker(), "q", [cand("a#1#0"), cand("b#1#0"), cand("c#1#0")])
    assert [c.chunk_id for c in ranked] == ["c#1#0", "b#1#0", "a#1#0"]
    assert ranked[0].best_score == 2.0


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "kb.core.db"
    migrate(connect(db, check_schema=False))
    c = connect(db)
    with c:
        for doc in ("small", "big", "twin"):
            c.execute("INSERT INTO documents (doc_id, source_path, file_hash) VALUES (?, ?, 'h')", (doc, doc))
        c.execute("INSERT INTO sections (section_id, doc_id, heading_path, ordinal, page_start, page_end, text, "
                  "token_count) VALUES ('small#2', 'small', '2 Small', 0, 3, 4, 'whole small section', 120)")
        c.execute("INSERT INTO sections (section_id, doc_id, heading_path, ordinal, page_start, page_end, text, "
                  "token_count) VALUES ('twin#2', 'twin', '2 Small', 0, 3, 4, 'whole small section', 120)")
        c.execute("INSERT INTO sections (section_id, doc_id, heading_path, ordinal, page_start, page_end, text, "
                  "token_count) VALUES ('big#5', 'big', '5 Big', 0, 10, 20, 'long', 2000)")
        for i in range(6):
            c.execute("INSERT INTO chunks (chunk_id, doc_id, section_id, chunk_index, header, text, content_type, "
                      "page_start, page_end, token_count) VALUES (?, 'big', 'big#5', ?, 'h', ?, 'text', ?, ?, 250)",
                      (f"big#5#{i}", i, f"chunk {i}", 10 + i, 10 + i))
    yield c
    c.close()


def test_assemble_whole_small_section_and_window_for_large(conn):
    units = assemble(conn, [cand("small#2#0", 0.9), cand("big#5#3", 0.8, index=3)])
    small, big = units
    assert (small.kind, small.text, small.tokens, small.page_start, small.page_end) == \
        ("section", "whole small section", 120, 3, 4)
    assert (big.kind, big.window, big.text) == ("window", (2, 4), "chunk 2\n\nchunk 3\n\nchunk 4")
    assert (big.page_start, big.page_end, big.tokens) == (12, 14, 750)
    assert small.citation == "SMALL, Section 2, pp. 3-4"


def test_assemble_widens_window_and_skips_duplicates_and_respects_limits(conn):
    units = assemble(conn, [cand("big#5#1", index=1), cand("big#5#4", index=4), cand("small#2#0"),
                            cand("twin#2#0")])
    assert len(units) == 2                                     # twin has identical text to small -> skipped
    assert units[0].window == (0, 5) and units[0].chunk_ids == ["big#5#1", "big#5#4"]
    assert len(assemble(conn, [cand("small#2#0"), cand("big#5#1", index=1)], max_units=1)) == 1
    assert [u.section_id for u in assemble(conn, [cand("big#5#1", index=1), cand("small#2#0")],
                                           max_tokens=800)] == ["big#5"]


BASE = ("Unzip the media file to create the distribution directory, change to it and double-click setup.exe "
        "to start the installation; follow the snapshots to complete it. ") * 3


def test_near_duplicate_ignores_cosmetic_edits_only():
    assert near_duplicate(BASE, BASE)
    assert near_duplicate(BASE, "During the installation, " + BASE.replace("snapshots", "screenshots", 1))
    assert not near_duplicate(BASE, BASE[: len(BASE) // 2] + "Configure the Oracle listener on port 1521. " * 8)
    assert not near_duplicate("whole small section", "whole large section")


def test_assemble_keeps_one_of_two_near_identical_sections(conn):
    with conn:
        for sid, doc, text in [("ora#4", "small", BASE), ("sql#4", "twin", "During setup, " + BASE)]:
            conn.execute("INSERT INTO sections (section_id, doc_id, heading_path, ordinal, page_start, page_end, "
                         "text, token_count) VALUES (?, ?, '4 Install', 1, 5, 6, ?, 60)", (sid, doc, text))
    units = assemble(conn, [cand("ora#4#0", section="ora#4"), cand("sql#4#0", section="sql#4"), cand("small#2#0")])
    assert [u.section_id for u in units] == ["ora#4", "small#2"]     # the lower-ranked near copy is skipped
    assert [(s.doc_id, s.section_id, s.pages) for s in units[0].same_text] == [("sql", "sql#4", "pp. 5-6")]
    assert units[1].same_text == []
    # a copy ranked below the unit limit is still recorded on its twin
    capped = assemble(conn, [cand("ora#4#0", section="ora#4"), cand("small#2#0"), cand("sql#4#0", section="sql#4")],
                      max_units=2)
    assert [s.doc_id for s in capped[0].same_text] == ["sql"]


def test_assemble_min_score_drops_low_and_unreranked_chunks(conn):
    high, low, unscored = cand("small#2#0"), cand("big#5#1", index=1), cand("twin#2#0")
    high.rerank_score, low.rerank_score = 0.9, 0.1
    units = assemble(conn, [high, low, unscored], min_score=0.3)
    assert [u.section_id for u in units] == ["small#2"]


class FakeEmbedder:
    def embed(self, texts):
        return [emb(2, 12) for _ in texts]


class FixedReranker:
    def score(self, query, passages):
        return [0.9 if "mssql" in p else 0.1 for p in passages]


def test_retriever_runs_all_stages_and_traces(client, tmp_path):
    db = tmp_path / "trace.db"
    migrate(connect(db, check_schema=False))
    c = connect(db)
    retriever = Retriever(c, client, COLLECTION, FakeEmbedder(), FixedReranker())
    result = retriever.search(SearchRequest("which port?", release=2026))
    assert result.candidates[0].doc_id == "mssql"
    assert result.top_score == 0.9
    assert set(result.timings_ms) == {"embed", "search", "rerank", "assemble"}

    trace = c.execute("SELECT * FROM traces WHERE trace_id = ?", (result.trace_id,)).fetchone()
    assert (trace["route"], trace["release_filter"], trace["top_rerank_score"]) == ("retrieval", "R2026x", 0.9)
    stages = c.execute("SELECT stage, data FROM trace_stages WHERE trace_id = ? ORDER BY seq",
                       (result.trace_id,)).fetchall()
    assert [s["stage"] for s in stages] == ["embed", "search", "rerank", "assemble"]
    assert json.loads(stages[1]["data"])["groups"] == ["all"]
    c.close()
