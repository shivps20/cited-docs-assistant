import dataclasses
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from kb.core.db import connect, migrate
from kb.ingest import index as ix
from kb.ingest.manifest import Document
from kb.store.embed import Embedding
from kb.store.vectorstore import DENSE, DENSE_DIM, SPARSE, create_collection, doc_filter

COLLECTION = "test_chunks"


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [Embedding([float((len(t) + i) % 7 + 1)] * DENSE_DIM, [3, 17, 42], [0.5, 0.25, 0.1])
                for i, t in enumerate(texts)]


def make_doc(**overrides) -> Document:
    base = {"doc_id": "doc-a", "path": Path("/tmp/a.pdf"), "title": "Doc A", "family": "doc-a", "version": "1.0",
            "release_min": 2015, "release_max": 9999, "allowed_groups": ("all",), "external_ok": False,
            "category": "installation", "doc_type": "pdf", "is_latest": True}
    return Document(**(base | overrides))


def add_document(conn, doc_id: str, n_chunks: int, status: str = "chunked") -> None:
    with conn:
        conn.execute("INSERT INTO documents (doc_id, source_path, file_hash, status) VALUES (?, ?, 'abcdef1234567890', ?)",
                     (doc_id, f"{doc_id}.pdf", status))
        conn.execute("INSERT INTO sections (section_id, doc_id, ordinal, text, token_count) VALUES (?, ?, 0, 'x', 1)",
                     (f"{doc_id}#3.1", doc_id))
        for i in range(n_chunks):
            conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, section_id, chunk_index, header, text, content_type, "
                "page_start, page_end, token_count) VALUES (?, ?, ?, ?, ?, ?, 'text', 8, 9, 20)",
                (f"{doc_id}#3.1#{i}", doc_id, f"{doc_id}#3.1", i, "Doc A > 3.1 Prerequisites", f"chunk text {i}"))


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "kb.db"
    migrate(connect(db, check_schema=False))
    conn = connect(db)
    client = QdrantClient(":memory:")
    create_collection(client, COLLECTION)
    embedder = FakeEmbedder()
    yield conn, client, embedder
    conn.close()


def run(conn, client, embedder, doc, **kw):
    return ix.index_document(conn, client, COLLECTION, doc, lambda: embedder, **kw)


def points(client, doc_id="doc-a"):
    found, _ = client.scroll(COLLECTION, scroll_filter=doc_filter(doc_id), limit=100, with_vectors=True)
    return found


def test_point_id_is_deterministic_uuid():
    assert ix.point_id("doc-a#3.1#0") == ix.point_id("doc-a#3.1#0")
    assert ix.point_id("doc-a#3.1#0") != ix.point_id("doc-a#3.1#1")
    uuid.UUID(ix.point_id("doc-a#3.1#0"))


def test_full_index_writes_points_with_payload_and_vectors(env):
    conn, client, embedder = env
    add_document(conn, "doc-a", 3)
    result = run(conn, client, embedder, make_doc())
    assert (result.action, result.chunks) == ("indexed", 3)

    stored = points(client)
    assert len(stored) == 3
    p = next(p for p in stored if p.payload["chunk_id"] == "doc-a#3.1#1")
    assert p.id == ix.point_id("doc-a#3.1#1")
    assert p.payload["section_number"] == "3.1"
    assert p.payload["allowed_groups"] == ["all"]
    assert (p.payload["release_min"], p.payload["release_max"], p.payload["release_label"]) == (2015, 9999, "R2015x+")
    assert p.payload["text"] == "chunk text 1" and p.payload["header"] == "Doc A > 3.1 Prerequisites"
    assert p.payload["index_version"].startswith(f"c{ix.CHUNKER_VERSION}.")
    assert len(p.vector[DENSE]) == DENSE_DIM
    assert p.vector[SPARSE].indices == [3, 17, 42]

    row = conn.execute("SELECT status, metadata_hash, index_version, indexed_at FROM documents").fetchone()
    assert row["status"] == "indexed" and row["metadata_hash"] and row["indexed_at"]


def test_rerun_is_skipped_without_embedding(env):
    conn, client, embedder = env
    add_document(conn, "doc-a", 2)
    run(conn, client, embedder, make_doc())
    result = run(conn, client, embedder, make_doc())
    assert result.action == "skipped"
    assert embedder.calls == 1


def test_manifest_change_updates_payload_in_place(env):
    conn, client, embedder = env
    add_document(conn, "doc-a", 2)
    run(conn, client, embedder, make_doc())
    result = run(conn, client, embedder, make_doc(allowed_groups=("internal",), is_latest=False))
    assert result.action == "metadata"
    assert embedder.calls == 1
    assert all(p.payload["allowed_groups"] == ["internal"] and p.payload["is_latest"] is False
               for p in points(client))
    assert conn.execute("SELECT allowed_groups FROM documents").fetchone()[0] == '["internal"]'
    assert run(conn, client, embedder, make_doc(allowed_groups=("internal",), is_latest=False)).action == "skipped"


def test_rechunked_document_replaces_old_points(env):
    conn, client, embedder = env
    add_document(conn, "doc-a", 3)
    run(conn, client, embedder, make_doc())
    with conn:
        conn.execute("DELETE FROM chunks WHERE chunk_id = 'doc-a#3.1#2'")
        conn.execute("UPDATE documents SET status = 'chunked'")
    result = run(conn, client, embedder, make_doc())
    assert (result.action, result.chunks) == ("indexed", 2)
    assert sorted(p.payload["chunk_id"] for p in points(client)) == ["doc-a#3.1#0", "doc-a#3.1#1"]


def test_force_reembeds_and_unchunked_documents_are_skipped(env):
    conn, client, embedder = env
    add_document(conn, "doc-a", 1)
    add_document(conn, "doc-b", 1, status="parsed")
    run(conn, client, embedder, make_doc())
    assert run(conn, client, embedder, make_doc(), force=True).action == "indexed"
    assert embedder.calls == 2
    assert run(conn, client, embedder, make_doc(doc_id="doc-b")).action == "not_chunked"


def test_prune_removes_documents_missing_from_manifest(env):
    conn, client, embedder = env
    add_document(conn, "doc-a", 1)
    add_document(conn, "doc-b", 2)
    run(conn, client, embedder, make_doc())
    run(conn, client, embedder, dataclasses.replace(make_doc(), doc_id="doc-b"))
    assert ix.prune_removed(conn, client, COLLECTION, {"doc-a"}) == ["doc-b"]
    assert points(client, "doc-b") == []
    assert len(points(client, "doc-a")) == 1
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = 'doc-b'").fetchone()[0] == 0
