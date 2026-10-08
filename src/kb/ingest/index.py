"""Ingestion steps 4-6: enrich chunks with manifest metadata, embed them, and index them in Qdrant.

Per document, based on its status in SQLite:
* chunked (or --force): full run. Embed every chunk (header + text), then replace the
  document's points in Qdrant. Point IDs are derived from chunk IDs, so re-runs never duplicate.
* indexed, manifest metadata unchanged: skipped.
* indexed, manifest metadata changed (allowed_groups, external_ok, release range, is_latest,
  category, title, ...): the payload is updated in place on all its points, without re-embedding.
* parsed / failed: skipped until `kb chunk` has run.
Documents removed from the manifest can be pruned (points and database rows deleted).
"""

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from kb.ingest.chunk import CHUNKER_VERSION
from kb.ingest.manifest import Document
from kb.store.embed import EMBEDDING_ID, Embedder
from kb.store.vectorstore import DENSE, SPARSE, doc_filter

# Fixed namespace so the same chunk_id always maps to the same Qdrant point ID.
POINT_NAMESPACE = uuid.UUID("6b3f2d7e-5a41-4c8e-9f0a-2d6c1e8b7a90")
UPSERT_BATCH = 64


@dataclass
class IndexResult:
    """Outcome of indexing one document, for the `kb index` report."""
    doc_id: str
    action: str                 # indexed | metadata | skipped | not_chunked
    chunks: int = 0
    embed_seconds: float = 0.0
    upsert_seconds: float = 0.0
    detail: str = ""


def point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point id for a chunk, so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def doc_metadata(doc: Document) -> dict:
    """Manifest metadata copied onto every point of the document (used as retrieval filters)."""
    return {
        "title": doc.title, "doc_type": doc.doc_type, "family": doc.family, "version": doc.version,
        "release_min": doc.release_min, "release_max": doc.release_max, "release_label": doc.release_label,
        "is_latest": doc.is_latest, "allowed_groups": list(doc.allowed_groups),
        "external_ok": doc.external_ok, "category": doc.category,
    }


def metadata_hash(meta: dict) -> str:
    """Short hash of the manifest metadata; a change means the payload needs updating."""
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]


def index_version(file_hash: str) -> str:
    """Version tag stored on every point: chunker version, embedding model and file hash."""
    return f"c{CHUNKER_VERSION}.{EMBEDDING_ID}.{file_hash[:12]}"


def chunk_payload(chunk: sqlite3.Row, meta: dict, version: str) -> dict:
    """Qdrant payload for one chunk: its text, location and the document's manifest metadata."""
    return {
        "doc_id": chunk["doc_id"], "chunk_id": chunk["chunk_id"], "section_id": chunk["section_id"],
        "section_number": chunk["section_id"].split("#", 1)[1], "chunk_index": chunk["chunk_index"],
        "header": chunk["header"], "text": chunk["text"], "content_type": chunk["content_type"],
        "page_start": chunk["page_start"], "page_end": chunk["page_end"], "token_count": chunk["token_count"],
        "index_version": version, **meta,
    }


def _update_document(conn: sqlite3.Connection, doc: Document, meta_hash: str, **fields) -> None:
    """Refresh manifest columns plus the given indexing fields on the documents row."""
    values = {
        "title": doc.title, "release_version": doc.release_label, "revision": doc.version,
        "is_latest": int(doc.is_latest), "category": doc.category,
        "allowed_groups": json.dumps(list(doc.allowed_groups)), "external_ok": int(doc.external_ok),
        "family": doc.family, "release_min": doc.release_min, "release_max": doc.release_max,
        "metadata_hash": meta_hash, **fields,
    }
    assignments = ", ".join(f"{k} = :{k}" for k in values)
    with conn:
        conn.execute(f"UPDATE documents SET {assignments}, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                     "WHERE doc_id = :doc_id", {**values, "doc_id": doc.doc_id})


def index_document(conn: sqlite3.Connection, client: QdrantClient, collection: str, doc: Document,
                   get_embedder: Callable[[], Embedder], *, force: bool = False) -> IndexResult:
    """Index one document; the embedder is only created (model loaded) if embedding is needed."""
    row = conn.execute("SELECT status, file_hash, metadata_hash FROM documents WHERE doc_id = ?",
                       (doc.doc_id,)).fetchone()
    if row is None or row["status"] not in ("chunked", "indexed"):
        status = row["status"] if row else "not parsed"
        return IndexResult(doc.doc_id, "not_chunked", detail=f"status {status}; run `kb parse` / `kb chunk`")

    meta = doc_metadata(doc)
    meta_hash = metadata_hash(meta)
    if row["status"] == "indexed" and not force:
        if row["metadata_hash"] == meta_hash:
            return IndexResult(doc.doc_id, "skipped", detail="up to date")
        client.set_payload(collection, payload=meta, points=doc_filter(doc.doc_id), wait=True)
        _update_document(conn, doc, meta_hash)
        return IndexResult(doc.doc_id, "metadata", detail="manifest metadata updated in place")

    chunks = conn.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY rowid", (doc.doc_id,)).fetchall()
    version = index_version(row["file_hash"])

    start = time.perf_counter()
    embeddings = get_embedder().embed([f"{c['header']}\n\n{c['text']}" for c in chunks]) if chunks else []
    embed_seconds = time.perf_counter() - start

    points = [
        models.PointStruct(
            id=point_id(c["chunk_id"]),
            vector={DENSE: e.dense, SPARSE: models.SparseVector(indices=e.sparse_indices, values=e.sparse_values)},
            payload=chunk_payload(c, meta, version),
        )
        for c, e in zip(chunks, embeddings, strict=True)
    ]
    start = time.perf_counter()
    client.delete(collection, points_selector=models.FilterSelector(filter=doc_filter(doc.doc_id)), wait=True)
    for i in range(0, len(points), UPSERT_BATCH):
        client.upsert(collection, points=points[i:i + UPSERT_BATCH], wait=True)
    stored = client.count(collection, count_filter=doc_filter(doc.doc_id), exact=True).count
    upsert_seconds = time.perf_counter() - start
    if stored != len(points):
        raise RuntimeError(f"{doc.doc_id}: {stored} points in Qdrant, expected {len(points)}")

    _update_document(conn, doc, meta_hash, status="indexed", index_version=version, error=None,
                     embed_seconds=round(embed_seconds, 2),
                     indexed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return IndexResult(doc.doc_id, "indexed", len(points), embed_seconds, upsert_seconds)


def prune_removed(conn: sqlite3.Connection, client: QdrantClient, collection: str,
                  manifest_ids: set[str]) -> list[str]:
    """Delete points and database rows of documents that are no longer in the manifest."""
    removed = [r["doc_id"] for r in conn.execute("SELECT doc_id FROM documents")
               if r["doc_id"] not in manifest_ids]
    for doc_id in removed:
        client.delete(collection, points_selector=models.FilterSelector(filter=doc_filter(doc_id)), wait=True)
        with conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))  # cascades to sections/chunks
    return removed
