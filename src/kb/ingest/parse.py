"""Ingestion step 1: parse source documents with Docling and cache the result as JSON.

Parsing is the slowest stage (seconds per page on the GPU), so each result is cached under
data/parsed/ keyed by file hash and parser version. Later stages (structure, chunking) load
the cache and can be re-run freely; a document is only re-parsed when its file changes,
PARSER_VERSION is bumped, or --force is given.
"""

import hashlib
import json
import logging
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    ImageRefMode,
)

from kb.core.config import get_settings
from kb.ingest.manifest import Document

# Bump when converter options change, so cached parses made with the old options are redone.
PARSER_VERSION = 1

HEADING_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}

log = logging.getLogger(__name__)


@dataclass
class ParseStats:
    """What Docling produced for one document, and how long it took (stored in `documents`)."""
    doc_id: str
    pages: int
    seconds: float
    cached: bool
    text_items: int
    headings: int
    tables: int
    pictures: int
    empty_pages: int     # pages with no body text or table (e.g. screenshot-only pages)
    furniture: int       # page headers/footers Docling separated from the body

    @property
    def seconds_per_page(self) -> float:
        """Parse time per page (0 for a document without pages)."""
        return self.seconds / self.pages if self.pages else 0.0


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MB blocks; detects changed source files."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_path(doc_id: str, file_hash: str) -> Path:
    """Where the parsed JSON for this document version is cached."""
    return get_settings().parsed_dir / f"{doc_id}.{file_hash[:12]}.v{PARSER_VERSION}.json"


def find_cached(doc_id: str) -> Path | None:
    """The cached parse for doc_id, if any (used by later ingestion stages)."""
    matches = sorted(get_settings().parsed_dir.glob(f"{doc_id}.*.v{PARSER_VERSION}.json"))
    return matches[-1] if matches else None


def build_converter():
    """Docling converter: layout + table structure on GPU when available, OCR off."""
    import torch
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        artifacts_path=str(get_settings().docling_artifacts_path),
        do_ocr=False,
        do_table_structure=True,
        generate_page_images=False,
        generate_picture_images=False,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU),
    )
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.PPTX, InputFormat.DOCX],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


def document_stats(dl: DoclingDocument) -> dict:
    """Counts of text items, headings, tables, pictures, empty pages and furniture in a parsed document."""
    labels: Counter = Counter()
    pages_with_body: set[int] = set()
    for item, _ in dl.iterate_items():
        labels[item.label] += 1
        if item.label != DocItemLabel.PICTURE:
            pages_with_body.update(p.page_no for p in getattr(item, "prov", []))
    furniture = sum(1 for _ in dl.iterate_items(included_content_layers={ContentLayer.FURNITURE}))
    return {
        "pages": len(dl.pages),
        "text_items": len(dl.texts),
        "headings": sum(labels[label] for label in HEADING_LABELS),
        "tables": len(dl.tables),
        "pictures": len(dl.pictures),
        "empty_pages": len(set(dl.pages) - pages_with_body),
        "furniture": furniture,
    }


def parse_document(doc: Document, converter, *, force: bool = False) -> tuple[DoclingDocument, ParseStats, str]:
    """Parse one document (or load its cache). Returns the Docling document, stats and file hash."""
    file_hash = file_sha256(doc.path)
    target = cache_path(doc.doc_id, file_hash)
    if target.exists() and not force:
        dl = DoclingDocument.load_from_json(target)
        return dl, ParseStats(doc.doc_id, seconds=0.0, cached=True, **document_stats(dl)), file_hash

    start = time.perf_counter()
    result = converter.convert(doc.path)
    seconds = time.perf_counter() - start
    dl = result.document

    target.parent.mkdir(parents=True, exist_ok=True)
    for stale in target.parent.glob(f"{doc.doc_id}.*.json"):
        stale.unlink()
    dl.save_as_json(target, image_mode=ImageRefMode.PLACEHOLDER)
    return dl, ParseStats(doc.doc_id, seconds=seconds, cached=False, **document_stats(dl)), file_hash


STAT_COLUMNS = ("text_items", "headings", "tables", "pictures", "empty_pages", "furniture")


def record_document(conn: sqlite3.Connection, doc: Document, *, status: str, file_hash: str | None = None,
                    stats: ParseStats | None = None, error: str | None = None) -> None:
    """Upsert the documents row from the manifest plus parse outcome and statistics.

    A cache hit on an already chunked/indexed document keeps its later status; parse_seconds
    and parsed_at only change on a real parse, not a cache hit.
    """
    rel_path = doc.path.relative_to(get_settings().docs_dir.resolve()).as_posix()
    real_parse = stats is not None and not stats.cached
    stat_values = {c: getattr(stats, c) if stats else None for c in STAT_COLUMNS}
    stat_insert = ", ".join(STAT_COLUMNS)
    stat_params = ", ".join(f":{c}" for c in STAT_COLUMNS)
    stat_update = ", ".join(f"{c} = COALESCE(excluded.{c}, documents.{c})" for c in STAT_COLUMNS)
    with conn:
        conn.execute(
            f"""
            INSERT INTO documents (doc_id, source_path, title, doc_type, file_hash, page_count, release_version,
                                   revision, is_latest, category, allowed_groups, external_ok, family,
                                   release_min, release_max, status, error, parse_seconds, {stat_insert},
                                   parsed_at, updated_at)
            VALUES (:doc_id, :source_path, :title, :doc_type, :file_hash, :page_count, :release_label,
                    :version, :is_latest, :category, :allowed_groups, :external_ok, :family,
                    :release_min, :release_max, :status, :error, :parse_seconds, {stat_params},
                    CASE WHEN :real_parse THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now') END,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            ON CONFLICT (doc_id) DO UPDATE SET
                source_path = excluded.source_path, title = excluded.title, doc_type = excluded.doc_type,
                page_count = COALESCE(excluded.page_count, documents.page_count),
                parse_seconds = COALESCE(excluded.parse_seconds, documents.parse_seconds),
                {stat_update},
                release_version = excluded.release_version, revision = excluded.revision,
                is_latest = excluded.is_latest, category = excluded.category,
                allowed_groups = excluded.allowed_groups, external_ok = excluded.external_ok,
                family = excluded.family, release_min = excluded.release_min, release_max = excluded.release_max,
                status = CASE WHEN documents.file_hash IS excluded.file_hash
                                   AND documents.status IN ('chunked', 'indexed')
                                   AND excluded.status = 'parsed'
                              THEN documents.status ELSE excluded.status END,
                file_hash = COALESCE(NULLIF(excluded.file_hash, ''), documents.file_hash),
                error = excluded.error,
                parsed_at = COALESCE(excluded.parsed_at, documents.parsed_at),
                updated_at = excluded.updated_at
            """,
            {
                "doc_id": doc.doc_id, "source_path": rel_path, "title": doc.title, "doc_type": doc.doc_type,
                # file_hash is NOT NULL and checked before the upsert; '' means "unknown, keep existing".
                "file_hash": file_hash or "", "page_count": stats.pages if stats else None,
                "release_label": doc.release_label, "version": doc.version, "is_latest": int(doc.is_latest),
                "category": doc.category, "allowed_groups": json.dumps(list(doc.allowed_groups)),
                "external_ok": int(doc.external_ok), "family": doc.family, "release_min": doc.release_min,
                "release_max": doc.release_max, "status": status, "error": error,
                "parse_seconds": round(stats.seconds, 2) if real_parse else None, "real_parse": int(real_parse),
                **stat_values,
            },
        )
