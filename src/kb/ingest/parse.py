"""Ingestion step 1: parse source documents with Docling and cache the result as JSON.

Parsing is the slowest stage (seconds per page on the GPU), so each result is cached under
data/parsed/ keyed by file hash and parser version. Later stages (structure, chunking) load
the cache and can be re-run freely; a document is only re-parsed when its file changes,
PARSER_VERSION is bumped, or --force is given.
"""

import hashlib
import json
import logging
import shutil
import sqlite3
import subprocess
import tempfile
import time
import warnings
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
from kb.ingest.manifest import LEGACY_FORMATS, Document, manifest_path_for

# Bump when converter options change, so cached parses made with the old options are redone.
PARSER_VERSION = 1

HEADING_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
# Docling warns once per page when an element's box lies outside the page (e.g. a slide-master shape placed
# off the slide) and clamps it; only page numbers are used here, so the warning is noise.
BBOX_WARNING = r"Provenance bbox coordinate .* is outside page bounds"
STALL_SECONDS = 120          # see ParseStats.stalled
STALL_CPU_SHARE = 0.1

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
    cpu_seconds: float = 0.0   # CPU time of the parse (all threads); far below `seconds` = the machine slept

    @property
    def seconds_per_page(self) -> float:
        """Parse time per page (0 for a document without pages)."""
        return self.seconds / self.pages if self.pages else 0.0

    @property
    def stalled(self) -> bool:
        """Took over 2 minutes while using under a tenth of that in CPU time: the machine was asleep or
        heavily throttled, not busy parsing (a normal parse uses 2-6 s of CPU per second)."""
        return not self.cached and self.seconds > STALL_SECONDS and self.cpu_seconds < STALL_CPU_SHARE * self.seconds


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


def parse_up_to_date(conn: sqlite3.Connection, doc: Document) -> bool:
    """Can `kb ingest` skip parsing this document? Yes when its row records a successful parse of the current
    file (same hash) and that parse is in the cache: then not even the cached JSON needs loading."""
    row = conn.execute("SELECT status, file_hash FROM documents WHERE doc_id = ?", (doc.doc_id,)).fetchone()
    if row is None or row["status"] not in ("parsed", "chunked", "indexed") or not row["file_hash"]:
        return False
    file_hash = file_sha256(doc.path)
    return row["file_hash"] == file_hash and cache_path(doc.doc_id, file_hash).exists()


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


def find_soffice() -> str | None:
    """LibreOffice's soffice: KB_SOFFICE, else on PATH, else the usual Windows install folders."""
    configured = get_settings().soffice
    if configured:
        return configured
    found = shutil.which("soffice") or shutil.which("soffice.exe")
    if found:
        return found
    for base in (Path(r"C:\Program Files\LibreOffice"), Path(r"C:\Program Files (x86)\LibreOffice")):
        candidate = base / "program" / "soffice.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def convert_legacy(doc: Document, file_hash: str) -> Path:
    """A .ppt / .doc converted to .pptx / .docx with LibreOffice (headless), cached by doc id and file hash
    under data/parsed/converted/; Docling then parses the converted copy. Raises RuntimeError when
    LibreOffice is missing or the conversion fails."""
    target_ext = LEGACY_FORMATS[doc.path.suffix.lower()]
    out_dir = get_settings().parsed_dir / "converted"
    target = out_dir / f"{doc.doc_id}.{file_hash[:12]}.{target_ext}"
    if target.exists():
        return target
    soffice = find_soffice()
    if soffice is None:
        raise RuntimeError(f"{doc.path.name}: .{doc.path.suffix.lower().lstrip('.')} needs LibreOffice to convert "
                           f"it (install it, set KB_SOFFICE, or save the file as .{target_ext})")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([soffice, "--headless", "--norestore", "--convert-to", target_ext, "--outdir", tmp, str(doc.path)],
                       check=True, capture_output=True, timeout=600)
        produced = Path(tmp) / f"{doc.path.stem}.{target_ext}"
        if not produced.is_file():
            raise RuntimeError(f"{doc.path.name}: LibreOffice did not produce a .{target_ext}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob(f"{doc.doc_id}.*.{target_ext}"):
            stale.unlink()
        shutil.move(str(produced), target)
    return target


def parse_document(doc: Document, converter, *, force: bool = False) -> tuple[DoclingDocument, ParseStats, str]:
    """Parse one document (or load its cache). Returns the Docling document, stats and file hash.
    A legacy .ppt / .doc is converted to .pptx / .docx first (convert_legacy); the hash is the original's."""
    file_hash = file_sha256(doc.path)
    target = cache_path(doc.doc_id, file_hash)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=BBOX_WARNING, category=UserWarning)
        if target.exists() and not force:
            dl = DoclingDocument.load_from_json(target)
            return dl, ParseStats(doc.doc_id, seconds=0.0, cached=True, **document_stats(dl)), file_hash

        start, cpu_start = time.perf_counter(), time.process_time()
        source = convert_legacy(doc, file_hash) if doc.path.suffix.lower() in LEGACY_FORMATS else doc.path
        result = converter.convert(source)
        seconds, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu_start
        dl = result.document

    target.parent.mkdir(parents=True, exist_ok=True)
    for stale in target.parent.glob(f"{doc.doc_id}.*.json"):
        stale.unlink()
    dl.save_as_json(target, image_mode=ImageRefMode.PLACEHOLDER)
    return dl, ParseStats(doc.doc_id, seconds=seconds, cached=False, cpu_seconds=cpu_seconds,
                          **document_stats(dl)), file_hash


STAT_COLUMNS = ("text_items", "headings", "tables", "pictures", "empty_pages", "furniture")


# ---------------------------------------------------------------------------- parsing in a worker process

PARSE_DOCS_PER_WORKER = 50        # a fresh worker (and fresh threads) after this many documents (kb.core.workers)
PARSE_TIMEOUT_S = 1800            # a document still parsing after 30 min is stopped and marked failed
_worker_converter = None          # the Docling converter of this worker process


def init_parse_worker() -> None:
    """Runs once in each parse worker: quiet Docling's logging and load the converter (models on the GPU)."""
    global _worker_converter
    logging.getLogger("docling").setLevel(logging.WARNING)
    _worker_converter = build_converter()


def parse_in_worker(job: tuple[Document, bool]) -> tuple[ParseStats, str]:
    """Parse one document in the worker (job = (doc, force)); the result goes to the parse cache, so only
    the statistics and the file hash travel back to the main process."""
    doc, force = job
    _, stats, file_hash = parse_document(doc, _worker_converter, force=force)
    return stats, file_hash


def record_document(conn: sqlite3.Connection, doc: Document, *, status: str, file_hash: str | None = None,
                    stats: ParseStats | None = None, error: str | None = None) -> None:
    """Upsert the documents row from the manifest plus parse outcome and statistics.

    A cache hit on an already chunked/indexed document keeps its later status; parse_seconds
    and parsed_at only change on a real parse, not a cache hit.
    """
    rel_path = manifest_path_for(doc.path, get_settings().docs_dir)     # relative, or absolute outside it
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
