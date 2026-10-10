"""`kb manifest`, `kb parse`, `kb status`, `kb inspect`, `kb chunk`, `kb coverage`, `kb index`: the ingestion commands."""

import sys

from kb.core.config import get_settings
from kb.ingest.manifest import (
    ManifestError,
    append_rows,
    find_unlisted,
    load_manifest,
    scan_folder,
)


def manifest_validate() -> int:
    """`kb manifest validate`: load the manifest, print every problem, list unlisted files."""
    s = get_settings()
    print(f"documents folder: {s.docs_dir} (KB_DOCS_DIR)")
    try:
        docs = load_manifest(s.manifest_path, s.docs_dir)
    except ManifestError as e:
        print(e)
        return 1

    print(f"{'doc_id':<32} {'type':<5} {'ver':<5} {'latest':<6} {'release':<14} {'groups':<14} {'ext':<5} category")
    for d in docs:
        print(f"{d.doc_id:<32} {d.doc_type:<5} {d.version:<5} {'yes' if d.is_latest else 'no':<6} "
              f"{d.release_label:<14} {';'.join(d.allowed_groups):<14} {'yes' if d.external_ok else 'no':<5} {d.category}")

    unlisted = find_unlisted(s.manifest_path, s.docs_dir)
    for path in unlisted:
        print(f"WARN not in manifest: {path.relative_to(s.docs_dir)} (run `kb manifest scan`)")
    print(f"\nok: {len(docs)} documents, {len(unlisted)} unlisted")
    return 0


def manifest_scan(dry_run: bool = False) -> int:
    """`kb manifest scan`: draft manifest rows for files in the documents folder (and its subfolders) that are
    not in the manifest yet; files whose content is already listed are reported, not drafted."""
    from kb.ingest.parse import file_sha256

    sys.stdout.reconfigure(encoding="utf-8")
    s = get_settings()
    if not s.docs_dir.is_dir():
        print(f"documents folder not found: {s.docs_dir} (set KB_DOCS_DIR)")
        return 1
    print(f"documents folder: {s.docs_dir} (KB_DOCS_DIR)")
    report = scan_folder(s.manifest_path, s.docs_dir, file_sha256)
    print(f"supported files (PDF, PPTX, DOCX): {report.supported} · in the manifest: {report.listed} · "
          f"new: {len(report.rows)} · duplicates skipped: {len(report.duplicates)}")
    for rel, original in report.duplicates:
        print(f"SKIP {rel}: same content as {original}")
    if not report.rows:
        print("no new files to draft")
        return 0
    print("\nnew files by folder:")
    for folder, n in report.by_folder().items():
        print(f"  {n:4d}  {'(documents folder itself)' if folder == '.' else folder + '/'}")
    print()
    for row in report.rows:
        print(f"{'would add' if dry_run else '+'} {row['doc_id']}  {row['path']}  release={row['release_min'] or '?'}")
    if dry_run:
        print(f"\ndry run: {len(report.rows)} draft rows not written; run without --dry-run to append them")
        return 0
    append_rows(s.manifest_path, report.rows)
    print(f"\nappended {len(report.rows)} draft rows to {s.manifest_path.name}.")
    print("REVIEW BEFORE kb index: drafted rows give access to everyone (allowed_groups=all) and keep "
          "external_ok=false; set allowed_groups, external_ok, category, title, version and the release range, "
          "then run `kb manifest validate`.")
    return 0


def _manifest_docs(doc_ids: list[str] | None = None):
    """Validated manifest documents, optionally filtered; None (after printing why) on error."""
    s = get_settings()
    try:
        docs = load_manifest(s.manifest_path, s.docs_dir)
    except ManifestError as e:
        print(e)
        return None
    if doc_ids:
        unknown = set(doc_ids) - {d.doc_id for d in docs}
        if unknown:
            print(f"unknown doc_id(s): {', '.join(sorted(unknown))}")
            return None
        docs = [d for d in docs if d.doc_id in doc_ids]
    return docs


def _structure_for(doc):
    """Rebuilt section structure for a parsed document, or None if it has not been parsed."""
    from kb.ingest.parse import find_cached
    from kb.ingest.structure import document_structure, load_parsed

    cached = find_cached(doc.doc_id)
    if cached is None:
        return None
    return document_structure(load_parsed(cached), doc.doc_type)


def parse_documents(doc_ids: list[str] | None, force: bool) -> int:
    """`kb parse`: parse new or changed documents with Docling (cached) and record their statistics."""
    import logging

    from kb.core.db import connect
    from kb.ingest.parse import build_converter, parse_document, record_document

    logging.getLogger("docling").setLevel(logging.WARNING)
    s = get_settings()
    try:
        docs = load_manifest(s.manifest_path, s.docs_dir)
    except ManifestError as e:
        print(e)
        return 1
    if doc_ids:
        unknown = set(doc_ids) - {d.doc_id for d in docs}
        if unknown:
            print(f"unknown doc_id(s): {', '.join(sorted(unknown))}")
            return 1
        docs = [d for d in docs if d.doc_id in doc_ids]

    conn = connect()
    converter = build_converter()
    header = f"{'doc_id':<44} {'type':<5} {'pages':>5} {'sec':>7} {'s/pg':>5} {'texts':>6} {'heads':>5} " \
             f"{'tables':>6} {'pics':>5} {'empty':>5} {'furn':>5}"
    print(header, flush=True)
    failed = stalled = 0
    for doc in docs:
        try:
            _, st, file_hash = parse_document(doc, converter, force=force)
        except Exception as e:  # noqa: BLE001 - record the failure and continue with the next document
            failed += 1
            record_document(conn, doc, status="failed", error=f"{type(e).__name__}: {e}")
            print(f"{doc.doc_id:<44} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        record_document(conn, doc, status="parsed", file_hash=file_hash, stats=st)
        seconds = "cached" if st.cached else f"{st.seconds:.1f}"
        print(f"{doc.doc_id:<44} {doc.doc_type:<5} {st.pages:>5} {seconds:>7} {st.seconds_per_page:>5.2f} "
              f"{st.text_items:>6} {st.headings:>5} {st.tables:>6} {st.pictures:>5} {st.empty_pages:>5} "
              f"{st.furniture:>5}", flush=True)
        if st.stalled:
            stalled += 1
            print(f"  WARN {st.seconds:.0f} s for {st.pages} pages but only {st.cpu_seconds:.0f} s of CPU: the machine "
                  f"slept or was throttled (keep it plugged in, lid open); re-parse with --doc {doc.doc_id} --force "
                  f"to record the real time", flush=True)
    conn.close()
    print(f"\n{len(docs) - failed} parsed, {failed} failed; cache: {s.parsed_dir}")
    if stalled:
        print(f"WARN {stalled} document(s) stalled: their parse times include time the machine was asleep or throttled")
    return 1 if failed else 0


def show_status() -> int:
    """Per-document status and parse statistics from the database, reconciled with the manifest."""
    from kb.core.db import connect

    s = get_settings()
    try:
        manifest_ids = {d.doc_id for d in load_manifest(s.manifest_path, s.docs_dir)}
    except ManifestError as e:
        print(f"WARN manifest invalid ({len(e.errors)} errors); showing database only")
        manifest_ids = None

    conn = connect()
    rows = {r["doc_id"]: r for r in conn.execute("SELECT * FROM documents ORDER BY doc_id")}
    conn.close()

    print(f"{'doc_id':<44} {'type':<5} {'status':<8} {'pages':>5} {'sec':>6} {'s/pg':>5} {'texts':>6} "
          f"{'heads':>5} {'tables':>6} {'pics':>5} {'empty':>5} {'furn':>5} {'chunks':>6}  parsed_at")

    def num(value, fmt="") -> str:
        """Format a nullable number for the table ('-' when missing)."""
        return "-" if value is None else format(value, fmt)

    for doc_id in sorted(set(rows) | (manifest_ids or set())):
        r = rows.get(doc_id)
        if r is None:
            print(f"{doc_id:<44} {'':<5} {'pending':<8}  (in manifest, not yet parsed)")
            continue
        per_page = r["parse_seconds"] / r["page_count"] if r["parse_seconds"] and r["page_count"] else None
        print(f"{doc_id:<44} {r['doc_type'] or '':<5} {r['status']:<8} {num(r['page_count']):>5} "
              f"{num(r['parse_seconds'], '.1f'):>6} {num(per_page, '.2f'):>5} {num(r['text_items']):>6} "
              f"{num(r['headings']):>5} {num(r['tables']):>6} {num(r['pictures']):>5} {num(r['empty_pages']):>5} "
              f"{num(r['furniture']):>5} {num(r['chunk_count']):>6}  {(r['parsed_at'] or '-')[:19]}")
        if r["error"]:
            print(f"{'':<44} error: {r['error']}")
        if manifest_ids is not None and doc_id not in manifest_ids:
            print(f"{'':<44} WARN no longer in manifest")

    by_status: dict[str, int] = {}
    for r in rows.values():
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    pending = len((manifest_ids or set()) - set(rows))
    if pending:
        by_status["pending"] = pending
    pages = sum(r["page_count"] or 0 for r in rows.values())
    seconds = sum(r["parse_seconds"] or 0 for r in rows.values())
    summary = ", ".join(f"{n} {status}" for status, n in sorted(by_status.items()))
    print(f"\n{len(rows) + pending} documents: {summary}; {pages} pages parsed in {seconds:.0f} s")
    return 0


def inspect_document(doc_id: str, section_number: str | None, details: bool) -> int:
    """`kb inspect`: show a document's section tree, one section's content, or structure diagnostics."""
    from kb.ingest.parse import find_cached
    from kb.ingest.structure import document_structure, load_parsed

    sys.stdout.reconfigure(encoding="utf-8")
    s = get_settings()
    docs = {d.doc_id: d for d in load_manifest(s.manifest_path, s.docs_dir)}
    if doc_id not in docs:
        print(f"unknown doc_id {doc_id!r}; known: {', '.join(sorted(docs))}")
        return 1
    cached = find_cached(doc_id)
    if cached is None:
        print(f"{doc_id} is not parsed yet; run `uv run kb parse --doc {doc_id}`")
        return 1
    doc = docs[doc_id]
    dl = load_parsed(cached)
    st = document_structure(dl, doc.doc_type)

    if section_number:
        section = st.get(section_number)
        if section is None:
            print(f"no section {section_number!r} in {doc_id}")
            return 1
        path = " > ".join(f"{p.number} {p.title}" for p in st.path(section))
        print(f"{doc.title}\n{path}\npages {section.page_start}-{section.page_end}, "
              f"{len(section.blocks)} blocks, {section.chars} chars\n")
        for block in section.blocks:
            pages = f"p{block.page_start}" + (f"-{block.page_end}" if block.page_end != block.page_start else "")
            print(f"[{block.kind} {pages}]\n{block.text}\n")
        return 0

    unit = "slides" if doc.doc_type == "pptx" else "pages"
    print(f"{doc.title} ({doc.doc_type}, {len(dl.pages)} {unit})")
    if st.toc:
        print(f"TOC: {len(st.toc)} entries, {len(st.toc) - len(st.unmatched_toc)} matched to sections, "
              f"{len(st.unmatched_toc)} unmatched")
    print(f"Headings demoted to text: {len(st.demoted_headings)}; "
          f"boilerplate lines removed: {sum(st.removed_lines.values())}\n")
    print(f"{'section':<14} {'pages':>9} {'blocks':>6} {'~tokens':>7}  title")
    for sec in st.sections:
        pages = f"{sec.page_start}-{sec.page_end}"
        number = "  " * (sec.level - 1) + sec.number
        print(f"{number:<14} {pages:>9} {len(sec.blocks):>6} {sec.chars // 4:>7}  {sec.title[:70]}")
    print(f"\n{len(st.sections)} sections")

    if details:
        print("\nUnmatched TOC entries (no section found in the body):")
        for t in st.unmatched_toc:
            print(f"  {t.number or '-':<8} {t.title}")
        print("\nHeadings demoted to text:")
        for text, n in sorted({h: st.demoted_headings.count(h) for h in st.demoted_headings}.items(),
                              key=lambda kv: -kv[1]):
            print(f"  {n:>3}x {text[:90]}")
        print("\nBoilerplate lines removed:")
        for text, n in sorted(st.removed_lines.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>3}x {text}")
    return 0


def inspect_chunks(doc_id: str, section_number: str | None) -> int:
    """`kb inspect --chunks`: show the chunks of a document (or one section) as they will be embedded."""
    from kb.ingest.chunk import bge_m3_token_counter, chunk_document, chunk_section

    sys.stdout.reconfigure(encoding="utf-8")
    docs = _manifest_docs([doc_id])
    if docs is None:
        return 1
    doc = docs[0]
    structure = _structure_for(doc)
    if structure is None:
        print(f"{doc_id} is not parsed yet; run `uv run kb parse --doc {doc_id}`")
        return 1
    count = bge_m3_token_counter()

    if section_number:
        section = structure.get(section_number)
        if section is None:
            print(f"no section {section_number!r} in {doc_id}")
            return 1
        for c in chunk_section(doc, structure, section, count):
            print(f"===== {c.chunk_id}  [{c.content_type}, p{c.page_start}-{c.page_end}, {c.token_count} tokens]")
            print(f"{c.embed_text}\n")
        return 0

    chunks = chunk_document(doc, structure, count)
    print(f"{doc.title}: {len(structure.sections)} sections, {len(chunks)} chunks\n")
    print(f"{'chunk_id':<44} {'type':<6} {'pages':>7} {'tokens':>6}  start of text")
    for c in chunks:
        chunk_label = c.chunk_id.split("#", 1)[1]
        start = c.text.replace("\n", " ")[:60]
        print(f"{chunk_label:<44} {c.content_type:<6} {c.page_start:>3}-{c.page_end:<3} {c.token_count:>6}  {start}")
    return 0


def chunk_documents(doc_ids: list[str] | None) -> int:
    """`kb chunk`: rebuild sections and chunks from the cached parses and store them in SQLite."""
    from kb.core.db import connect
    from kb.ingest.chunk import bge_m3_token_counter, chunk_document, store_document

    docs = _manifest_docs(doc_ids)
    if docs is None:
        return 1
    count = bge_m3_token_counter()
    conn = connect()
    print(f"{'doc_id':<44} {'sections':>8} {'chunks':>6} {'avg tok':>7} {'max tok':>7} "
          f"{'tables':>6} {'code':>5} {'mixed':>5}")
    failed = total = 0
    for doc in docs:
        structure = _structure_for(doc)
        if structure is None:
            print(f"{doc.doc_id:<44} SKIPPED not parsed; run `uv run kb parse --doc {doc.doc_id}`")
            failed += 1
            continue
        try:
            chunks = chunk_document(doc, structure, count)
            store_document(conn, doc, structure, chunks, count)
        except Exception as e:  # noqa: BLE001 - report and continue with the next document
            print(f"{doc.doc_id:<44} FAILED {type(e).__name__}: {e}")
            failed += 1
            continue
        total += len(chunks)
        tokens = [c.token_count for c in chunks] or [0]
        kinds = [c.content_type for c in chunks]
        print(f"{doc.doc_id:<44} {len(structure.sections):>8} {len(chunks):>6} {sum(tokens) // len(tokens):>7} "
              f"{max(tokens):>7} {kinds.count('table'):>6} {kinds.count('code'):>5} {kinds.count('mixed'):>5}")
    conn.close()
    print(f"\n{len(docs) - failed} documents chunked, {failed} failed/skipped; {total} chunks stored")
    return 1 if failed else 0


def show_coverage(show_all: bool) -> int:
    """`kb coverage`: report golden questions whose facts are missing from the stored chunks."""
    from kb.core.db import connect
    from kb.evaluation.coverage import check_coverage

    sys.stdout.reconfigure(encoding="utf-8")
    conn = connect()
    if conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0:
        print("no chunks stored yet; run `uv run kb chunk` first")
        return 1
    results = check_coverage(conn)
    conn.close()
    for r in results:
        if r.passed and not show_all:
            continue
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.qid}  {r.question[:80]}")
        if r.no_chunks:
            print(f"        no chunks on cited pages: {', '.join(r.no_chunks)}")
        if r.missing:
            print(f"        missing: {', '.join(repr(m) for m in r.missing)}")
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} answerable golden questions fully covered by stored chunks")
    return 0 if passed == len(results) else 1


def _warn_if_ollama_holds_gpu() -> None:
    """The 6 GB GPU cannot hold bge-m3 and a loaded LLM at once; warn before embedding."""
    import httpx

    try:
        loaded = httpx.get(f"{get_settings().ollama_host}/api/ps", timeout=2).json().get("models", [])
    except httpx.HTTPError:
        return
    for m in loaded:
        print(f"WARN Ollama has '{m['name']}' loaded on the GPU; free it with `ollama stop {m['name']}`")


def index_documents(doc_ids: list[str] | None, force: bool, prune: bool) -> int:
    """`kb index`: embed and upsert (re)chunked documents, update changed metadata, optionally prune."""
    from kb.core.db import connect
    from kb.ingest.index import index_document, prune_removed
    from kb.store.embed import BgeM3Embedder
    from kb.store.vectorstore import get_client

    s = get_settings()
    all_docs = _manifest_docs()
    docs = _manifest_docs(doc_ids)
    if docs is None or all_docs is None:
        return 1
    conn = connect()
    client = get_client()
    if not client.collection_exists(s.qdrant_collection):
        print(f"collection '{s.qdrant_collection}' missing; run `uv run python scripts/init_qdrant.py`")
        return 1

    embedder = None

    def get_embedder():
        """Load bge-m3 on first use only, so runs with nothing to embed stay fast."""
        nonlocal embedder
        if embedder is None:
            _warn_if_ollama_holds_gpu()
            print("loading bge-m3 ...", flush=True)
            embedder = BgeM3Embedder()
            print(f"bge-m3 loaded on {embedder.device}", flush=True)
        return embedder

    if prune:
        for doc_id in prune_removed(conn, client, s.qdrant_collection, {d.doc_id for d in all_docs}):
            print(f"pruned {doc_id} (no longer in manifest)")

    print(f"{'doc_id':<44} {'action':<11} {'chunks':>6} {'embed s':>7} {'chunk/s':>7} {'upsert s':>8}  detail")
    failed = 0
    for doc in docs:
        try:
            r = index_document(conn, client, s.qdrant_collection, doc, get_embedder, force=force)
        except Exception as e:  # noqa: BLE001 - record the failure and continue with the next document
            failed += 1
            with conn:
                conn.execute("UPDATE documents SET error = ? WHERE doc_id = ?", (f"{type(e).__name__}: {e}", doc.doc_id))
            print(f"{doc.doc_id:<44} FAILED      {type(e).__name__}: {e}", flush=True)
            continue
        rate = f"{r.chunks / r.embed_seconds:.1f}" if r.embed_seconds else "-"
        timing = f"{r.embed_seconds:>7.1f} {rate:>7} {r.upsert_seconds:>8.1f}" if r.action == "indexed" \
            else f"{'-':>7} {'-':>7} {'-':>8}"
        print(f"{doc.doc_id:<44} {r.action:<11} {r.chunks:>6} {timing}  {r.detail}", flush=True)

    total = client.count(s.qdrant_collection, exact=True).count
    conn.close()
    print(f"\n{len(docs) - failed} ok, {failed} failed; collection '{s.qdrant_collection}' holds {total} points")
    return 1 if failed else 0
