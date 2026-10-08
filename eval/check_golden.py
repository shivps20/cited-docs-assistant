"""Consistency checks for the golden set (eval/golden.json) against the source documents.

    uv run python eval/check_golden.py

Errors (exit code 1):
  - structure: unknown type, unanswerable question with sources (or answerable without)
  - a cited doc_id is not in the manifest, or a cited page is outside the document
  - a must_include string is missing from its own expected answer
  - an article number in an answer does not exist anywhere in the corpus

Warnings (review by hand):
  - an article number or URL on a cited page is not in the answer. The check is per page, so it
    also flags references that belong to a neighbouring section on the same page.
"""

import json
import re
import sys
from functools import cache
from pathlib import Path

import pypdfium2 as pdfium

from kb.core.config import get_settings
from kb.core.domain import get_domain
from kb.ingest.manifest import load_manifest

TYPES = {"lookup", "howto", "compare", "unanswerable"}
URL = re.compile(r"https?://\S+")


@cache
def page_texts(path: Path) -> list[str]:
    """Text of every page of a PDF (cached), with soft-hyphen markers turned into '-'."""
    doc = pdfium.PdfDocument(path)
    # U+FFFE appears where PDFs soft-hyphenate words; normalise it so URLs compare cleanly.
    return [doc[i].get_textpage().get_text_range().replace("￾", "-") for i in range(len(doc))]


def main() -> int:
    """Check every golden question and print errors and warnings; returns 1 if there are errors."""
    sys.stdout.reconfigure(encoding="utf-8")
    settings = get_settings()
    docs = {d.doc_id: d for d in load_manifest(settings.manifest_path, settings.docs_dir)}
    pdfs = {doc_id: d.path for doc_id, d in docs.items() if d.doc_type == "pdf"}
    corpus = " ".join(t for path in pdfs.values() for t in page_texts(path))
    questions = json.loads(settings.golden_path.read_text(encoding="utf-8"))

    errors, warnings = [], []
    for q in questions:
        qid, answer = q["id"], q["expected_answer"]
        if q["type"] not in TYPES:
            errors.append(f"{qid}: unknown type {q['type']!r}")
        if (q["type"] == "unanswerable") == bool(q["sources"]):
            errors.append(f"{qid}: unanswerable questions must have no sources, others must have some")
        for s in q["must_include"]:
            if s.lower() not in answer.lower():
                errors.append(f"{qid}: must_include {s!r} not in expected_answer")
        for ref in sorted(set(get_domain().find_references(answer))):
            if ref not in corpus:
                errors.append(f"{qid}: {ref} does not appear in any document")

        for src in q["sources"]:
            doc_id = src["doc_id"]
            if doc_id not in docs:
                errors.append(f"{qid}: doc_id {doc_id!r} not in manifest")
                continue
            if doc_id not in pdfs:
                continue  # page checks only for PDFs for now
            pages = page_texts(pdfs[doc_id])
            first, last = src["pages"][0], src["pages"][-1]
            if not 1 <= first <= last <= len(pages):
                errors.append(f"{qid}: pages {first}-{last} outside {doc_id} (1-{len(pages)})")
                continue
            for page in range(first, last + 1):
                text = pages[page - 1]
                for ref in sorted(set(get_domain().find_references(text)) - set(get_domain().find_references(answer))):
                    warnings.append(f"{qid}: {doc_id} p{page} mentions {ref}, not in answer")
                for url in sorted(set(URL.findall(text))):
                    if url.rstrip(".,)") not in answer:
                        warnings.append(f"{qid}: {doc_id} p{page} URL not in answer: {url}")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(f"\n{len(questions)} questions: {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
