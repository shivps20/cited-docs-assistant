"""The knowledge-base tools used by multi-step answering: `search_kb`, `get_section`, `outline` and
`read_section`.

Access control is injected here, never passed by the caller (or by an LLM choosing the arguments):
a `KBTools` object is created per request with the user's groups, and every search and section
lookup is filtered by them. A section or document id alone does not give access: `get_section`,
`outline` and `read_section` check the document's allowed groups (and that it is the latest
edition) first.
"""

import json
import sqlite3
from dataclasses import dataclass, replace

from kb.core.tracing import Tracer
from kb.retrieve.assemble import ContextUnit, page_text, window_text
from kb.retrieve.pipeline import Retriever, SearchRequest, SearchResult
from kb.retrieve.search import PUBLIC_GROUP

OUTLINE_MAX_ENTRIES = 60  # sections listed per document outline (levels 1-3 first)
OUTLINE_MAX_LEVEL = 3
READ_MAX_TOKENS = 600     # a longer section is read as its first chunks up to about this size


@dataclass
class SectionText:
    """A whole section as `get_section` returns it."""

    section_id: str
    doc_id: str
    title: str
    heading_path: str
    page_start: int
    page_end: int
    release: str
    text: str
    doc_type: str = "pdf"       # pdf | pptx | docx: how the page reference reads

    @property
    def pages(self) -> str:
        """Page reference, e.g. 'p. 9', 'slides 4-6', or '' for a Word file (page_text)."""
        return page_text(self.page_start, self.page_end, self.doc_type)


@dataclass
class OutlineEntry:
    """One line of a document outline: a section's number, heading and size."""

    section_id: str
    number: str
    heading: str
    level: int
    tokens: int


class KBTools:
    """Search and section lookup for one request, always within the user's access groups."""

    def __init__(self, conn: sqlite3.Connection, retriever: Retriever, request: SearchRequest, trace: Tracer):
        """Keep the request's access (groups) and options; searches are recorded in `trace`."""
        self.conn = conn
        self.retriever = retriever
        self.request = request
        self.trace = trace
        self.groups = sorted({PUBLIC_GROUP, *request.groups})

    def search_kb(self, query: str, *, release: int | None = None, category: str | None = None,
                  side: str | None = None) -> SearchResult:
        """Hybrid search → rerank → context units for `query`, filtered by the user's groups.

        release / category narrow the search; the groups cannot be changed by the caller.
        side: label recorded in the trace stages (comparisons).
        """
        req = replace(self.request, query=query, groups=self.groups,
                      release=release if release is not None else self.request.release, category=category)
        return self.retriever.retrieve(req, self.trace, side=side)

    def get_section(self, section_id: str) -> SectionText | None:
        """The whole section, or None when it does not exist or its document is not visible to the user."""
        row = self._visible_section(section_id)
        if row is None:
            return None
        return SectionText(row["section_id"], row["doc_id"], row["title"], row["heading_path"], row["page_start"],
                           row["page_end"], row["release_version"] or "", row["text"], row["doc_type"] or "pdf")

    def outline(self, doc_id: str, *, max_entries: int = OUTLINE_MAX_ENTRIES) -> list[OutlineEntry]:
        """The document's sections in reading order (number, heading, size); [] when the document does
        not exist or is not visible to the user. Long outlines keep levels 1-3, then the first entries."""
        if not self._visible(doc_id):
            return []
        rows = self.conn.execute("SELECT section_id, heading_path, level, token_count FROM sections "
                                 "WHERE doc_id = ? ORDER BY ordinal", (doc_id,)).fetchall()
        entries = [OutlineEntry(r["section_id"], r["section_id"].split("#", 1)[1], r["heading_path"].split(" > ")[-1],
                                r["level"] or 1, r["token_count"] or 0) for r in rows]
        if len(entries) > max_entries:
            entries = [e for e in entries if e.level <= OUTLINE_MAX_LEVEL]
        return entries[:max_entries]

    def read_section(self, section_id: str, *, side: str = "",
                     max_tokens: int = READ_MAX_TOKENS) -> ContextUnit | None:
        """A section as a context unit (for the prompt and citations), or None when it is not visible.
        A section longer than max_tokens is read as its first chunks, up to about that size."""
        row = self._visible_section(section_id)
        if row is None:
            return None
        text, tokens, start, end, kind, window = row["text"], row["token_count"] or 0, row["page_start"], \
            row["page_end"], "section", None
        if tokens > max_tokens:
            text, tokens, start, end, last = window_text(self.conn, section_id, max_tokens)
            kind, window = "window", (0, last)
        number = section_id.split("#", 1)[1]
        return ContextUnit(doc_id=row["doc_id"], title=row["title"], section_id=section_id, section_number=number,
                           heading_path=row["heading_path"], header=row["title"], page_start=start, page_end=end,
                           text=text, kind=kind, score=0.0, tokens=tokens, window=window,
                           release=row["release_version"] or "", external_ok=bool(row["external_ok"]), side=side,
                           source_path=row["source_path"] or "", doc_type=row["doc_type"] or "pdf")

    def _visible(self, doc_id: str) -> bool:
        """Is the document the latest edition and in one of the user's groups?"""
        row = self.conn.execute("SELECT allowed_groups FROM documents WHERE doc_id = ? AND is_latest = 1",
                                (doc_id,)).fetchone()
        return row is not None and bool(set(json.loads(row["allowed_groups"] or "[]")) & set(self.groups))

    def _visible_section(self, section_id: str) -> sqlite3.Row | None:
        """The section row with its document's title, release and external_ok, if the user may see it."""
        row = self.conn.execute(
            "SELECT s.section_id, s.doc_id, s.heading_path, s.page_start, s.page_end, s.text, s.token_count, d.title, "
            "d.allowed_groups, d.release_version, d.external_ok, d.source_path, d.doc_type FROM sections s "
            "JOIN documents d ON d.doc_id = s.doc_id "
            "WHERE s.section_id = ? AND d.is_latest = 1", (section_id,)).fetchone()
        if row is None or not set(json.loads(row["allowed_groups"] or "[]")) & set(self.groups):
            return None
        return row
