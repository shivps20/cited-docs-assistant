"""The knowledge-base tools used by multi-step answering: `search_kb` and `get_section`.

Access control is injected here, never passed by the caller (or, later, by an LLM choosing the
arguments): a `KBTools` object is created per request with the user's groups, and every search and
section lookup is filtered by them. A section id alone does not give access: `get_section` checks
the document's allowed groups first.
"""

import json
import sqlite3
from dataclasses import dataclass, replace

from kb.core.tracing import Tracer
from kb.retrieve.assemble import ContextUnit, page_text
from kb.retrieve.pipeline import Retriever, SearchRequest, SearchResult
from kb.retrieve.search import PUBLIC_GROUP

SNIPPET_CHARS = 300      # text per search result in the compact listing


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

    @property
    def pages(self) -> str:
        """Page reference, e.g. 'p. 9' or 'pp. 9-10'."""
        return page_text(self.page_start, self.page_end)


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
        row = self.conn.execute(
            "SELECT s.section_id, s.doc_id, s.heading_path, s.page_start, s.page_end, s.text, d.title, "
            "d.allowed_groups, d.release_version FROM sections s JOIN documents d ON d.doc_id = s.doc_id "
            "WHERE s.section_id = ? AND d.is_latest = 1", (section_id,)).fetchone()
        if row is None or not set(json.loads(row["allowed_groups"] or "[]")) & set(self.groups):
            return None
        return SectionText(row["section_id"], row["doc_id"], row["title"], row["heading_path"], row["page_start"],
                           row["page_end"], row["release_version"] or "", row["text"])


def listing(units: list[ContextUnit]) -> list[dict]:
    """Compact view of context units (id, citation, score, start of the text) for logs and, later, an LLM."""
    return [{"section_id": u.section_id, "citation": u.citation, "score": round(u.score, 4),
             "snippet": u.text[:SNIPPET_CHARS]} for u in units]
