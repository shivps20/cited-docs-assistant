"""Golden coverage check: is every golden fact present in some stored chunk on the cited pages?

Runs without any search. If a must_include string is not in a chunk of the cited document and
pages, no retrieval method can find it (lost table, screenshot-only page, bad split), so this
separates ingestion problems from search problems before retrieval is built.
"""

import json
import sqlite3
from dataclasses import dataclass, field

from kb.core.config import get_settings


@dataclass
class CoverageResult:
    """Coverage outcome for one golden question."""
    qid: str
    question: str
    missing: list[str] = field(default_factory=list)       # must_include strings not found
    no_chunks: list[str] = field(default_factory=list)     # cited sources with no chunk on those pages

    @property
    def passed(self) -> bool:
        """True when every must_include string was found and every cited page has chunks."""
        return not self.missing and not self.no_chunks


def check_coverage(conn: sqlite3.Connection, golden: list[dict] | None = None) -> list[CoverageResult]:
    """For each answerable golden question, look for its must_include strings in the stored
    chunks of the cited documents and pages (no search involved).
    """
    golden = golden if golden is not None else json.loads(get_settings().golden_path.read_text(encoding="utf-8"))
    results = []
    for q in golden:
        if not q["sources"]:
            continue  # unanswerable questions have nothing to cover
        result = CoverageResult(q["id"], q["question"])
        texts = []
        for src in q["sources"]:
            first, last = src["pages"][0], src["pages"][-1]
            rows = conn.execute(
                "SELECT text FROM chunks WHERE doc_id = ? AND page_start <= ? AND page_end >= ?",
                (src["doc_id"], last, first)).fetchall()
            if not rows:
                result.no_chunks.append(f"{src['doc_id']} p{first}-{last}")
            texts.extend(r["text"] for r in rows)
        combined = "\n".join(texts).lower()
        result.missing = [s for s in q["must_include"] if s.lower() not in combined]
        results.append(result)
    return results
