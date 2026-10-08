"""Context assembly: turn ranked chunks into the units of context the LLM will see.

For each ranked chunk, best first:
* its section is small (<= SECTION_MAX_TOKENS): use the whole section (from SQLite);
* otherwise: the chunk plus its neighbours (+/-1) in the same section; further hits in that
  section widen the same window instead of adding a new unit.
Stops at MAX_UNITS units / MAX_TOKENS tokens. Units whose text is the same apart from small edits
(the MSSQL and Oracle guides share most sections, often with a word or two changed) are kept once:
the higher-ranked copy wins.
"""

import sqlite3
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from kb.retrieve.search import Candidate

MAX_UNITS = 6               # was 4: the unit cap, not the token budget, cut off overview sections (TD-17)
MAX_TOKENS = 3000
SECTION_MAX_TOKENS = 800
NEAR_DUPLICATE_RATIO = 0.95  # word-level similarity; MSSQL/Oracle copies with cosmetic edits are 0.96-0.99,
                             # sections that differ in content are 0.92 or lower


@dataclass
class ContextUnit:
    """One block of context for the LLM: a whole section or a window of chunks, with its citation data."""
    doc_id: str
    title: str
    section_id: str
    section_number: str
    heading_path: str
    header: str
    page_start: int
    page_end: int
    text: str
    kind: str                    # section | window
    score: float                 # best score of the chunks behind this unit
    tokens: int
    chunk_ids: list[str] = field(default_factory=list)
    window: tuple[int, int] | None = None   # chunk_index range for windows
    release: str = ""                       # release label from the manifest, e.g. 'R2015x+'
    external_ok: bool = False               # may this text be sent to an external LLM

    @property
    def pages(self) -> str:
        """Page reference, e.g. 'p. 9' or 'pp. 9-10'."""
        return f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"

    @property
    def heading(self) -> str:
        """The section's own heading, e.g. '2.2.3 Configure the SAML Metadata'."""
        return self.heading_path.split(" > ")[-1]

    @property
    def citation(self) -> str:
        """Short citation, e.g. 'SAML Guide, Section 2.2.3, pp. 9-10'."""
        return f"{self.title}, Section {self.section_number}, {self.pages}"


def _window(conn: sqlite3.Connection, section_id: str, lo: int, hi: int) -> tuple[str, int, int, int]:
    """Text, tokens and page range of the chunks lo..hi of a section, joined in order."""
    rows = conn.execute("SELECT text, token_count, page_start, page_end FROM chunks WHERE section_id = ? "
                        "AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index", (section_id, lo, hi)).fetchall()
    text = "\n\n".join(r["text"] for r in rows)
    return text, sum(r["token_count"] for r in rows), min(r["page_start"] for r in rows), max(r["page_end"] for r in rows)


def near_duplicate(a: str, b: str, ratio: float = NEAR_DUPLICATE_RATIO) -> bool:
    """Are two texts the same apart from small edits? Compared word by word, cheapest bounds first."""
    if a == b:
        return True
    wa, wb = a.split(), b.split()
    matcher = SequenceMatcher(None, wa, wb, autojunk=False)
    return matcher.real_quick_ratio() >= ratio and matcher.quick_ratio() >= ratio and matcher.ratio() >= ratio


def assemble(conn: sqlite3.Connection, ranked: list[Candidate], *, max_units: int = MAX_UNITS,
             max_tokens: int = MAX_TOKENS, section_max_tokens: int = SECTION_MAX_TOKENS,
             min_score: float | None = None) -> list[ContextUnit]:
    """min_score (reranker scale): skip chunks below it, and chunks that were not reranked."""
    units: list[ContextUnit] = []
    for c in ranked:
        if min_score is not None and (c.rerank_score is None or c.rerank_score < min_score):
            continue
        section = conn.execute("SELECT heading_path, text, token_count, page_start, page_end FROM sections "
                               "WHERE section_id = ?", (c.section_id,)).fetchone()
        if section is None:
            continue
        existing = next((u for u in units if u.section_id == c.section_id), None)
        used = sum(u.tokens for u in units)

        if existing is not None:
            if existing.kind == "window":            # widen the window to include this chunk's neighbours
                lo = min(existing.window[0], c.chunk_index - 1)
                hi = max(existing.window[1], c.chunk_index + 1)
                text, tokens, p_start, p_end = _window(conn, c.section_id, lo, hi)
                if used - existing.tokens + tokens <= max_tokens:
                    existing.text, existing.tokens, existing.window = text, tokens, (lo, hi)
                    existing.page_start, existing.page_end = p_start, p_end
                    existing.chunk_ids.append(c.chunk_id)
            elif c.chunk_id not in existing.chunk_ids:
                existing.chunk_ids.append(c.chunk_id)
            continue

        if len(units) >= max_units:
            continue
        if section["token_count"] <= section_max_tokens:
            kind, window = "section", None
            text, tokens = section["text"], section["token_count"]
            p_start, p_end = section["page_start"], section["page_end"]
        else:
            kind, window = "window", (c.chunk_index - 1, c.chunk_index + 1)
            text, tokens, p_start, p_end = _window(conn, c.section_id, *window)
        if used + tokens > max_tokens or any(near_duplicate(u.text, text) for u in units):
            continue
        units.append(ContextUnit(
            doc_id=c.doc_id, title=c.title, section_id=c.section_id, section_number=c.section_number,
            heading_path=section["heading_path"], header=c.header, page_start=p_start, page_end=p_end,
            text=text, kind=kind, score=c.best_score, tokens=tokens, chunk_ids=[c.chunk_id], window=window,
            release=c.payload.get("release_label", ""), external_ok=bool(c.payload.get("external_ok", False)),
        ))
    return units
