"""Context assembly: turn ranked chunks into the units of context the LLM will see.

For each ranked chunk, best first:
* its section is small (<= SECTION_MAX_TOKENS): use the whole section (from SQLite);
* otherwise: the chunk plus its neighbours (+/-1) in the same section; further hits in that
  section widen the same window instead of adding a new unit.
Stops at MAX_UNITS units / MAX_TOKENS tokens. Units whose text is the same apart from small edits
(the MSSQL and Oracle guides share most sections, often with a word or two changed) are kept once:
the higher-ranked copy wins, and the copies from other documents are recorded on it (`same_text`)
so the answer can cite them too.

Executive summaries are front matter: they restate the whole document in general terms and, ranked
high on broad wording, took context slots from content sections (TD-14). Their rerank score is
multiplied by FRONT_MATTER_FACTOR for the ordering here. Numbered introductions are not demoted:
they are golden sources for some questions (e.g. which services a guide covers).

Comparisons search once per side; `merge_contexts` combines the sides' units, reserving slots for
each side so one side cannot fill the whole context.
"""

import re
import sqlite3
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from kb.retrieve.search import Candidate

MAX_UNITS = 6               # was 4: the unit cap, not the token budget, cut off overview sections (TD-17)
MAX_TOKENS = 3000
SECTION_MAX_TOKENS = 800
NEAR_DUPLICATE_RATIO = 0.95  # word-level similarity; MSSQL/Oracle copies with cosmetic edits are 0.96-0.99,
                             # sections that differ in content are 0.92 or lower
FRONT_MATTER_FACTOR = 0.8    # executive summaries rank as if their rerank score were 20% lower
_FRONT_MATTER = re.compile(r"^(?:0\s+)?executive\s+summary\b", re.IGNORECASE)
UNITS_PER_SIDE = 3           # comparisons: context slots reserved for each side
COMPARE_MAX_TOKENS = 4000    # comparisons: total context budget (two or three sides)


def page_text(page_start: int, page_end: int) -> str:
    """Page reference, e.g. 'p. 9' or 'pp. 9-10'."""
    return f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"


@dataclass
class SameText:
    """A near-identical copy of a context unit in another document: not sent to the LLM, cited with the unit."""
    doc_id: str
    title: str
    section_id: str
    section_number: str
    page_start: int
    page_end: int
    release: str = ""

    @property
    def pages(self) -> str:
        """Page reference, e.g. 'p. 9' or 'pp. 9-10'."""
        return page_text(self.page_start, self.page_end)

    @property
    def citation(self) -> str:
        """Short citation, e.g. 'MSSQL Guide, Section 3.4, p. 8'."""
        return f"{self.title}, Section {self.section_number}, {self.pages}"


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
    same_text: list[SameText] = field(default_factory=list)   # near-identical copies in other documents
    side: str = ""                          # comparisons: the side this unit was found for

    @property
    def pages(self) -> str:
        """Page reference, e.g. 'p. 9' or 'pp. 9-10'."""
        return page_text(self.page_start, self.page_end)

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


def window_text(conn: sqlite3.Connection, section_id: str, max_tokens: int) -> tuple[str, int, int, int, int]:
    """The first chunks of a section up to about max_tokens (at least one): text, tokens, page range and
    the last chunk index read."""
    rows = conn.execute("SELECT chunk_index, text, token_count, page_start, page_end FROM chunks "
                        "WHERE section_id = ? ORDER BY chunk_index", (section_id,)).fetchall()
    taken, used = [], 0
    for r in rows:
        if taken and used + r["token_count"] > max_tokens:
            break
        taken.append(r)
        used += r["token_count"]
    if not taken:
        return "", 0, 0, 0, 0
    return ("\n\n".join(r["text"] for r in taken), used, min(r["page_start"] for r in taken),
            max(r["page_end"] for r in taken), taken[-1]["chunk_index"])


def near_duplicate(a: str, b: str, ratio: float = NEAR_DUPLICATE_RATIO) -> bool:
    """Are two texts the same apart from small edits? Compared word by word, cheapest bounds first."""
    if a == b:
        return True
    wa, wb = a.split(), b.split()
    matcher = SequenceMatcher(None, wa, wb, autojunk=False)
    return matcher.real_quick_ratio() >= ratio and matcher.quick_ratio() >= ratio and matcher.ratio() >= ratio


def _add_same_text(unit: ContextUnit, c: Candidate, page_start: int, page_end: int) -> None:
    """Record candidate `c`'s section as a copy of `unit`, once per other document."""
    if c.doc_id == unit.doc_id or any(s.doc_id == c.doc_id for s in unit.same_text):
        return
    unit.same_text.append(SameText(doc_id=c.doc_id, title=c.title, section_id=c.section_id,
                                   section_number=c.section_number, page_start=page_start, page_end=page_end,
                                   release=c.payload.get("release_label", "")))


def is_front_matter(c: Candidate) -> bool:
    """Is the candidate from an executive summary (unnumbered front matter)?"""
    return bool(_FRONT_MATTER.match(c.header.rsplit(" > ", 1)[-1].strip()))


def demote_front_matter(ranked: list[Candidate], factor: float = FRONT_MATTER_FACTOR) -> list[Candidate]:
    """Reorder the reranked candidates with executive summaries' scores multiplied by `factor`.

    Only reranked candidates move (their scores are comparable); the unreranked tail keeps its place.
    """
    head = [c for c in ranked if c.rerank_score is not None]
    tail = [c for c in ranked if c.rerank_score is None]
    head.sort(key=lambda c: c.rerank_score * (factor if is_front_matter(c) else 1.0), reverse=True)
    return head + tail


def assemble(conn: sqlite3.Connection, ranked: list[Candidate], *, max_units: int = MAX_UNITS,
             max_tokens: int = MAX_TOKENS, section_max_tokens: int = SECTION_MAX_TOKENS,
             min_score: float | None = None) -> list[ContextUnit]:
    """min_score (reranker scale): skip chunks below it, and chunks that were not reranked."""
    units: list[ContextUnit] = []
    for c in demote_front_matter(ranked):
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

        if section["token_count"] <= section_max_tokens:
            kind, window = "section", None
            text, tokens = section["text"], section["token_count"]
            p_start, p_end = section["page_start"], section["page_end"]
        else:
            kind, window = "window", (c.chunk_index - 1, c.chunk_index + 1)
            text, tokens, p_start, p_end = _window(conn, c.section_id, *window)
        # Checked before the limits: a copy ranked below the cut-off is still cited with its twin.
        twin = next((u for u in units if near_duplicate(u.text, text)), None)
        if twin is not None:
            _add_same_text(twin, c, p_start, p_end)
            continue
        if len(units) >= max_units or used + tokens > max_tokens:
            continue
        units.append(ContextUnit(
            doc_id=c.doc_id, title=c.title, section_id=c.section_id, section_number=c.section_number,
            heading_path=section["heading_path"], header=c.header, page_start=p_start, page_end=p_end,
            text=text, kind=kind, score=c.best_score, tokens=tokens, chunk_ids=[c.chunk_id], window=window,
            release=c.payload.get("release_label", ""), external_ok=bool(c.payload.get("external_ok", False)),
        ))
    return units


def merge_contexts(sides: list[tuple[str, list[ContextUnit]]], *, per_side: int = UNITS_PER_SIDE,
                   max_tokens: int = COMPARE_MAX_TOKENS) -> list[ContextUnit]:
    """One context from each side's units: taken in turns (best of each side first), at most `per_side`
    units per side and `max_tokens` in total.

    A section found for several sides is used once, labelled with the first side that found it; a
    near-identical section is recorded as a copy (`same_text`) of the unit already chosen.
    """
    merged: list[ContextUnit] = []
    taken = {label: 0 for label, _ in sides}
    for rank in range(max((len(units) for _, units in sides), default=0)):
        for label, units in sides:
            if rank >= len(units) or taken[label] >= per_side:
                continue
            unit = units[rank]
            if any(u.section_id == unit.section_id for u in merged):
                continue
            twin = next((u for u in merged if near_duplicate(u.text, unit.text)), None)
            if twin is not None:
                copies = [SameText(unit.doc_id, unit.title, unit.section_id, unit.section_number,
                                   unit.page_start, unit.page_end, unit.release), *unit.same_text]
                for copy in copies:
                    if copy.doc_id != twin.doc_id and all(s.doc_id != copy.doc_id for s in twin.same_text):
                        twin.same_text.append(copy)
                continue
            if sum(u.tokens for u in merged) + unit.tokens > max_tokens:
                continue
            unit.side = label
            merged.append(unit)
            taken[label] += 1
    return merged
