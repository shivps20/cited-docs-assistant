"""Ingestion step 2: rebuild a document's section structure from the Docling parse.

Docling reports every heading at the same level and also tags many non-headings as headings
("For example:", "Analysis:", page headers, code lines). This module turns the flat item
stream into numbered sections that match the document's own numbering, so answers can cite
"Section 3.1.7, p. 20":

* Numbered body headings ("3.2. Server Configuration ...") give the hierarchy (level = parts).
  A numbered heading is accepted if it appears in the document's Table of Contents or is a
  plausible next number; otherwise (e.g. a numbered step "1. Unzip the media") it is text.
* Unnumbered headings are accepted only if they appear in the TOC. They become children of
  the most recent numbered section and get positional numbers (3.1 -> 3.1.1, 3.1.2, ...).
  This matches the convention the documents use when referring to their own sub-sections.
* Everything else Docling called a heading is demoted to ordinary text.
* Content before the first numbered section is section "0" (cover + executive summary).
* Dropped: the TOC itself, "Document History" sections, page headers/footers that leaked
  into the body, and lines repeated on a large share of pages.
* PPTX: one section per slide, titled by the slide title; agenda and disclaimer slides dropped.
* Documents without a TOC (e.g. DOCX) fall back to accepting all headings.
* Only the run of TOC pages near the start is the TOC; a table Docling labels as TOC later in the
  document is an ordinary table. A 'Contents' heading in the front matter drops only the TOC pages.
* A TOC that lists titles without numbers makes its matching headings top-level sections 1, 2, …
  (TD-18: without these three rules some documents lost almost all of their text).

The core (build_structure) works on plain Element objects so it can be tested without Docling.
"""

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from docling_core.types.doc import (
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    TableItem,
)

from kb.core.domain import get_domain

# Section titles dropped together with their sub-sections.
DROP_SECTION_TITLES = {"document history", "table of contents", "contents"}
# Slide titles dropped in presentations.
DROP_SLIDE_TITLES = {"agenda", "proprietary disclosure statement", "table of contents", "contents"}
# A body line repeated on at least this share of pages (and >= 3 times) is page furniture.
REPEATED_LINE_SHARE = 0.3
MAX_HEADING_CHARS = 150
# The table of contents must start within the first pages (8, or 10% of a long document);
# tables Docling labels as TOC further on are treated as ordinary tables.
TOC_START_MAX_PAGE = 8
TOC_START_MAX_SHARE = 0.1
# A TOC where fewer than this share of entries carry a section number is an unnumbered TOC.
UNNUMBERED_TOC_SHARE = 0.2

_NUMBER = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+(?=\S)")
_LEADER = re.compile(r"(?:\s*\.){3,}|…+|\uFFFD")
_BULLET_CELL = re.compile(r"^(?:[o•▪◦\-–]\s*)+$")


# ---------------------------------------------------------------------------- data model

@dataclass
class Element:
    """One body item from the parse, simplified."""
    kind: str                 # heading | text | list | code | table | toc
    text: str
    page: int
    depth: int = 0            # list nesting depth
    rows: list[list[str]] | None = None  # table / toc cells


@dataclass
class Block:
    """A piece of section content: a paragraph, list, code block or table (as Markdown)."""
    kind: str                 # text | list | code | table
    text: str
    page_start: int
    page_end: int


@dataclass
class Section:
    """A numbered section with its title, place in the hierarchy and content blocks."""
    number: str               # "3.1.7"; "0" = front matter; slide number for PPTX
    title: str
    level: int
    parent: str | None
    heading_page: int
    blocks: list[Block] = field(default_factory=list)

    @property
    def page_start(self) -> int:
        """First page of the section (its heading or earliest block)."""
        return min([self.heading_page] + [b.page_start for b in self.blocks])

    @property
    def page_end(self) -> int:
        """Last page of the section."""
        return max([self.heading_page] + [b.page_end for b in self.blocks])

    @property
    def chars(self) -> int:
        """Total characters of content, used by `kb inspect` to show section sizes."""
        return sum(len(b.text) for b in self.blocks)


@dataclass
class TocEntry:
    """One table-of-contents entry: its number (if printed), title and matching key."""
    number: str | None
    title: str
    key: str                  # normalized title used for matching


@dataclass
class Structure:
    """A document's rebuilt sections plus diagnostics (unmatched TOC entries, demoted headings, removed
    lines).
    """
    sections: list[Section]
    toc: list[TocEntry]
    unmatched_toc: list[TocEntry]
    demoted_headings: list[str]
    removed_lines: dict[str, int]

    def get(self, number: str) -> Section | None:
        """The section with this number, or None."""
        return next((s for s in self.sections if s.number == number), None)

    def path(self, section: Section) -> list[Section]:
        """Ancestors from the top level down to (and including) the section."""
        by_number = {s.number: s for s in self.sections}
        chain = [section]
        while chain[-1].parent and chain[-1].parent in by_number:
            chain.append(by_number[chain[-1].parent])
        return chain[::-1]


# ---------------------------------------------------------------------------- text helpers

def split_number(text: str) -> tuple[str | None, str]:
    """'3.1. The config.xml File' -> ('3.1', 'The config.xml File')."""
    text = text.strip()
    match = _NUMBER.match(text)
    if not match:
        return None, text
    return match.group(1), text[match.end():].strip()


def normalize_title(text: str) -> str:
    """Matching key: lowercase alphanumerics, no numbering, no stray page numbers."""
    _, title = split_number(text)
    words = re.sub(r"[^a-z0-9]+", " ", title.lower()).split()
    return " ".join(w for w in words if not w.isdigit())


_STOPWORDS = {"a", "an", "and", "for", "in", "of", "on", "or", "the", "to", "with"}


def _content_words(key: str) -> set[str]:
    """Words of a title key without stop words, for checking whether two titles are related."""
    return {w for w in key.split() if w not in _STOPWORDS}


def normalize_line(text: str) -> str:
    """Key for detecting repeated page furniture: case/whitespace-insensitive, digits masked."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.strip().lower()))


def number_tuple(number: str) -> tuple[int, ...]:
    """'3.1.7' -> (3, 1, 7)."""
    return tuple(int(p) for p in number.split("."))


def plausible_next(previous: tuple[int, ...] | None, candidate: tuple[int, ...]) -> bool:
    """Is `candidate` a believable next section number after `previous`?

    Allows the next sibling at any level (skipping at most one missed heading) and the first
    child; rejects numbered steps such as "1." in the middle of section 4.4.
    """
    if previous is None:
        return len(candidate) == 1 and candidate[0] <= 2
    allowed = {previous + (1,), previous + (2,)}
    for level in range(len(previous)):
        for step in (1, 2):
            allowed.add(previous[:level] + (previous[level] + step,))
    return candidate in allowed


# ---------------------------------------------------------------------------- table of contents

def toc_row_text(row: list[str]) -> str:
    """Text of one TOC table row, with empty and repeated (merged) cells removed."""
    cells = []
    for cell in row:
        cell = cell.strip()
        if cell and (not cells or cells[-1] != cell):  # merged cells repeat in the grid
            cells.append(cell)
    return " ".join(cells)


def parse_toc(rows: list[str]) -> list[TocEntry]:
    """Entries from TOC row texts like '2.1. What is the Launcher ....... 4'.

    Tolerates split cells, two entries merged into one row, and stray page numbers.
    """
    entries = []
    for row in rows:
        for segment in _LEADER.split(row):
            segment = re.sub(r"^\d+\s+(?=\d+(\.\d+)*\.?\s+\D)", "", segment.strip())  # previous entry's page
            segment = re.sub(r"\s+\d+$", "", segment).strip()                          # this entry's page
            if not segment or segment.isdigit():
                continue
            number, title = split_number(segment)
            key = normalize_title(title)
            if key and key not in DROP_SECTION_TITLES:
                entries.append(TocEntry(number, title, key))
    return entries


FUZZY_MATCH_RATIO = 0.93


def match_toc(key: str, toc: list[TocEntry], *, fuzzy: bool, used: set[int] | None = None) -> TocEntry | None:
    """TOC entry for a heading key. Repeated titles resolve to the first entry not yet used."""
    if not key:
        return None
    used = used or set()
    exact = [e for e in toc if e.key == key]
    if exact:
        return next((e for e in exact if id(e) not in used), exact[0])
    if fuzzy and len(key) >= 8:
        scored = sorted(((SequenceMatcher(None, e.key, key).ratio(), i, e) for i, e in enumerate(toc)),
                        key=lambda t: (-t[0], id(t[2]) in used, t[1]))
        if scored and scored[0][0] >= FUZZY_MATCH_RATIO:
            return scored[0][2]
    return None


def match_toc_loose(key: str, toc: list[TocEntry], *, used: set[int], min_words: int
                    ) -> tuple[TocEntry, str] | None:
    """Looser TOC match for headings Docling mislabelled (list item, code, text) or cut in two.

    Returns (entry, mode); only unused entries are considered. Modes, tried in order:
      exact       same title (at least `min_words` words)
      extra_word  the title with one stray leading word ('Management Program Management (PRG)')
      toc_tail    the tail of a merged TOC row ('... (SUP) ...... RFQ Management (SRC)')
      toc_cut     the TOC row was cut short ('Installing the Search ... Index on' + 'a Single Machine')
      body_cut    the body heading was cut short ('Upgrading from Version 13 or Later')
    """
    words = key.split()
    if not words or len(words) > 15:
        return None
    candidates = [(e, e.key.split()) for e in toc if id(e) not in used]
    rules = [
        ("exact", lambda kw: len(kw) >= min_words and words == kw),
        ("extra_word", lambda kw: len(kw) >= 3 and len(words) == len(kw) + 1 and words[1:] == kw),
        ("toc_tail", lambda kw: len(words) >= 3 and len(kw) > len(words) and kw[-len(words):] == words),
        ("toc_cut", lambda kw: len(kw) >= 5 and len(words) > len(kw) and words[:len(kw)] == kw),
        ("body_cut", lambda kw: len(words) >= 4 and len(kw) > len(words) and kw[:len(words)] == words),
    ]
    for mode, rule in rules:
        for entry, kw in candidates:
            if rule(kw):
                return entry, mode
    return None


def match_heading_prefix(text: str, toc: list[TocEntry], *, used: set[int] | None = None
                         ) -> tuple[TocEntry, str, str, bool] | None:
    """A paragraph that starts with a TOC title Docling merged into it.

    'Sending a System Report to Support The system report is ...' ->
    (entry, 'Sending a System Report to Support', 'The system report is ...', True).
    The title may also be the tail of a TOC entry, because Docling sometimes merges two TOC rows
    ('Configuration Files Optimal Performance Recommendations ...'); the last value says whether
    the whole entry matched. Requires a title of at least 3 words followed by a capitalised sentence.
    """
    words = text.split()
    prefixes: dict[tuple[str, ...], int] = {}
    for n in range(3, min(len(words) - 1, 25) + 1):
        prefixes.setdefault(tuple(normalize_title(" ".join(words[:n])).split()), n)

    best = None
    used = used or set()
    for entry in toc:
        if id(entry) in used:                       # that heading already started a section
            continue
        key_words = entry.key.split()
        for start in range(max(len(key_words) - 2, 0)):
            n = prefixes.get(tuple(key_words[start:]))
            if n and words[n][:1].isupper() and (best is None or n > best[1]):
                best = (entry, n, start == 0)
    if best is None:
        return None
    entry, n, whole = best
    return entry, " ".join(words[:n]), " ".join(words[n:]), whole


# ---------------------------------------------------------------------------- tables

def clean_table_rows(rows: list[list[str]]) -> list[list[str]]:
    """Drop columns that hold only bullet glyphs or nothing (e.g. 'o o o' beside list cells)."""
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [c for c in range(width)
            if any(r[c].strip() and not _BULLET_CELL.match(r[c].strip()) for r in rows)]
    return [[r[c].strip() for c in keep] for r in rows]


def table_markdown(rows: list[list[str]]) -> str:
    """A table as a Markdown pipe table (first row as header), after cleaning empty rows and columns."""
    rows = clean_table_rows(rows)
    if not rows or not rows[0]:
        return ""

    def line(cells: list[str]) -> str:
        """One Markdown table row, with pipes escaped and line breaks flattened."""
        return "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |"

    return "\n".join([line(rows[0]), "| " + " | ".join("---" for _ in rows[0]) + " |"]
                     + [line(r) for r in rows[1:]])


# ---------------------------------------------------------------------------- structure

def _is_boilerplate(text: str) -> bool:
    """Is the line copyright or confidentiality boilerplate that repeats on every page?"""
    return any(p.match(text.strip()) for p in get_domain().boilerplate)


def remove_furniture(elements: list[Element], furniture: set[str], n_pages: int, *, slides: bool = False
                     ) -> tuple[list[Element], dict[str, int]]:
    """Drop legal boilerplate, leaked page headers/footers and lines repeated on many pages.

    slides: in a deck, headings are slide titles and a chapter title often repeats over consecutive slides
    ("Storage Systems" on slides 4-8), so headings neither count nor go as repeated lines; a running footer on
    slides is plain text and still goes. Boilerplate patterns and Docling's furniture apply to everything."""
    repeatable = ("text",) if slides else ("heading", "text")
    pages_by_line: dict[str, set[int]] = defaultdict(set)
    for e in elements:
        if e.kind in repeatable and len(e.text) <= 200:
            pages_by_line[normalize_line(e.text)].add(e.page)
    threshold = max(3, REPEATED_LINE_SHARE * n_pages)
    repeated = {line for line, pages in pages_by_line.items() if len(pages) >= threshold}

    kept, removed = [], defaultdict(int)
    for e in elements:
        if e.kind in ("heading", "text"):
            key = normalize_line(e.text)
            if (_is_boilerplate(e.text) or (key in repeated and e.kind in repeatable)
                    or (key in furniture and len(e.text) <= 200)):
                removed[e.text.strip()[:80]] += 1
                continue
        kept.append(e)
    return kept, dict(removed)


class _Builder:
    """Accumulates sections while walking the document's elements in order."""

    def __init__(self):
        """Start with no sections and no pending list items."""
        self.sections: list[Section] = []
        self.current: Section | None = None
        self.pending_list: list[Element] = []

    def start(self, number: str, title: str, page: int) -> Section:
        """Start a new section (made unique with a '-2' suffix if the number repeats) and return it."""
        self.flush_list()
        base, n = number, 2
        while any(s.number == number for s in self.sections):  # keep numbers unique
            number, n = f"{base}-{n}", n + 1
        parent = None
        parts = base.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:cut])
            if any(s.number == candidate for s in self.sections):
                parent = candidate
                break
        level = 1 if base == "0" else len(parts)
        self.current = Section(number, title, level, parent, page)
        self.sections.append(self.current)
        return self.current

    def add(self, e: Element) -> None:
        """Add an element to the current section; consecutive list items are collected into one list block.
        """
        if e.kind == "list":
            self.pending_list.append(e)
            return
        self.flush_list()
        if e.kind == "table":
            text = table_markdown(e.rows or [])
            kind = "table"
        else:
            text, kind = e.text.strip(), ("code" if e.kind == "code" else "text")
        if text:
            self.current.blocks.append(Block(kind, text, e.page, e.page))

    def flush_list(self) -> None:
        """Write the collected list items as one indented Markdown list block."""
        if not self.pending_list or self.current is None:
            self.pending_list = []
            return
        base = min(e.depth for e in self.pending_list)
        lines = [f"{'  ' * (e.depth - base)}- {e.text.strip()}" for e in self.pending_list]
        self.current.blocks.append(Block("list", "\n".join(lines), self.pending_list[0].page,
                                         self.pending_list[-1].page))
        self.pending_list = []


def _front_toc(elements: list[Element], n_pages: int) -> tuple[list[Element], list[int]]:
    """Keep only the document's real table of contents: the run of consecutive TOC pages near the start.

    Docling sometimes labels an ordinary table late in a document as a TOC (e.g. on page 110 of
    214). Taken as TOC, it made everything up to that page front matter, so no section could start
    and the body was lost. TOC elements outside the leading run become ordinary tables again.
    Returns the elements and the pages of the leading TOC run (empty if there is none).
    """
    pages = sorted({e.page for e in elements if e.kind == "toc"})
    run: list[int] = []
    if pages and pages[0] <= max(TOC_START_MAX_PAGE, n_pages * TOC_START_MAX_SHARE):
        run = [pages[0]]
        for page in pages[1:]:
            if page - run[-1] > 2:                  # allow one page without a TOC table in between
                break
            run.append(page)
    keep = set(run)
    return [e if e.kind != "toc" or e.page in keep else Element("table", e.text, e.page, e.depth, e.rows)
            for e in elements], run


def _drop_sections(sections: list[Section], titles: set[str]) -> list[Section]:
    """Remove sections with one of the given titles (TOC, Document History) and all their sub-sections."""
    doomed = {s.number for s in sections if normalize_title(s.title) in titles}
    return [s for s in sections
            if not any(s.number == d or s.number.startswith(d + ".") for d in doomed)]


def build_structure(elements: list[Element], *, doc_type: str, n_pages: int,
                    furniture: set[str] | None = None) -> Structure:
    """Rebuild the numbered section tree of a PDF/DOCX from Docling elements.

    The TOC is the list of valid titles: headings that match it (exactly, fuzzily or
    loosely for mislabelled or cut headings) start sections and take its numbers; numbered
    headings not in the TOC start sections only if their number is plausible and the TOC
    does not contradict it. Everything up to the TOC page is front matter (section 0).
    PPTX files go to _build_slides instead.
    """
    elements, removed = remove_furniture(elements, furniture or set(), n_pages, slides=doc_type == "pptx")
    if doc_type == "pptx":
        return _build_slides(elements, removed)

    elements, toc_pages = _front_toc(elements, n_pages)
    toc = parse_toc([toc_row_text(row) for e in elements if e.kind == "toc" for row in (e.rows or [])])
    # Everything up to and including the TOC page(s) is front matter (cover, executive summary).
    front_last_page = max(toc_pages, default=0)
    # A TOC that lists titles without section numbers: its headings become top-level sections 1, 2, …
    unnumbered_toc = bool(toc) and sum(1 for t in toc if t.number) < UNNUMBERED_TOC_SHARE * len(toc)
    b = _Builder()
    front = b.start("0", "Front Matter", 1)
    last_numbered: tuple[int, ...] | None = None   # last explicitly numbered section
    child_count = 0                                 # unnumbered children under it so far
    top_count = 0                                   # top-level sections in documents without numbering
    matched: set[int] = set()
    demoted: list[str] = []
    dropping = False                                # inside a dropped section (TOC, Document History)
    dropping_front = False                          # ... started by a heading in the front matter

    toc_by_number = {t.number: t for t in toc if t.number}
    last_chapter = max((int(n) for n in toc_by_number if n.isdigit()), default=None)

    def fits_toc(number: str, key: str) -> bool:
        """May a numbered heading that matched no TOC title start a section?

        The TOC may garble a title ('3.6. Windows Roles Best Practices ...' for 3.5) or stop at a
        shallower level (no 3.1.7), so a heading is rejected only when the TOC contradicts it: the
        TOC gives its number a title with no word in common, or it is a chapter past the TOC's
        last one. That catches numbered lists styled as headings ('6. Can I use self-signed
        certificates?' in an FAQ, while the TOC's 6 is 'References').
        """
        if not toc:
            return True
        if number in toc_by_number:
            return bool(_content_words(key) & _content_words(toc_by_number[number].key))
        return not (number.isdigit() and last_chapter is not None and int(number) > last_chapter)

    def next_number(number: str | None) -> str | None:
        """Number for a section start: explicit, or positional under the last numbered section."""
        nonlocal last_numbered, child_count, top_count
        if number:
            last_numbered, child_count = number_tuple(number), 0
            return number
        if last_numbered is not None:
            child_count += 1
            return ".".join(map(str, last_numbered + (child_count,)))
        if not toc or unnumbered_toc:
            top_count += 1
            return str(top_count)
        return None

    for e in elements:
        if e.kind == "toc":
            continue
        in_front = e.page <= front_last_page
        if dropping and dropping_front and not in_front:
            # A 'Contents' heading in the front matter only covers the TOC pages; otherwise, when no
            # numbered or TOC-matched heading starts a section afterwards, the whole body was lost.
            dropping = dropping_front = False
        new_number = title = None
        remainder: Element | None = None            # paragraph text after a merged-in heading

        if e.kind in ("heading", "text", "list", "code") and len(e.text) <= MAX_HEADING_CHARS:
            number, title = split_number(e.text)
            key = normalize_title(e.text)
            if e.kind == "heading" and key in DROP_SECTION_TITLES:
                dropping = True                     # skip everything until the next section starts
                dropping_front = in_front
                continue
            # Headings match the TOC fuzzily; other items only on an exact or near-exact title
            # (headings Docling labelled as text, list items or code).
            hit, borrow = None, False
            if toc and e.kind in ("heading", "text"):
                hit = match_toc(key, toc, fuzzy=False, used=matched)
                borrow = hit is not None
                if hit is None and e.kind == "heading":
                    hit = match_toc(key, toc, fuzzy=True, used=matched)
            if toc and hit is None:
                loose = match_toc_loose(key, toc, used=matched, min_words=1 if e.kind == "text" else 3)
                # A loose match must also fit the numbering: a workflow step that repeats a later
                # section's title ('Configure the ... metadata to define ...' under 2.1) is not 2.2.3.
                if loose and loose[1] != "toc_tail" and loose[0].number and last_numbered is not None \
                        and not plausible_next(last_numbered, number_tuple(loose[0].number)):
                    loose = None
                # The tail of a merged TOC row is matched for headings only: a list item that ends
                # a TOC title ('Derived Data Server') is usually just a list item.
                if loose and loose[1] == "toc_tail" and e.kind not in ("heading", "text"):
                    loose = None
                if loose:
                    hit, mode = loose
                    borrow = mode != "toc_tail"     # a merged row's number belongs to its first entry
                    if mode == "extra_word":
                        title = " ".join(title.split()[1:])
                    elif mode == "body_cut":
                        title = hit.title           # the heading wrapped; the TOC has the full title
                    elif mode == "toc_cut" and (prefix := match_heading_prefix(e.text, [hit])):
                        # the 'longer heading' is the title with its first paragraph merged in
                        title = split_number(prefix[1])[1]
                        remainder = Element("text", prefix[2], e.page)
            if borrow and not number and hit.number and not unnumbered_toc:
                number = hit.number  # heading lost its number in parsing; the TOC still has it

            if in_front:
                pass
            elif number and (hit or (e.kind == "heading" and plausible_next(last_numbered, number_tuple(number))
                                     and fits_toc(number, key))):
                new_number = next_number(number)
            elif not number and (hit or (e.kind == "heading" and not toc)):
                new_number = next_number(None)
            if new_number and hit:
                matched.add(id(hit))
            if not new_number and e.kind == "heading" and not dropping:
                if b.current is front and key == "executive summary":
                    front.title = "Executive Summary"
                demoted.append(e.text.strip())

        if not new_number and e.kind in ("text", "list") and toc and not in_front:
            prefix = match_heading_prefix(e.text, toc, used=matched)
            if prefix:
                entry, title, rest, whole = prefix
                title = split_number(title)[1]
                new_number = next_number(entry.number if whole else None)
                if new_number:
                    matched.add(id(entry))
                    remainder = Element("text", rest, e.page)

        if new_number:
            dropping = False
            b.start(new_number, title, e.page)
            if remainder:
                b.add(remainder)
            continue
        if dropping:
            continue
        b.add(e)
    b.flush_list()

    sections = _drop_sections(b.sections, DROP_SECTION_TITLES)
    sections = [s for s in sections if s.blocks or any(c.parent == s.number for c in sections)]
    unmatched = [t for t in toc if id(t) not in matched]
    return Structure(sections, toc, unmatched, demoted, removed)


def _build_slides(elements: list[Element], removed: dict[str, int]) -> Structure:
    """One section per slide (consecutive slides with the same title merge, '(n/m)' suffixes removed)."""
    by_slide: dict[int, list[Element]] = defaultdict(list)
    for e in elements:
        by_slide[e.page].append(e)

    b = _Builder()
    demoted = []
    previous: tuple[str, int] | None = None         # (title key, slide) of the last kept slide
    for slide in sorted(by_slide):
        items = by_slide[slide]
        title_el = next((e for e in items if e.kind == "heading"), None)
        title = title_el.text.strip() if title_el else f"Slide {slide}"
        title = re.sub(r"\s*\(\d+\s*/\s*\d+\)$", "", title)   # "Key Functionality (1/2)" -> one section
        key = normalize_title(title)
        if key in DROP_SLIDE_TITLES:
            continue
        # Consecutive slides with the same title continue one section (e.g. a 6-slide walkthrough).
        if not (title_el and previous and previous == (key, slide - 1)):
            b.start(str(slide), title, slide)
        previous = (key, slide)
        for e in items:
            if e is title_el:
                continue
            if e.kind == "heading":
                demoted.append(e.text.strip())
            b.add(e)
    b.flush_list()
    sections = [s for s in b.sections if s.blocks]
    return Structure(sections, [], [], demoted, removed)


# ---------------------------------------------------------------------------- Docling adapter

def load_parsed(path) -> DoclingDocument:
    """Load a cached Docling JSON document, silencing Docling's bounding-box warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Docling warns about bounding boxes clamped to the page
        return DoclingDocument.load_from_json(path)


def extract_elements(dl: DoclingDocument) -> tuple[list[Element], set[str]]:
    """Flatten the body into Elements; also return normalized page-furniture lines."""
    elements: list[Element] = []
    page = 1
    for item, depth in dl.iterate_items():
        if getattr(item, "prov", None):
            page = item.prov[0].page_no
        label = item.label
        if isinstance(item, TableItem):
            rows = [[cell.text for cell in row] for row in item.data.grid]
            elements.append(Element("toc" if label == DocItemLabel.DOCUMENT_INDEX else "table", "", page, rows=rows))
            continue
        text = (getattr(item, "text", "") or "").strip()
        if not text or label == DocItemLabel.PICTURE:
            continue
        if label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
            elements.append(Element("heading", text, page))
        elif label == DocItemLabel.LIST_ITEM:
            elements.append(Element("list", text, page, depth=depth))
        elif label == DocItemLabel.CODE:
            elements.append(Element("code", text, page))
        else:
            elements.append(Element("text", text, page))

    furniture = {normalize_line(getattr(item, "text", "") or "")
                 for item, _ in dl.iterate_items(included_content_layers={ContentLayer.FURNITURE})
                 if (getattr(item, "text", "") or "").strip()}
    return elements, furniture


def document_structure(dl: DoclingDocument, doc_type: str) -> Structure:
    """Elements from a parsed document, turned into its section structure."""
    elements, furniture = extract_elements(dl)
    return build_structure(elements, doc_type=doc_type, n_pages=max(len(dl.pages), 1), furniture=furniture)
