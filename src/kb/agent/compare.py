"""Comparison answering: one search per side, then (optionally) read more sections per side.

    "What certificate requirements apply to SAML delegation on Cloud versus HTTPS for on-premises?"
      → decompose (local LLM, JSON): [{label: "SAML delegation on Cloud", query: "…"},
                                      {label: "HTTPS for on-premises services", query: "…"}]
      → search_kb once per side (the user's groups; a release only if the user named it)
      → merge: up to UNITS_PER_SIDE units per side, taken in turns, within COMPARE_MAX_TOKENS
      → read step (KB_COMPARE_READ): the local LLM sees each side's guide as a table of contents,
        with the sections it already has marked, and picks up to READ_PER_SIDE more sections per
        side; the server reads them (access-checked) and adds them to the context
      → the normal gate, LLM and citation checks, with a prompt that names the sides.

The read step exists because each side's search usually finds the right guide but not always the
section with the deciding fact (a prerequisites table, a list of fixed bugs): 11 of the 24 key facts
missed by comparison answers in the 2026-10-09 evaluation were in the right guide but not in context.

Decomposition always runs on the local model (the question may name internal topics) with the
answer model's settings, so Ollama does not reload the model. If the reply is not usable, the
question is answered on the normal path instead.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from kb.agent.tools import KBTools, OutlineEntry
from kb.llm.providers import LLMError, LLMProvider
from kb.retrieve.assemble import (
    COMPARE_MAX_TOKENS,
    UNITS_PER_SIDE,
    ContextUnit,
    merge_contexts,
)
from kb.retrieve.pipeline import SearchResult
from kb.retrieve.release import releases_in
from kb.retrieve.search import Candidate

MAX_SIDES = 3
MAX_QUERY_WORDS = 40
MIN_QUERY_WORDS = 2            # after removing the other sides' labels
MAX_LABEL_WORDS = 10
READ_PER_SIDE = 2          # sections the read step may add per side
READ_MAX_TOKENS = 1500     # all sections added by the read step together

DECOMPOSE_SYSTEM = """You prepare a document search for a question that compares two or more items. \
Split it into one search question per item (2 or 3 items).

Rules for each search question:
- It is about one item only: never mention the other items.
- No comparison words (differ, difference, compare, versus, vs, between).
- Keep the product names, versions, releases, databases and technical terms of the original question.

Example 1
Question: How does configuring the license server differ between Windows and Linux in R2025x?
{"sides": [{"label": "Windows", "query": "How do I configure the license server on Windows in R2025x?"}, \
{"label": "Linux", "query": "How do I configure the license server on Linux in R2025x?"}]}

Example 2
Question: Is the report designer still supported, and how does the guidance differ between R2023x and R2025x?
{"sides": [{"label": "R2023x", "query": "Is the report designer supported in R2023x and how is it set up?"}, \
{"label": "R2025x", "query": "Is the report designer supported in R2025x and how is it set up?"}]}

Reply with JSON only, in this form:
{"sides": [{"label": "short name of the item", "query": "the question about that item"}]}"""


@dataclass
class Side:
    """One item of a comparison: a short label and the question to search for it."""

    label: str
    query: str


@dataclass
class Decomposition:
    """The sides of a comparison (empty when the question could not be split), and why."""

    sides: list[Side]
    reason: str
    seconds: float = 0.0
    raw: str = ""


def decompose_messages(question: str) -> list[dict]:
    """Messages asking the LLM to split the comparison into one search question per side."""
    return [{"role": "system", "content": DECOMPOSE_SYSTEM}, {"role": "user", "content": f"Question: {question}"}]


_CONNECTOR = r"(?:and|or|vs\.?|versus|than)"
_DANGLING = re.compile(r"\s*\b(?:and|or|vs\.?|versus|than|between|from|with|the)\s*(?=[?.,;]|$)", re.IGNORECASE)


def without_labels(query: str, labels: list[str]) -> str:
    """The query with the other sides' labels removed, plus connectors left dangling by that
    ('… differ between MSSQL and ?' → '… differ between MSSQL?')."""
    for label in labels:   # the label with a connector on either side: '… and Oracle', 'R2019x and …'
        query = re.sub(rf"(?:\b{_CONNECTOR}\s+)?(?<!\w){re.escape(label)}(?!\w)(?:\s+{_CONNECTOR}\b)?",
                       " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\s+", " ", query).strip()
    while (cleaned := _DANGLING.sub("", query)) != query:
        query = cleaned
    return re.sub(r"\s+([?.,;])", r"\1", query).strip()


def parse_sides(text: str) -> list[Side]:
    """The sides from the LLM's JSON reply; [] when the reply is not 2-3 usable, distinct sides.

    A side whose question still names another side ('What does R2021x cover that R2019x does not?')
    is kept with the other labels removed, so its search and its release filter are about its own
    item only (TD-22); it is dropped only when too little of the question remains or two sides end
    up with the same question."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = data.get("sides") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    sides = []
    for item in items:
        if not isinstance(item, dict):
            return []
        label, query = str(item.get("label") or "").strip(), str(item.get("query") or "").strip()
        if not label or not query or len(query.split()) > MAX_QUERY_WORDS or len(label.split()) > MAX_LABEL_WORDS:
            return []
        sides.append(Side(label, query))
    if not 2 <= len(sides) <= MAX_SIDES:
        return []
    labels = [s.label for s in sides]
    sides = [Side(s.label, without_labels(s.query, [x for x in labels if x.lower() != s.label.lower()]))
             for s in sides]
    if any(len(s.query.split()) < MIN_QUERY_WORDS for s in sides) or len({s.query.lower() for s in sides}) < len(sides):
        return []
    return sides


def decompose(planner: LLMProvider | None, question: str) -> Decomposition:
    """Split a comparison question into sides with the local LLM; no sides when that fails."""
    if planner is None:
        return Decomposition([], "no local model for decomposition")
    start = time.perf_counter()
    try:
        reply = planner.generate(decompose_messages(question), json_format=True).text
    except LLMError as e:
        return Decomposition([], f"decomposition failed: {e}", time.perf_counter() - start)
    sides = parse_sides(reply)
    reason = f"{len(sides)} sides" if sides else "reply not usable; answered with one search"
    return Decomposition(sides, reason, time.perf_counter() - start, reply)


def side_release(question: str, side: Side, default: int | None) -> int | None:
    """The release for one side's search: a release the user named in the question that this side
    mentions (when the question names several), else the request's release.

    Only releases from the user's own words count, so the LLM cannot add a release filter.
    """
    asked = releases_in(question)
    if len(asked) >= 2:
        mine = [y for y in releases_in(side.query) if y in asked]
        if len(mine) == 1:
            return mine[0]
    return default


def retrieve_sides(tools: KBTools, question: str, sides: list[Side], *, per_side: int = UNITS_PER_SIDE,
                   max_tokens: int = COMPARE_MAX_TOKENS,
                   on_side: Callable[[int, Side], None] | None = None,
                   side_docs: dict[str, str] | None = None) -> SearchResult:
    """Search once per side and merge the results into one context with slots for every side.

    The candidates of all sides are combined (best score per chunk) so the gate sees the best match.
    on_side(index, side) is called before each side's search (1-based index), for progress reports.
    side_docs, if given, receives each side's best document (side label → doc_id) for the read step.
    """
    results = []
    for index, side in enumerate(sides, start=1):
        if on_side:
            on_side(index, side)
        results.append((side, tools.search_kb(side.query, release=side_release(question, side, tools.request.release),
                                              side=side.label)))
    if side_docs is not None:
        side_docs.update({side.label: r.context[0].doc_id for side, r in results if r.context})
    context = merge_contexts([(side.label, r.context) for side, r in results], per_side=per_side,
                             max_tokens=max_tokens)
    best: dict[str, Candidate] = {}
    for _, r in results:
        for c in r.candidates:
            if c.chunk_id not in best or c.best_score > best[c.chunk_id].best_score:
                best[c.chunk_id] = c
    # reranked candidates first (their scores are comparable; the gate reads the top rerank score)
    candidates = sorted(best.values(), key=lambda c: (c.rerank_score is not None, c.best_score), reverse=True)
    return SearchResult(candidates, context, tools.trace.trace_id, {})


# ------------------------------------------------------------------------------------- read step

READ_SYSTEM = """You help answer a question that compares items. Each item is described in a guide; \
you get each guide's table of contents, and sections marked * are already provided.

Choose the sections that most likely hold facts still needed to answer the question, such as \
versions, settings and values, prerequisites, lists of changes or fixed bugs, steps.

Rules:
- At most 2 sections per item; never a section marked *.
- Only section numbers from that item's own table of contents.
- If the provided sections already answer the question, choose none.

Reply with JSON only, in this form:
{"read": [{"item": "A", "section": "4.1.2"}]}"""


@dataclass
class ReadPlan:
    """Sections the read step chose (side label, section id), and why / how long it took."""

    reads: list[tuple[str, str]]
    reason: str
    seconds: float = 0.0
    raw: str = ""


def _outline_line(entry: OutlineEntry, have: set[str]) -> str:
    """'* 4.1.2 cas.properties' (provided) or '  4.1.2 cas.properties'; the heading without its number."""
    heading = entry.heading
    if heading.split(" ", 1)[0] == entry.number:
        heading = heading.split(" ", 1)[1] if " " in heading else ""
    return f"{'*' if entry.section_id in have else ' '} {entry.number} {heading}".rstrip()


def read_messages(question: str, items: list[tuple[str, str, str, list[OutlineEntry]]], have: set[str]) -> list[dict]:
    """Messages asking which sections to read: per item (letter, side label, guide title, outline)."""
    blocks = [f"Item {letter}: {label}\nGuide: {title}\nTable of contents:\n"
              + "\n".join(_outline_line(e, have) for e in entries)
              for letter, label, title, entries in items]
    user = f"Question: {question}\n\n" + "\n\n".join(blocks)
    return [{"role": "system", "content": READ_SYSTEM}, {"role": "user", "content": user}]


def parse_reads(text: str, items: list[tuple[str, str, str, list[OutlineEntry]]], have: set[str],
                per_side: int = READ_PER_SIDE) -> list[tuple[str, str]] | None:
    """(side label, section id) pairs from the reply: only sections in that item's outline, not yet
    provided, at most per_side per item, no repeats. None when the reply is not usable JSON."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    picks = data.get("read") if isinstance(data, dict) else None
    if not isinstance(picks, list):
        return None
    by_letter = {letter.upper(): (label, {e.number: e.section_id for e in entries})
                 for letter, label, _, entries in items}
    reads: list[tuple[str, str]] = []
    for pick in picks:
        if not isinstance(pick, dict):
            continue
        letter = str(pick.get("item") or "").strip().upper()[:1]
        number = str(pick.get("section") or "").strip().rstrip(".")
        if letter not in by_letter:
            continue
        label, numbers = by_letter[letter]
        section_id = numbers.get(number)
        if section_id is None or section_id in have or any(sid == section_id for _, sid in reads):
            continue
        if sum(1 for lab, _ in reads if lab == label) < per_side:
            reads.append((label, section_id))
    return reads


def plan_reads(planner: LLMProvider | None, tools: KBTools, question: str, context: list[ContextUnit],
               side_docs: dict[str, str]) -> ReadPlan:
    """Ask the local LLM which sections of each side's guide to read next (see READ_SYSTEM)."""
    if planner is None or not side_docs:
        return ReadPlan([], "no local model or no documents for the read step")
    have = {u.section_id for u in context} | {s.section_id for u in context for s in u.same_text}
    items = []
    for i, (label, doc_id) in enumerate(side_docs.items()):
        entries = tools.outline(doc_id)
        title = next((u.title for u in context if u.doc_id == doc_id), doc_id)
        if entries and any(e.section_id not in have for e in entries):
            items.append((chr(65 + i), label, title, entries))
    if not items:
        return ReadPlan([], "nothing left to read")
    start = time.perf_counter()
    try:
        reply = planner.generate(read_messages(question, items, have), json_format=True).text
    except LLMError as e:
        return ReadPlan([], f"read step failed: {e}", time.perf_counter() - start)
    reads = parse_reads(reply, items, have)
    seconds = time.perf_counter() - start
    if reads is None:
        return ReadPlan([], "reply not usable", seconds, reply)
    return ReadPlan(reads, f"{len(reads)} sections chosen", seconds, reply)


def read_sections(tools: KBTools, reads: list[tuple[str, str]], *, max_tokens: int = READ_MAX_TOKENS) -> list[ContextUnit]:
    """The chosen sections as context units labelled with their side, within max_tokens together;
    sections the user may not see are skipped (tools.read_section checks access)."""
    units, used = [], 0
    for side, section_id in reads:
        unit = tools.read_section(section_id, side=side)
        if unit is None or used + unit.tokens > max_tokens:
            continue
        units.append(unit)
        used += unit.tokens
    return units
