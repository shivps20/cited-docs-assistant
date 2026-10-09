"""Comparison answering, version 1: one search per side.

    "What certificate requirements apply to SAML delegation on Cloud versus HTTPS for on-premises?"
      → decompose (local LLM, JSON): [{label: "SAML delegation on Cloud", query: "…"},
                                      {label: "HTTPS for on-premises services", query: "…"}]
      → search_kb once per side (the user's groups; a release only if the user named it)
      → merge: up to UNITS_PER_SIDE units per side, taken in turns, within COMPARE_MAX_TOKENS
      → the normal gate, LLM and citation checks, with a prompt that names the sides.

Decomposition always runs on the local model (the question may name internal topics) with the
answer model's settings, so Ollama does not reload the model. If the reply is not usable, the
question is answered on the normal path instead.
"""

import json
import re
import time
from dataclasses import dataclass

from kb.agent.tools import KBTools
from kb.llm.providers import LLMError, LLMProvider
from kb.retrieve.assemble import COMPARE_MAX_TOKENS, UNITS_PER_SIDE, merge_contexts
from kb.retrieve.pipeline import SearchResult
from kb.retrieve.release import releases_in
from kb.retrieve.search import Candidate

MAX_SIDES = 3
MAX_QUERY_WORDS = 40
MIN_QUERY_WORDS = 2            # after removing the other sides' labels
MAX_LABEL_WORDS = 10

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
                   max_tokens: int = COMPARE_MAX_TOKENS) -> SearchResult:
    """Search once per side and merge the results into one context with slots for every side.

    The candidates of all sides are combined (best score per chunk) so the gate sees the best match.
    """
    results = [(side, tools.search_kb(side.query, release=side_release(question, side, tools.request.release),
                                      side=side.label))
               for side in sides]
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
