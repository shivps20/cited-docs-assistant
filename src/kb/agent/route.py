"""Question routing: does the question compare two or more things?

A comparison needs evidence for every side, which one search often does not deliver: the side
that matches the wording best fills the context (TD-14). Comparisons therefore go to the
comparison path (`kb.agent.compare`), which searches once per side.

Word rules, no LLM call. A question is a comparison when it:
* uses comparison wording: compare / comparison, vs / versus, differ, difference(s) between;
* names two or more different releases (R2024x and R2026x);
* asks for a choice: "Should …, X or Y?", "Is it better to …, X or Y?", "Which is better …";
* names two documents (TD-22): "the Apache and the F5 load balancing guides", "the two Launcher
  documents", "…, while the single-port guide uses …".

"different" alone is not comparison wording ("What are the different components …?" asks for a list).
Cross-document questions without comparison wording stay on the normal path.
"""

import re
from dataclasses import dataclass

from kb.retrieve.release import releases_in

ANSWER = "answer"
COMPARE = "compare"

_COMPARISON_WORDS = re.compile(
    r"\bcompar(?:e|es|ed|ing|ison|isons)\b|\bvs\.?(?=\s)|\bversus\b|\bdiffer(?:s|ed|ing)?\b"
    r"|\bdifferences?\s+(?:between|from|in)\b", re.IGNORECASE)
_CHOICE = re.compile(r"^\s*(?:should|would|is\s+it\s+better|which\s+is\s+better|which\s+one)\b.*\bor\b",
                     re.IGNORECASE)
_DOC = r"(?:guides?|documents?|checklists?|white\s?papers?|presentations?|best\s+practices)"
_TWO_DOCUMENTS = re.compile(
    rf"\bthe\s+[\w-]+(?:\s+[\w-]+){{0,4}}\s+and\s+the\s+(?:[\w-]+\s+){{0,4}}{_DOC}\b"   # the X (guide) and the Y guide
    rf"|\b(?:two|both)\s+(?:[\w-]+\s+){{0,3}}{_DOC}\b"                                # the two Launcher documents
    rf"|\bwhile\s+the\s+(?:[\w-]+\s+){{0,3}}{_DOC}\b",                                # X recommends …, while the Y guide …
    re.IGNORECASE)


@dataclass
class Route:
    """Which path answers the question, and the rule that decided it."""

    kind: str          # answer | compare
    reason: str        # e.g. 'comparison wording: differ', 'two releases: R2024x, R2026x'


def route_question(question: str) -> Route:
    """Route a (standalone) question to the normal path or the comparison path."""
    match = _COMPARISON_WORDS.search(question)
    if match:
        return Route(COMPARE, f"comparison wording: {match.group(0).strip().lower()}")
    years = releases_in(question)
    if len(years) >= 2:
        return Route(COMPARE, "two releases: " + ", ".join(f"R{y}x" for y in years))
    if _CHOICE.search(question):
        return Route(COMPARE, "choice between options")
    if _TWO_DOCUMENTS.search(question):
        return Route(COMPARE, "two documents named")
    return Route(ANSWER, "no comparison wording")
