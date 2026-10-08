"""Release filter for a chat question, sticky for the rest of the session.

"How do I install the platform on R2025x?" sets the session's release to R2025x; later questions
without a release keep using it; "any release" (or "all releases") clears it. A question naming
two or more releases (a comparison) is not filtered and leaves the sticky release unchanged.

The release is read from the user's own words only, never from a condensed rewrite, so the LLM
can never introduce a release filter the user did not ask for.
"""

import re
from dataclasses import dataclass

_RELEASE = re.compile(r"\b(?:V6)?R(20\d\d)x\b", re.IGNORECASE)
_CLEAR = re.compile(r"\b(any|all|every|regardless of(?: the)?)\s+releases?\b|\bno release filter\b",
                    re.IGNORECASE)


@dataclass
class ReleaseChoice:
    """The release filter for this question and the session's sticky release afterwards."""

    release: int | None       # year used in the retrieval filter (None = all releases)
    sticky: str | None        # session release after this question, e.g. 'R2025x'
    reason: str               # 'from question', 'sticky', 'cleared', 'several releases', 'none'


def releases_in(text: str) -> list[int]:
    """Distinct release years mentioned in the text, in order ('R2025x' -> 2025)."""
    years: list[int] = []
    for match in _RELEASE.finditer(text):
        year = int(match.group(1))
        if year not in years:
            years.append(year)
    return years


def resolve_release(question: str, sticky: str | None) -> ReleaseChoice:
    """Which release filters this question, and what the session remembers afterwards."""
    if _CLEAR.search(question):
        return ReleaseChoice(None, None, "cleared")
    years = releases_in(question)
    if len(years) == 1:
        label = f"R{years[0]}x"
        return ReleaseChoice(years[0], label, "from question")
    if len(years) > 1:
        return ReleaseChoice(None, sticky, "several releases")
    if sticky:
        return ReleaseChoice(int(sticky[1:5]), sticky, "sticky")
    return ReleaseChoice(None, None, "none")
