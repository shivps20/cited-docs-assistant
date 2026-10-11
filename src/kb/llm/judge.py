"""LLM-as-judge faithfulness check: is every claim in an answer supported by its context?

The judge (qwen2.5 by default, the same model that wrote the answer) splits the answer into
factual claims and marks each one supported or not by the numbered sources, replying in JSON.
Faithfulness = supported claims / all claims.

Caveat: a model judging its own output is lenient on its own mistakes, so treat the score as a
lower bound on problems (unsupported claims it does flag are worth reading), not as proof.

Two corrections in code (Phase 7, TD-12):
- A claim the judge leaves without a usable quote still counts as supported when every distinctive
  value in it (codes such as ORA-01157, numbers of 3+ digits, file names, paths, identifiers, `inline
  code`) occurs in the sources: the 7B judge often lumps a list into one claim without a quote.
- Long answers are judged in parts (paragraph groups of about PART_CHARS), so the judge's JSON reply
  never runs past its output cap (9 long answers failed that way in the Phase 5 run).
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from kb.llm.prompts import format_context
from kb.llm.providers import LLMError, LLMProvider
from kb.retrieve.assemble import ContextUnit

JUDGE_SYSTEM = """You verify answers against sources. Be strict and literal.

Split the ANSWER into its individual factual claims: every step or instruction, value, default, \
command, query, path, name, number, URL and statement of fact. Ignore citation markers such as \
[1], lines that only introduce a list or code (ending with ':'), and sentences that only say what \
the sources do or do not contain.

For each claim, first search the SOURCES (including tables) for the text that states it, then decide:
- "evidence": the shortest exact quote from the SOURCES that states the claim, copied character \
for character; "" if there is none.
- "supported": true if the evidence states the claim (wording may differ, but commands, values, \
names, paths and numbers must match exactly); false if there is no evidence or it contradicts the claim.

Reply with JSON only, in exactly this form:
{"claims": [{"claim": "<the claim, short>", "evidence": "<exact quote>", "supported": true, "source": 1}, \
{"claim": "<the claim, short>", "evidence": "", "supported": false, "source": null}]}"""


@dataclass
class Claim:
    """One factual claim from the answer and the judge's verdict on it."""

    text: str
    supported: bool
    source: int | None = None
    evidence: str = ""              # the judge's quote from the sources
    evidence_found: bool = False    # the quote really occurs in the sources (checked in code)
    by_values: bool = False         # supported because all its distinctive values occur in the sources


@dataclass
class Verdict:
    """The judge's result for one answer: its claims, or the error that prevented judging."""

    claims: list[Claim] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0

    @property
    def faithfulness(self) -> float | None:
        """Share of claims supported by the sources (None if the judge failed or found no claims)."""
        if self.error or not self.claims:
            return None
        return sum(c.supported for c in self.claims) / len(self.claims)

    @property
    def unsupported(self) -> list[str]:
        """Text of the claims the judge found unsupported."""
        return [c.text for c in self.claims if not c.supported]


def judge_messages(answer: str, context: Sequence[ContextUnit]) -> list[dict]:
    """Messages for the judge: rules, then the numbered sources and the answer (markers removed)."""
    clean = re.sub(r"\[\d+\]", "", answer).strip()
    user = f"SOURCES:\n\n{format_context(context)}\n\nANSWER:\n{clean}"
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def _words(text: str) -> list[str]:
    """Lowercase alphanumeric words (punctuation, markdown and table pipes ignored)."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def evidence_in_sources(evidence: str, sources_text: str) -> bool:
    """Does the quoted evidence occur in the sources, word for word (case, punctuation and markdown
    ignored)? A quote shortened with '...' and a quote of several lines are checked piece by piece:
    judges often quote the same fact from two sources on two lines, which never occurs as one passage
    (trace aaadab91: 3 of 9 claims wrongly unsupported).

    Deliberately strict: a judge that 'quotes' the source with one value changed (HTTP-Redirect
    instead of HTTP-POST) must not count as evidence; every piece must occur as it is.
    """
    haystack = " " + " ".join(_words(sources_text)) + " "
    pieces = [" ".join(_words(p)) for p in re.split(r"\.\.\.|…|\n", evidence)]
    pieces = [p for p in pieces if p]
    return bool(pieces) and all(f" {p} " in haystack for p in pieces)


PART_CHARS = 1200      # answers longer than this are judged in parts
_VALUE = re.compile(r"`([^`]{3,})`|\b[A-Za-z]{2,}-\d+\b|\b\d{3,}(?:\.\d+)*\b|\b[\w-]+(?:[./\\][\w-]+)+\b")


def distinctive_values(text: str) -> list[str]:
    """Values in a claim that can be checked literally: `inline code`, codes like ORA-01157, numbers
    of 3+ digits (ports, versions), file names, paths and dotted identifiers."""
    values = []
    for match in _VALUE.finditer(text):
        value = (match.group(1) or match.group(0)).strip()
        if value and value not in values:
            values.append(value)
    return values


def values_in_sources(claim: str, sources_text: str) -> bool:
    """Does every distinctive value of the claim occur in the sources (case ignored)? False when the
    claim has none (then only a quote can support it)."""
    values = distinctive_values(claim)
    haystack = sources_text.lower()
    return bool(values) and all(v.lower() in haystack for v in values)


def split_answer(answer: str, limit: int = PART_CHARS) -> list[str]:
    """The answer in parts of about `limit` characters, cut only between paragraphs (code blocks and
    lists stay whole); a short answer is one part."""
    if len(answer) <= limit:
        return [answer]
    paragraphs, block, in_code = [], [], False
    for line in answer.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        block.append(line)
        if not line.strip() and not in_code:
            paragraphs.append("\n".join(block).strip())
            block = []
    if block:
        paragraphs.append("\n".join(block).strip())
    parts, current = [], ""
    for paragraph in (p for p in paragraphs if p):
        if current and len(current) + len(paragraph) > limit:
            parts.append(current)
            current = ""
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts


def parse_verdict(reply: str, sources_text: str = "") -> Verdict:
    """Read the judge's JSON reply; a malformed reply becomes a Verdict with an error.

    With `sources_text`, each quote is checked against the sources; a claim the judge calls
    unsupported but backs with a real quote is counted as supported (small judges contradict
    their own evidence), and one it calls supported with a quote that is not there is not.
    """
    try:
        data = json.loads(reply)
        claims = []
        for c in data.get("claims", []):
            if not isinstance(c, dict):
                continue
            evidence = str(c.get("evidence") or "").strip()
            claim = Claim(text=str(c.get("claim", "")).strip(), supported=bool(c.get("supported")),
                          source=c.get("source") if isinstance(c.get("source"), int) else None, evidence=evidence)
            if sources_text:
                claim.evidence_found = evidence_in_sources(evidence, sources_text)
                claim.supported = claim.evidence_found
                if not claim.supported and values_in_sources(claim.text, sources_text):
                    claim.supported = claim.by_values = True
            claims.append(claim)
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        return Verdict(error=f"unreadable judge reply ({type(e).__name__}): {reply[:120]!r}")
    # Lead-ins ('To set these parameters, use the following commands:') state nothing to verify.
    return Verdict(claims=[c for c in claims if c.text and not c.text.rstrip().endswith(":")])


def judge_faithfulness(judge: LLMProvider, answer: str, context: Sequence[ContextUnit]) -> Verdict:
    """Ask the judge model which claims of `answer` the `context` supports (quotes checked in code).
    A long answer is judged in parts and the claims are merged; a part that fails sets the error."""
    sources_text = "\n".join(u.text for u in context)
    merged = Verdict()
    for part in split_answer(answer):
        try:
            generation = judge.generate(judge_messages(part, context), json_format=True)
        except LLMError as e:
            return Verdict(claims=merged.claims, error=str(e), seconds=merged.seconds)
        verdict = parse_verdict(generation.text, sources_text)
        merged.seconds += generation.seconds
        if verdict.error:
            return Verdict(claims=merged.claims, error=verdict.error, seconds=merged.seconds)
        merged.claims.extend(verdict.claims)
    return merged
