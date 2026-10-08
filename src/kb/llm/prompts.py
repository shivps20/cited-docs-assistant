"""The answer prompt, and checking the citations in what the model wrote.

The model sees numbered sources and must cite them as [n]. The list of sources under the
answer is built in code from the numbers it actually cited, so a source line can never name
a document or page that was not in the context.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from kb.core.domain import Domain, get_domain
from kb.retrieve.assemble import ContextUnit

NOT_FOUND = "I could not find the answer to this question in the available documents."

# {synonyms}, {reference_label} and {reference_example} come from the domain file (kb.core.domain).
_SYSTEM_TEMPLATE = """You are a technical documentation assistant. You answer questions using ONLY \
the numbered sources given in the user's message.

Rules:
1. Use only facts stated in the sources. Never use outside knowledge and never guess.
2. Read the question for its intent: different words for the same thing still match (e.g. {synonyms}).
3. Answer whenever the sources contain information that addresses the question, even if only \
partly or in different words; if something asked is not covered, say so in one sentence at the end. \
Only when no source addresses the question at all, reply with exactly this sentence and nothing else:
   {not_found}
   A source about a different product, version, database or protocol than the one asked about does \
not address it (for example, Apache instructions do not answer a question about NGINX).
4. After every statement, step or paragraph, cite the source(s) it comes from in square brackets, \
e.g. [1] or [2][3]. Cite only source numbers that appear in the message.
5. Copy exactly, never paraphrase: {reference_label} (e.g. {reference_example}), URLs, commands, \
queries, file names, paths, parameter names and values. Put commands, queries and configuration \
snippets in code blocks.
6. Be complete and detailed, never just one line. Start with the direct answer, then add every \
relevant detail the sources give: what each item does, accepted values and defaults, how to set, \
change or check it (with the exact commands, queries or examples shown), prerequisites, notes and \
warnings.
7. For procedures, give every step as a numbered list in the order of the source.
8. If the sources say which releases or versions they apply to, state it.
9. For comparisons, describe each item from its own sources, then state the differences explicitly.
10. State facts directly. Do not talk about the sources themselves: no phrases such as "the \
sources provide", "according to the documentation" or "this can be found in".
11. Do not end with a list of sources or a line of citation numbers; sources are listed automatically."""


def system_prompt(domain: Domain | None = None) -> str:
    """The answer rules, with the domain's synonym examples and article-number format filled in."""
    d = domain or get_domain()
    return _SYSTEM_TEMPLATE.format(synonyms=d.synonym_examples, not_found=NOT_FOUND,
                                   reference_label=d.reference_label, reference_example=d.reference_example)


def release_text(release: str) -> str:
    """Release label for the prompt and source lines ('all releases' when unrestricted)."""
    return "all releases" if release in ("", "any") else release


def format_context(context: Sequence[ContextUnit]) -> str:
    """The context units as numbered sources: '[n] title (applies to ...) | Section ... | pages' + text."""
    blocks = []
    for n, u in enumerate(context, start=1):
        label = f"[{n}] {u.title} (applies to {release_text(u.release)}) | Section {u.heading_path} | {u.pages}"
        blocks.append(f"{label}\n{u.text.strip()}")
    return "\n\n".join(blocks)


def build_messages(question: str, context: Sequence[ContextUnit]) -> list[dict]:
    """System and user messages for the LLM: the rules, the numbered sources, then the question."""
    user = (f"Sources:\n\n{format_context(context)}\n\n"
            f"Question: {question}\n\n"
            # The last instruction weighs most with a small model: phrased answer-first, because
            # "if they do not contain the answer, reply ..." made it refuse broad but answerable questions.
            f"Answer the question from the sources above, citing them as [n]. Only if none of the "
            f"sources addresses the question, reply exactly: {NOT_FOUND}")
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


# ------------------------------------------------------------------------------------- citations

_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_REFUSAL_SENTENCE = re.compile(r"(?:^|(?<=[.!?\n]))[ \t]*(?:I\s+)?could\s+not\s+find\s+the\s+answer[^.\n]*\.?",
                               re.IGNORECASE)
_NOT_FOUND_SENTENCE = re.compile(r"[ \t]*" + r"\s+".join(map(re.escape, NOT_FOUND.split())), re.IGNORECASE)


@dataclass
class Citations:
    """The answer after citation checking, with the cited and invalid source numbers."""
    text: str               # answer with invalid citation markers removed
    cited: list[int]        # valid source numbers, in order of first citation
    invalid: list[int]      # numbers the model cited that were not in the context
    refused: bool           # the model said the sources do not contain the answer
    dropped_not_found: bool = False   # a contradictory not-found sentence was removed from an answer


def check_citations(answer: str, n_sources: int) -> Citations:
    """Keep [n] markers that point at a source; drop the rest. '[1, 3]' is accepted as [1][3]."""
    cited: list[int] = []
    invalid: list[int] = []

    def replace(match: re.Match) -> str:
        """Rewrite one [..] marker keeping only valid numbers, recording cited and invalid ones."""
        numbers = [int(x) for x in re.split(r"\s*[,;]\s*", match.group(1))]
        valid = [n for n in numbers if 1 <= n <= n_sources]
        invalid.extend(n for n in numbers if n not in valid and n not in invalid)
        cited.extend(n for n in valid if n not in cited)
        return "".join(f"[{n}]" for n in valid)

    text = _CITATION.sub(replace, answer)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text).strip()   # no space left before punctuation
    text = tidy_citations(text)
    # A refusal says nothing else: once sentences starting "I could not find the answer ..." and
    # citation markers are removed, nothing is left (covers 'NOT_FOUND [1][2][3]' and rewordings).
    # Anything with content is an answer, cited or not; a not-found sentence tacked onto it is removed.
    residual = _REFUSAL_SENTENCE.sub("", re.sub(_MARKERS, "", text)).strip(" .\t\n")
    refused = not residual
    dropped = False
    if refused:
        cited = []
    else:
        cleaned = re.sub(r"\n{3,}", "\n\n", _NOT_FOUND_SENTENCE.sub("", text)).strip()
        dropped, text = cleaned != text, cleaned
    return Citations(text=NOT_FOUND if refused else text, cited=cited, invalid=invalid, refused=refused,
                     dropped_not_found=dropped)


def _normalise_snippet(text: str) -> str:
    """Lowercase, prompts such as 'SQL>' removed, whitespace collapsed: for matching quoted commands."""
    text = re.sub(r"\b[A-Za-z]{2,10}\s*(?:<>|>)\s*", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def infer_sources(answer: str, context: Sequence[ContextUnit], min_chars: int = 12) -> list[int]:
    """Source numbers whose text contains code the answer quotes (inline `code` or ``` blocks).

    Used when the model cites nothing: commands, queries and paths are copied verbatim, so they
    show reliably where the answer came from. Snippets shorter than `min_chars` are ignored.
    """
    blocks = re.findall(r"```[^\n]*\n(.*?)```", answer, re.DOTALL)
    snippets = [line for block in blocks for line in block.splitlines()]
    snippets += re.findall(r"`([^`\n]+)`", re.sub(r"```.*?```", "", answer, flags=re.DOTALL))
    keys = {s for s in map(_normalise_snippet, snippets) if len(s) >= min_chars}
    texts = [_normalise_snippet(u.text) for u in context]
    return [n for n, text in enumerate(texts, start=1) if any(k in text for k in keys)]


_MARKERS = r"(?:\[\d+\])+"


def tidy_citations(text: str) -> str:
    """Remove citation markers that only repeat others.

    '[1][1]' -> '[1]'; 'command [1]. [1]' -> 'command [1].'; and a final line holding only
    markers is dropped when every number in it is already cited above (models often end with one).
    """
    text = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", text)
    text = re.sub(rf"({_MARKERS})([.!?])[ \t]+\1(?=[ \t]*(?:\n|$))", r"\1\2", text)
    lines = text.rstrip().split("\n")
    while len(lines) > 1 and re.fullmatch(rf"\s*{_MARKERS}\s*", lines[-1]):
        earlier = set(re.findall(r"\[(\d+)\]", "\n".join(lines[:-1])))
        if not set(re.findall(r"\[(\d+)\]", lines[-1])) <= earlier:
            break
        lines.pop()
    return "\n".join(lines).rstrip()


URL = re.compile(r"https?://[^\s<>()\[\]{}|`'\"]+")
REFERENCES_HEADING = "Articles and links in the cited sections:"


def missing_references(answer: str, cited: Sequence[tuple[int, ContextUnit]], limit: int = 12) -> list[str]:
    """Article numbers (domain reference_patterns) and URLs in the cited sources that the answer lacks.

    Returned as ready-to-append lines ('- KB0012345 [2]'), in source order, each with the [n] of the
    first source containing it. The project rule is that every article number and URL in a cited
    section appears in the answer; the 7B model often drops them, so code adds them instead.
    """
    domain = get_domain()
    lines, seen = [], set()
    for n, unit in cited:
        for ref in domain.find_references(unit.text) + [u.rstrip(".,:;") for u in URL.findall(unit.text)]:
            if ref in seen or ref in answer:
                continue
            seen.add(ref)
            lines.append(f"- {ref} [{n}]")
    return lines[:limit]


def source_line(n: int, unit: ContextUnit) -> str:
    """'[2] Admin Guide (R2026x), Section 2.2.3 Configure the Metadata, pp. 9-10', plus any near-identical
    copies: '· same text: Other Guide (R2026x), Section 2.2.3, p. 9'."""
    line = f"[{n}] {unit.title} ({release_text(unit.release)}), Section {unit.heading}, {unit.pages}"
    copies = [f"{s.title} ({release_text(s.release)}), Section {s.section_number}, {s.pages}" for s in unit.same_text]
    return line + (" · same text: " + "; ".join(copies) if copies else "")
