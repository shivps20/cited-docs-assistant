"""Follow-up condensing: rewrite a question that depends on earlier turns into a standalone one.

"and on Oracle?" after "Which port does the app server use on MSSQL?" cannot be searched on its
own; the local LLM rewrites it as "Which port does the app server use on Oracle?". Retrieval and
the answer then use the standalone question.

Only questions that look like follow-ups are condensed (word rules in `needs_condensing`), which
saves an LLM call (~1-2 s) for self-contained questions. Condensing always runs on the local model:
the conversation can contain text from documents that may not be sent to an external LLM.
A rewrite that looks wrong (empty, an answer instead of a question, far too long) falls back to
the original question.
"""

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from kb.llm.providers import LLMError, LLMProvider

HISTORY_TURNS = 6            # messages of history given to the condenser
ANSWER_CHARS = 400           # assistant answers are cut to this length in the history
MAX_FOLLOW_UP_WORDS = 15     # longer questions with a back-reference are assumed self-contained
SHORT_QUESTION_WORDS = 6     # questions this short are always treated as follow-ups

_OPENERS = re.compile(r"^\s*(and|also|then|so|but|or|what about|how about|same for|what if|and what|and how|"
                      r"and on|and for|and in|is it|is that|does it|does that|can it|can i also)\b", re.IGNORECASE)
# Pronouns and words that only make sense relative to something already said
# ("any other port?", "is there an alternative?", "what else …?").
_BACK_REFERENCE = re.compile(r"\b(it|its|this|that|these|those|they|them|there|the same|above|previous|earlier|"
                             r"former|latter|other|another|else|instead|alternatives?|different|again)\b",
                             re.IGNORECASE)

CONDENSE_SYSTEM = """You rewrite the user's last question so that it can be understood without the \
conversation. Use the conversation only to fill in what the question refers to (product, component, \
release, database, error). Keep names, versions, error codes and commands exactly as written. Do not \
answer the question and do not add anything that was not asked. Reply with the rewritten question only, \
on one line."""


@dataclass
class Condensed:
    """The question to search for, and how it was obtained."""

    question: str                 # standalone question (the original when not condensed)
    condensed: bool               # True when the LLM rewrote it
    reason: str                   # why it was or was not condensed
    seconds: float = 0.0


def needs_condensing(question: str, history: Sequence[dict]) -> bool:
    """Does the question look like it depends on earlier turns? Never without history."""
    if not history:
        return False
    words = question.split()
    if len(words) <= SHORT_QUESTION_WORDS or _OPENERS.match(question):
        return True
    return len(words) <= MAX_FOLLOW_UP_WORDS and bool(_BACK_REFERENCE.search(question))


def condense_messages(question: str, history: Sequence[dict]) -> list[dict]:
    """Messages for the condenser: the rules, the recent conversation and the question to rewrite."""
    lines = []
    for m in history[-HISTORY_TURNS:]:
        if m["role"] == "user":
            lines.append(f"User: {m.get('standalone_query') or m['content']}")
        else:
            text = re.sub(r"\s+", " ", re.sub(r"\[\d+\]", "", m["content"])).strip()
            lines.append(f"Assistant: {text[:ANSWER_CHARS]}{'…' if len(text) > ANSWER_CHARS else ''}")
    user = "Conversation:\n" + "\n".join(lines) + f"\n\nLast question: {question}\n\nRewritten question:"
    return [{"role": "system", "content": CONDENSE_SYSTEM}, {"role": "user", "content": user}]


def clean_rewrite(text: str, question: str) -> str | None:
    """The rewritten question from the model's reply, or None if it does not look like one."""
    line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
    line = re.sub(r"^(rewritten question|standalone question|question)\s*:\s*", "", line, flags=re.IGNORECASE)
    line = line.strip().strip('"').strip("'").strip()
    if not line or len(line.split()) > max(40, 3 * len(question.split())):
        return None
    if "could not find" in line.lower():
        return None
    return line


def condense(provider: LLMProvider | None, question: str, history: Sequence[dict]) -> Condensed:
    """Standalone version of `question`; the original when it is self-contained or condensing fails."""
    if provider is None:
        return Condensed(question, False, "no condenser configured")
    if not needs_condensing(question, history):
        return Condensed(question, False, "self-contained" if history else "first question")
    start = time.perf_counter()
    try:
        reply = provider.generate(condense_messages(question, history)).text
    except LLMError as e:
        return Condensed(question, False, f"condenser failed: {e}", time.perf_counter() - start)
    rewrite = clean_rewrite(reply, question)
    seconds = time.perf_counter() - start
    if rewrite is None:
        return Condensed(question, False, "rewrite rejected; original question kept", seconds)
    return Condensed(rewrite, rewrite != question, "follow-up rewritten", seconds)
