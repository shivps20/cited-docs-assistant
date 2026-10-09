"""Answer evaluation against the golden set (Phase 3 step 6).

Every golden question goes through the full pipeline (retrieval, gate, LLM, citation checks) with
full access and no release filter. Per question:

  status        answerable questions must be answered, unanswerable ones refused
  must_include  share of the question's key strings present in the answer
  refs / urls   share of the expected answer's article numbers (domain reference_patterns) and URLs present
  citations     precision: share of cited sources on a golden document + pages;
                doc recall: share of golden documents cited (matters for comparisons).
                A source's near-identical copies in other documents (same_text) count as cited too.
  context_hit   the context sent to the LLM contained a golden source (separates retrieval
                failures from generation failures)
  style         answer had no [n] markers / tacked-on not-found sentence / talk about "the sources"
  faithfulness  optional LLM judge: share of the answer's claims supported by the context

The summary splits answerable from unanswerable questions; the report keeps every answer.
"""

import re
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from kb.answer.pipeline import ANSWERED, Answer
from kb.core.domain import get_domain
from kb.evaluation.retrieval import _overlaps, _percentile
from kb.llm.judge import Verdict

URL = re.compile(r"https?://[^\s)\]>,;'\"]+")
META_TALK = re.compile(r"\b(?:the|these|provided) (?:sources?|context)\b|according to the documentation|"
                       r"can be found in the documentation", re.IGNORECASE)


@dataclass
class AnswerResult:
    """Scores and details for one golden question."""

    qid: str
    qtype: str
    answerable: bool
    status: str                         # answered | not_found
    refused_by: str | None              # gate | llm | None
    status_ok: bool
    must_found: int
    must_total: int
    missing: list[str]
    ref_found: int
    ref_total: int
    urls_found: int
    urls_total: int
    cited: int                          # number of sources listed under the answer
    cited_correct: int                  # of those, on a golden document + pages
    golden_docs: int
    golden_docs_cited: int
    context_hit: bool
    no_markers: bool                    # answered without any [n] marker
    dropped_not_found: bool             # a contradictory not-found sentence was removed
    meta_talk: bool                     # talks about "the sources" instead of answering
    invalid_citations: int
    top_score: float | None
    retrieval_ms: float
    generation_ms: float
    judge_ms: float
    total_ms: float
    output_tokens: int | None
    faithfulness: float | None = None
    unsupported: list[str] = field(default_factory=list)
    judge_error: str | None = None
    answer: str = ""
    raw_output: str = ""
    sources: list[str] = field(default_factory=list)
    trace_id: str = ""
    route: str = "answer"                 # answer | compare (searched once per side)
    sides: list[str] = field(default_factory=list)
    retried: bool = False                 # answered on the second attempt after a refusal (TD-23)
    read_sections: int = 0                # sections added by the comparison read step


def normalise(text: str) -> str:
    """Lowercase, markdown emphasis/code marks removed, whitespace collapsed: for string matching."""
    return re.sub(r"\s+", " ", re.sub(r"[`*_]", "", text.lower())).strip()


def contains(answer: str, needle: str) -> bool:
    """Is `needle` in `answer`, ignoring case, markdown marks and whitespace differences?"""
    return normalise(needle) in normalise(answer)


def _locations(source) -> list[tuple[str, int, int]]:
    """(doc_id, page_start, page_end) of a cited source or context unit and of its near-identical copies
    (dicts on a Source, SameText objects on a ContextUnit)."""
    copies = [(s["doc_id"], s["page_start"], s["page_end"]) if isinstance(s, dict) else (s.doc_id, s.page_start, s.page_end)
              for s in source.same_text]
    return [(source.doc_id, source.page_start, source.page_end), *copies]


def _on_golden(source, golden_sources: list[dict]) -> bool:
    """Is the source, or one of its copies, on a golden document and pages?"""
    return any(_overlaps(doc, start, end, golden_sources) for doc, start, end in _locations(source))


def score_answer(q: dict, answer: Answer, verdict: Verdict | None, total_ms: float) -> AnswerResult:
    """Score one Answer against its golden question (and the judge's verdict, if any)."""
    answerable = bool(q["sources"])
    answered = answer.status == ANSWERED
    text = answer.text if answered else ""
    expected = q.get("expected_answer", "")

    must = q.get("must_include", []) if answerable else []
    missing = [m for m in must if not contains(text, m)]
    refs = sorted(set(get_domain().find_references(expected))) if answerable else []
    urls = sorted({u.rstrip(".") for u in URL.findall(expected)}) if answerable else []
    golden_docs = {s["doc_id"] for s in q["sources"]}
    correct = [s for s in answer.sources if _on_golden(s, q["sources"])]
    cited_docs = {doc for s in answer.sources for doc, _, _ in _locations(s)}
    cited_markers = bool(re.search(r"\[\d+\]", text))
    timings = answer.timings_ms

    return AnswerResult(
        qid=q["id"], qtype=q["type"], answerable=answerable, status=answer.status, refused_by=answer.refused_by,
        status_ok=answered == answerable,
        must_found=len(must) - len(missing), must_total=len(must), missing=missing,
        ref_found=sum(1 for x in refs if x in text), ref_total=len(refs),
        urls_found=sum(1 for u in urls if u in text), urls_total=len(urls),
        cited=len(answer.sources), cited_correct=len(correct),
        golden_docs=len(golden_docs), golden_docs_cited=len(golden_docs & cited_docs),
        context_hit=any(_on_golden(u, q["sources"]) for u in answer.context),
        no_markers=answered and not cited_markers,
        dropped_not_found=any("contradictory" in n for n in answer.notices),
        meta_talk=bool(META_TALK.search(text)),
        invalid_citations=len(answer.invalid_citations),
        top_score=answer.gate.top_score,
        retrieval_ms=sum(timings.get(k, 0.0) for k in ("embed", "search", "rerank", "assemble")),
        generation_ms=timings.get("generate", 0.0), judge_ms=(verdict.seconds * 1000 if verdict else 0.0),
        total_ms=total_ms,
        output_tokens=answer.generation.output_tokens if answer.generation else None,
        faithfulness=verdict.faithfulness if verdict else None,
        unsupported=verdict.unsupported if verdict else [],
        judge_error=verdict.error if verdict else None,
        answer=answer.text, raw_output=answer.generation.text if answer.generation else "",
        sources=[s.line for s in answer.sources], trace_id=answer.trace_id,
        route=answer.route, sides=list(answer.sides), retried=answer.retried_from > 0,
        read_sections=answer.read_sections,
    )


def _ratio(found: int, total: int) -> float | None:
    """found / total, or None when there is nothing to find."""
    return found / total if total else None


def _mean(values: list[float | None]) -> float | None:
    """Mean of the values that are not None (None if there are none)."""
    present = [v for v in values if v is not None]
    return statistics.mean(present) if present else None


def summarize_answers(results: list[AnswerResult]) -> dict:
    """Aggregate metrics: answerable quality, refusals, style problems, faithfulness and latency."""
    ans = [r for r in results if r.answerable]
    unans = [r for r in results if not r.answerable]
    answered = [r for r in ans if r.status == ANSWERED]

    def share(rs: list[AnswerResult], pred) -> float | None:
        """Share of results for which `pred` is true (None for an empty list)."""
        return sum(1 for r in rs if pred(r)) / len(rs) if rs else None

    by_type = {}
    for qtype in sorted({r.qtype for r in ans}):
        rs = [r for r in ans if r.qtype == qtype]
        by_type[qtype] = {"n": len(rs), "answered": share(rs, lambda r: r.status == ANSWERED),
                          "must_include": _mean([_ratio(r.must_found, r.must_total) for r in rs]),
                          "faithfulness": _mean([r.faithfulness for r in rs])}
    generated = [r for r in results if r.generation_ms]
    return {
        "n": len(results), "answerable": len(ans), "unanswerable": len(unans),
        # answerable questions
        "answered": share(ans, lambda r: r.status == ANSWERED),
        "wrong_refusals": [r.qid for r in ans if r.status != ANSWERED],
        "must_include": _mean([_ratio(r.must_found, r.must_total) for r in answered]),
        "must_include_all": share(answered, lambda r: r.must_found == r.must_total),
        "article_numbers": _ratio(sum(r.ref_found for r in answered), sum(r.ref_total for r in answered)),
        "urls": _ratio(sum(r.urls_found for r in answered), sum(r.urls_total for r in answered)),
        "citation_precision": _ratio(sum(r.cited_correct for r in answered), sum(r.cited for r in answered)),
        "citation_doc_recall": _ratio(sum(r.golden_docs_cited for r in answered),
                                      sum(r.golden_docs for r in answered)),
        "context_hit": share(ans, lambda r: r.context_hit),
        "faithfulness": _mean([r.faithfulness for r in answered]),
        "fully_faithful": share([r for r in answered if r.faithfulness is not None], lambda r: r.faithfulness == 1),
        "judge_errors": sum(1 for r in answered if r.judge_error),
        # style problems (answered questions)
        "no_markers": share(answered, lambda r: r.no_markers),
        "dropped_not_found": share(answered, lambda r: r.dropped_not_found),
        "meta_talk": share(answered, lambda r: r.meta_talk),
        "invalid_citations": sum(r.invalid_citations for r in answered),
        "retried": [r.qid for r in results if r.retried],      # answered only on the second attempt
        # unanswerable questions
        "refused": share(unans, lambda r: r.status != ANSWERED),
        "wrong_answers": [r.qid for r in unans if r.status == ANSWERED],
        "refused_by_gate": sum(1 for r in unans if r.refused_by == "gate"),
        "refused_by_llm": sum(1 for r in unans if r.refused_by == "llm"),
        # latency
        "total_ms_p50": _percentile([r.total_ms for r in results], 50),
        "total_ms_p95": _percentile([r.total_ms for r in results], 95),
        "retrieval_ms_p50": _percentile([r.retrieval_ms for r in results], 50),
        "generation_ms_p50": _percentile([r.generation_ms for r in generated], 50),
        "judge_ms_p50": _percentile([r.judge_ms for r in results if r.judge_ms], 50),
        "by_type": by_type,
    }


def run_answer_eval(answer_fn: Callable[[str], Answer], golden: list[dict],
                    judge_fn: Callable[[Answer], Verdict] | None = None,
                    progress: Callable[[AnswerResult], None] = lambda _: None) -> dict:
    """Answer every golden question, judge answered ones (if judge_fn), and score them.

    Returns {"summary": ..., "questions": [...]}.
    """
    results = []
    for q in golden:
        start = time.perf_counter()
        answer = answer_fn(q["question"])
        verdict = judge_fn(answer) if judge_fn and answer.status == ANSWERED else None
        result = score_answer(q, answer, verdict, (time.perf_counter() - start) * 1000)
        results.append(result)
        progress(result)
    return {"summary": summarize_answers(results), "questions": [asdict(r) for r in results]}
