"""Retrieval evaluation against the golden set.

A retrieved chunk is a hit if it comes from a cited document and overlaps the cited pages.
Per configuration (search mode, reranking depth, reranker input length):
  Recall@1/5/10   share of answerable questions with a hit in the top k
  MRR             mean of 1/rank of the first hit (0 if none in the candidate list)
  Context recall  share of questions whose assembled context (what the LLM will see) has a hit
  All-docs@5      compare questions: every cited document has a hit in the top 5
  Latency         median / 95th percentile of total retrieval time, and of the rerank stage
  Top scores      best score for answerable questions (hit at rank 1) vs unanswerable ones;
                  the gap between them is where the "not found" threshold goes.
Questions run with full access (all groups) and no release filter; access and release
filtering are checked separately.
"""

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from kb.core.config import get_settings
from kb.ingest.manifest import Document
from kb.retrieve.pipeline import SearchResult
from kb.retrieve.search import PUBLIC_GROUP


@dataclass(frozen=True)
class EvalConfig:
    """One retrieval configuration to evaluate: search mode, reranking on/off, depth and input length."""
    name: str
    mode: str = "hybrid"
    rerank: bool = False
    rerank_top: int | None = None
    max_length: int = 512


CONFIGS = [
    EvalConfig("dense", mode="dense"),
    EvalConfig("sparse", mode="sparse"),
    EvalConfig("hybrid"),
    EvalConfig("hybrid+rr10", rerank=True, rerank_top=10, max_length=512),
    EvalConfig("hybrid+rr15@256", rerank=True, rerank_top=15, max_length=256),
    EvalConfig("hybrid+rr30", rerank=True, rerank_top=30, max_length=512),
]


@dataclass
class QuestionResult:
    """Outcome of one golden question under one configuration."""
    qid: str
    qtype: str
    first_hit: int | None            # 1-based rank of the first hit among the candidates
    context_hit: bool
    docs_found_at5: bool             # all cited documents hit within the top 5
    top_score: float | None
    total_ms: float
    rerank_ms: float
    top_results: list[str] = field(default_factory=list)


def _overlaps(doc_id: str, page_start: int, page_end: int, sources: list[dict]) -> bool:
    """Is the chunk from a cited document and on (or overlapping) its cited pages?"""
    return any(doc_id == s["doc_id"] and page_start <= s["pages"][-1] and page_end >= s["pages"][0]
               for s in sources)


def judge(result: SearchResult, sources: list[dict]) -> tuple[int | None, bool, bool]:
    """(rank of first hit, context has a hit, every cited document hit in top 5)."""
    first = next((rank for rank, c in enumerate(result.candidates, start=1)
                  if _overlaps(c.doc_id, c.page_start, c.page_end, sources)), None)
    context_hit = any(_overlaps(x.doc_id, x.page_start, x.page_end, sources)       # a unit or its same-text copies
                      for u in result.context for x in (u, *u.same_text))
    top5 = result.candidates[:5]
    cited_docs = {s["doc_id"] for s in sources}
    all_docs = all(any(_overlaps(c.doc_id, c.page_start, c.page_end, [s for s in sources if s["doc_id"] == d])
                       for c in top5) for d in cited_docs)
    return first, context_hit, all_docs


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of the values (0 for an empty list)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def summarize(results: list[QuestionResult], unanswerable: list[QuestionResult]) -> dict:
    """Aggregate metrics (recall@k, MRR, context recall, latency, top scores) for one configuration."""
    n = len(results) or 1

    def recall(k: int, rs: list[QuestionResult]) -> float:
        """Share of questions with a hit within the top k."""
        return sum(1 for r in rs if r.first_hit is not None and r.first_hit <= k) / (len(rs) or 1)

    def mrr(rs: list[QuestionResult]) -> float:
        """Mean reciprocal rank of the first hit (0 when there is none)."""
        return sum(1 / r.first_hit for r in rs if r.first_hit) / (len(rs) or 1)

    by_type = {}
    for qtype in sorted({r.qtype for r in results}):
        rs = [r for r in results if r.qtype == qtype]
        by_type[qtype] = {"n": len(rs), "recall@5": recall(5, rs), "mrr": mrr(rs)}
    compare = [r for r in results if r.qtype == "compare"]
    correct_top = [r.top_score for r in results if r.first_hit == 1 and r.top_score is not None]
    unans_top = [r.top_score for r in unanswerable if r.top_score is not None]
    return {
        "n": len(results),
        "recall@1": recall(1, results), "recall@5": recall(5, results), "recall@10": recall(10, results),
        "mrr": mrr(results),
        "context_recall": sum(r.context_hit for r in results) / n,
        "all_docs@5": sum(r.docs_found_at5 for r in compare) / (len(compare) or 1),
        "latency_ms_p50": _percentile([r.total_ms for r in results + unanswerable], 50),
        "latency_ms_p95": _percentile([r.total_ms for r in results + unanswerable], 95),
        "rerank_ms_p50": _percentile([r.rerank_ms for r in results + unanswerable], 50),
        "by_type": by_type,
        "correct_top_score": {"min": min(correct_top, default=None),
                              "median": statistics.median(correct_top) if correct_top else None},
        "unanswerable_top_score": {"max": max(unans_top, default=None), "scores": unans_top},
    }


def run_eval(search_fn: Callable[[str, EvalConfig], SearchResult], configs: list[EvalConfig],
             golden: list[dict], progress: Callable[[str], None] = lambda _: None) -> dict:
    """Run every golden question through every configuration; returns per-config summaries and details."""
    report = {}
    for config in configs:
        answerable, unanswerable = [], []
        for q in golden:
            start = time.perf_counter()
            result = search_fn(q["question"], config)
            total_ms = (time.perf_counter() - start) * 1000
            first, context_hit, all_docs = judge(result, q["sources"]) if q["sources"] else (None, False, False)
            qr = QuestionResult(
                qid=q["id"], qtype=q["type"], first_hit=first, context_hit=context_hit, docs_found_at5=all_docs,
                top_score=result.top_score, total_ms=total_ms, rerank_ms=result.timings_ms.get("rerank", 0.0),
                top_results=[f"{c.doc_id}#{c.section_number} p{c.page_start}" for c in result.candidates[:5]],
            )
            (answerable if q["sources"] else unanswerable).append(qr)
            progress(f"{config.name}: {q['id']}")
        report[config.name] = {
            "config": asdict(config),
            "summary": summarize(answerable, unanswerable),
            "questions": [asdict(r) for r in answerable + unanswerable],
        }
    return report


def load_golden(ids: list[str] | None = None) -> list[dict]:
    """Golden questions from KB_GOLDEN_PATH (default eval/golden.json), optionally only the given ids."""
    golden = json.loads(get_settings().golden_path.read_text(encoding="utf-8"))
    return [q for q in golden if not ids or q["id"] in ids]


def full_access(docs: list[Document]) -> list[str]:
    """Every access group used in the manifest: evaluation questions run as a user who sees all documents."""
    return sorted({PUBLIC_GROUP, *(g for d in docs for g in d.allowed_groups)})


def restricted_docs(docs: list[Document]) -> list[Document]:
    """Documents the public group may not see (for the access-filter check)."""
    return [d for d in docs if PUBLIC_GROUP not in d.allowed_groups]
