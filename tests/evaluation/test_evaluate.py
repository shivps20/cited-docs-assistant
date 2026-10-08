import pytest

from kb.evaluation.retrieval import EvalConfig, judge, run_eval
from kb.retrieve.assemble import ContextUnit
from kb.retrieve.pipeline import SearchResult
from kb.retrieve.rerank import rerank
from kb.retrieve.search import Candidate


def cand(doc, pages, score=0.5, rerank_score=None):
    return Candidate(point_id=f"{doc}{pages}", chunk_id=f"{doc}#1#0", doc_id=doc, section_id=f"{doc}#1",
                     section_number="1", chunk_index=0, title=doc, header=doc, text="t", page_start=pages[0],
                     page_end=pages[1], score=score, rerank_score=rerank_score)


def unit(doc, pages):
    return ContextUnit(doc_id=doc, title=doc, section_id=f"{doc}#1", section_number="1", heading_path="1",
                       header=doc, page_start=pages[0], page_end=pages[1], text="t", kind="section", score=1.0,
                       tokens=10)


def result(cands, units=(), rerank_ms=0.0):
    return SearchResult(list(cands), list(units), "trace", {"rerank": rerank_ms})


SOURCES = [{"doc_id": "mssql", "pages": [22]}, {"doc_id": "oracle", "pages": [20]}]


def test_judge_rank_context_and_all_docs():
    r = result([cand("launcher", (4, 4)), cand("mssql", (21, 23)), cand("mssql", (50, 50)),
                cand("oracle", (20, 20))], units=[unit("mssql", (22, 22))])
    assert judge(r, SOURCES) == (2, True, True)
    only_one = result([cand("mssql", (22, 22))])
    assert judge(only_one, SOURCES) == (1, False, False)          # oracle missing from top 5
    wrong_pages = result([cand("mssql", (60, 61))])
    assert judge(wrong_pages, SOURCES)[0] is None


def test_run_eval_metrics():
    golden = [
        {"id": "q1", "type": "lookup", "question": "a", "sources": [{"doc_id": "d1", "pages": [1]}]},
        {"id": "q2", "type": "lookup", "question": "b", "sources": [{"doc_id": "d2", "pages": [5]}]},
        {"id": "q3", "type": "howto", "question": "c", "sources": [{"doc_id": "d3", "pages": [9]}]},
        {"id": "q4", "type": "unanswerable", "question": "d", "sources": []},
    ]
    canned = {
        "a": result([cand("d1", (1, 1), rerank_score=0.98)], units=[unit("d1", (1, 1))], rerank_ms=100),
        "b": result([cand("x", (1, 1), rerank_score=0.4)] * 2 + [cand("d2", (5, 5), rerank_score=0.3)], rerank_ms=300),
        "c": result([cand("x", (1, 1), rerank_score=0.2)], rerank_ms=200),
        "d": result([cand("x", (1, 1), rerank_score=0.05)], rerank_ms=400),
    }
    report = run_eval(lambda q, cfg: canned[q], [EvalConfig("hybrid+rr", rerank=True)], golden)
    m = report["hybrid+rr"]["summary"]
    assert m["n"] == 3
    assert m["recall@1"] == pytest.approx(1 / 3)
    assert m["recall@5"] == pytest.approx(2 / 3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3) / 3)
    assert m["context_recall"] == pytest.approx(1 / 3)
    assert m["by_type"]["lookup"] == {"n": 2, "recall@5": 1.0, "mrr": pytest.approx((1 + 1 / 3) / 2)}
    assert m["correct_top_score"]["min"] == 0.98
    assert m["unanswerable_top_score"]["max"] == 0.05
    assert m["rerank_ms_p50"] in (200, 300)
    assert {q["qid"] for q in report["hybrid+rr"]["questions"]} == {"q1", "q2", "q3", "q4"}


class LengthReranker:
    def __init__(self):
        self.seen = 0

    def score(self, query, passages):
        self.seen += len(passages)
        return [float(len(p)) for p in passages]


def test_rerank_top_only_scores_head_and_keeps_tail_order():
    cands = [cand("a", (1, 1)), cand("bbbb", (1, 1)), cand("cc", (1, 1)), cand("d", (1, 1))]
    for c, header in zip(cands, ["a", "bbbb", "cc", "d"], strict=True):
        c.header = header
    rr = LengthReranker()
    ranked = rerank(rr, "q", cands, top=3)
    assert rr.seen == 3
    assert [c.doc_id for c in ranked] == ["bbbb", "cc", "a", "d"]
    assert ranked[-1].rerank_score is None
