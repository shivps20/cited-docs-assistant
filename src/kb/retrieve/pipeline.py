"""Retrieval pipeline: embed -> filtered search -> rerank -> assemble, every stage traced."""

import sqlite3
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from kb.core.config import get_settings
from kb.core.tracing import Tracer
from kb.retrieve.assemble import MAX_TOKENS, MAX_UNITS, ContextUnit, assemble
from kb.retrieve.rerank import Reranker, rerank
from kb.retrieve.search import Candidate, build_filter, search
from kb.store.embed import Embedder


@dataclass
class SearchRequest:
    """One retrieval request: the question, the user's access and release filters, and tuning options."""
    query: str
    groups: list[str] = field(default_factory=lambda: ["all"])
    release: int | None = None               # release year, e.g. 2024 for R2024x
    category: str | None = None              # manifest category, e.g. 'installation' (None = all)
    mode: str = "hybrid"
    rerank: bool = True
    candidates: int = 30
    rerank_top: int | None = field(default_factory=lambda: get_settings().rerank_top)  # None = all
    min_context_score: float | None = None   # drop context chunks scoring below this (needs rerank)
    max_context_units: int = MAX_UNITS       # context size, from the answer model's budget (Phase 6)
    max_context_tokens: int = MAX_TOKENS
    user_id: str | None = None
    session_id: str | None = None


@dataclass
class SearchResult:
    """Ranked candidates, the assembled context, the trace id and per-stage timings."""
    candidates: list[Candidate]
    context: list[ContextUnit]
    trace_id: str
    timings_ms: dict[str, float]

    @property
    def top_score(self) -> float | None:
        """Best score of the top candidate (rerank score if reranked), or None if nothing was found."""
        return self.candidates[0].best_score if self.candidates else None


class Retriever:
    """Runs the retrieval stages (embed, search, rerank, assemble) against Qdrant and SQLite."""

    def __init__(self, conn: sqlite3.Connection, client: QdrantClient, collection: str,
                 embedder: Embedder, reranker: Reranker | None = None):
        """Keep the database, Qdrant client, collection name and models; no reranker = search order only."""
        self.conn = conn
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker

    def search(self, req: SearchRequest) -> SearchResult:
        """Retrieval only, traced as its own query."""
        with Tracer(self.conn, req.query, user_id=req.user_id, session_id=req.session_id) as trace:
            result = self.retrieve(req, trace)
            trace.set(route="retrieval")
        result.timings_ms = stage_timings(trace)
        return result

    def retrieve(self, req: SearchRequest, trace: Tracer, *, side: str | None = None) -> SearchResult:
        """Retrieval stages recorded into an existing trace (shared with answer generation).

        side: for comparisons, which side this search is for; recorded in every stage's data.
        """
        release_label = f"R{req.release}x" if req.release is not None else None
        query_filter = build_filter(req.groups, release=req.release, category=req.category)
        tag = {"side": side} if side else {}

        with trace.stage("embed", **tag):
            query = self.embedder.embed([req.query])[0]

        with trace.stage("search", mode=req.mode, limit=req.candidates, groups=req.groups,
                         release=release_label, category=req.category, **tag) as stage:
            candidates = search(self.client, self.collection, query, query_filter,
                                mode=req.mode, limit=req.candidates)
            stage["hits"] = [[c.chunk_id, round(c.score, 4)] for c in candidates]

        reranked = req.rerank and self.reranker is not None and bool(candidates)
        if reranked:
            with trace.stage("rerank", candidates=len(candidates), top=req.rerank_top,
                             max_length=getattr(self.reranker, "max_length", None), **tag) as stage:
                candidates = rerank(self.reranker, req.query, candidates, top=req.rerank_top)
                # every reranked candidate (top 20 by default), for the trace view (kb trace, "Behind the scenes")
                stage["top"] = [[c.chunk_id, round(c.rerank_score, 4)] for c in candidates
                                if c.rerank_score is not None]

        with trace.stage("assemble", min_score=req.min_context_score, **tag) as stage:
            context = assemble(self.conn, candidates, max_units=req.max_context_units,
                               max_tokens=req.max_context_tokens,
                               min_score=req.min_context_score if reranked else None)
            stage["units"] = [[u.section_id, u.kind, u.tokens] for u in context]
            same = {u.section_id: [s.section_id for s in u.same_text] for u in context if u.same_text}
            if same:
                stage["same_text"] = same          # near-duplicates cited with the unit, not sent to the LLM

        top_rerank = candidates[0].rerank_score if candidates else None
        trace.set(release_filter=release_label, top_rerank_score=top_rerank)
        return SearchResult(candidates, context, trace.trace_id, stage_timings(trace))


def stage_timings(trace: Tracer) -> dict[str, float]:
    """Duration of each recorded stage in milliseconds, by stage name; a stage that ran more than once
    (a comparison searches once per side) is the sum of its runs."""
    timings: dict[str, float] = {}
    for s in trace.stages:
        timings[s["stage"]] = timings.get(s["stage"], 0.0) + s["duration_ms"]
    return {name: round(ms, 1) for name, ms in timings.items()}
