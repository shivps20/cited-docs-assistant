"""Qdrant search: access/release/latest filters and dense, sparse or hybrid (RRF) retrieval.

Filters run inside Qdrant (in every prefetch), so chunks a user may not see are never
returned, not merely hidden afterwards.
"""

from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from kb.store.embed import Embedding
from kb.store.vectorstore import DENSE, SPARSE

MODES = ("hybrid", "dense", "sparse")
PUBLIC_GROUP = "all"


@dataclass
class Candidate:
    """A chunk returned by search, with its retrieval score and (after reranking) rerank score."""
    point_id: str
    chunk_id: str
    doc_id: str
    section_id: str
    section_number: str
    chunk_index: int
    title: str
    header: str
    text: str
    page_start: int
    page_end: int
    score: float                         # retrieval score (RRF, cosine or sparse dot product)
    rerank_score: float | None = None
    payload: dict = field(default_factory=dict, repr=False)

    @property
    def best_score(self) -> float:
        """The rerank score if the chunk was reranked, else the retrieval score."""
        return self.rerank_score if self.rerank_score is not None else self.score


def build_filter(groups: list[str], *, release: int | None = None, latest_only: bool = True,
                 external_only: bool = False, category: str | None = None) -> models.Filter:
    """Access (user groups or public), optional release, latest revision, optional external_ok and category."""
    allowed = sorted(set(groups) | {PUBLIC_GROUP})
    must: list[models.Condition] = [
        models.FieldCondition(key="allowed_groups", match=models.MatchAny(any=allowed)),
    ]
    if latest_only:
        must.append(models.FieldCondition(key="is_latest", match=models.MatchValue(value=True)))
    if release is not None:
        must.append(models.FieldCondition(key="release_min", range=models.Range(lte=release)))
        must.append(models.FieldCondition(key="release_max", range=models.Range(gte=release)))
    if external_only:
        must.append(models.FieldCondition(key="external_ok", match=models.MatchValue(value=True)))
    if category is not None:
        must.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    return models.Filter(must=must)


def _candidate(point: models.ScoredPoint) -> Candidate:
    """Build a Candidate from a Qdrant point and its payload."""
    p = point.payload or {}
    return Candidate(
        point_id=str(point.id), chunk_id=p.get("chunk_id", ""), doc_id=p.get("doc_id", ""),
        section_id=p.get("section_id", ""), section_number=p.get("section_number", ""),
        chunk_index=p.get("chunk_index", 0), title=p.get("title", ""), header=p.get("header", ""),
        text=p.get("text", ""), page_start=p.get("page_start", 0), page_end=p.get("page_end", 0),
        score=point.score, payload=p,
    )


def search(client: QdrantClient, collection: str, query: Embedding, query_filter: models.Filter, *,
           mode: str = "hybrid", limit: int = 30, prefetch_limit: int = 50) -> list[Candidate]:
    """Top `limit` chunks for the query embedding, best first."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    dense = query.dense
    sparse = models.SparseVector(indices=query.sparse_indices, values=query.sparse_values)

    if mode == "dense":
        response = client.query_points(collection, query=dense, using=DENSE, query_filter=query_filter,
                                       limit=limit, with_payload=True)
    elif mode == "sparse":
        response = client.query_points(collection, query=sparse, using=SPARSE, query_filter=query_filter,
                                       limit=limit, with_payload=True)
    else:
        response = client.query_points(
            collection,
            prefetch=[
                models.Prefetch(query=dense, using=DENSE, filter=query_filter, limit=prefetch_limit),
                models.Prefetch(query=sparse, using=SPARSE, filter=query_filter, limit=prefetch_limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, with_payload=True,
        )
    return [_candidate(p) for p in response.points]
