"""Cross-encoder reranking with bge-reranker-v2-m3 (CPU by default, keeps the GPU for the LLM)."""

from typing import Protocol

from kb.core.config import get_settings
from kb.retrieve.search import Candidate


class Reranker(Protocol):
    """Anything that scores (query, passage) pairs: bge-reranker in production, a fake in tests."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance score for each passage against the query, in the same order."""
        ...


class BgeReranker:
    """Scores (query, passage) pairs; normalized to 0..1 so scores are comparable across queries."""

    def __init__(self, device: str = "cpu", batch_size: int = 4, max_length: int = 512):
        """Load bge-reranker-v2-m3 on `device` (CPU by default) and warm it up.

        batch_size 4: FlagEmbedding sorts pairs by length, so small batches pad little; on the
        CPU, 20 passages took ~10-13 s in batches of 4 against ~19-26 s in a single batch.
        """
        from FlagEmbedding import FlagReranker

        from kb.core.progress import silence_progress_bars

        silence_progress_bars()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = FlagReranker(str(get_settings().rerank_model_path), use_fp16=device != "cpu", devices=device)
        self.score("warm-up", ["warm-up"])

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Normalised 0-1 relevance scores for the passages."""
        if not passages:
            return []
        scores = self.model.compute_score([(query, p) for p in passages], batch_size=self.batch_size,
                                          max_length=self.max_length, normalize=True)
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]


def rerank(reranker: Reranker, query: str, candidates: list[Candidate], top: int | None = None) -> list[Candidate]:
    """Rerank the first `top` candidates (all if None) on header + text and sort them best first.

    Candidates beyond `top` keep their retrieval order after the reranked ones (rerank_score None);
    reranking cost grows linearly with the number of candidates.
    """
    head = candidates if top is None else candidates[:top]
    tail = [] if top is None else candidates[top:]
    scores = reranker.score(query, [f"{c.header}\n\n{c.text}" for c in head])
    for candidate, score in zip(head, scores, strict=True):
        candidate.rerank_score = score
    return sorted(head, key=lambda c: c.rerank_score, reverse=True) + tail
