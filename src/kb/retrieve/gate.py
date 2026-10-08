"""Confidence gate v1: answer "not found" without calling the LLM when retrieval found nothing usable.

Evaluation showed the top rerank score separates clearly unrelated questions (< 0.1) from
answerable ones (lowest ~0.70), but not near misses whose context is on-topic without the
answer (0.76-0.90). Broad answerable questions can also score low (0.25 for a broad "what are
the different commands for …?" question, with the right section at rank 1). The gate therefore
only catches clear misses (threshold 0.1); the prompt makes the LLM refuse everything else.
"""

from dataclasses import dataclass

from kb.retrieve.pipeline import SearchResult

PASS = "pass"
NO_CONTEXT = "no_context"
LOW_SCORE = "low_score"


@dataclass
class GateDecision:
    """The gate's verdict, the top rerank score it was based on and the threshold used."""
    decision: str                 # pass | no_context | low_score
    top_score: float | None       # top rerank score (None when not reranked)
    threshold: float

    @property
    def passed(self) -> bool:
        """True when the question should go to the LLM."""
        return self.decision == PASS

    def explain(self) -> str:
        """One-line explanation of the decision for the user."""
        if self.decision == NO_CONTEXT:
            return "no matching documents (none found, or none you have access to)"
        if self.decision == LOW_SCORE:
            return f"best match scored {self.top_score:.3f}, below the {self.threshold:.2f} threshold"
        score = f"{self.top_score:.3f}" if self.top_score is not None else "not reranked"
        return f"passed (top rerank score {score})"


def gate(result: SearchResult, threshold: float) -> GateDecision:
    """No context -> no_context; reranked and top score < threshold -> low_score; else pass.

    Without reranking there is no calibrated score, so the question goes to the LLM.
    """
    top = result.candidates[0].rerank_score if result.candidates else None
    if not result.context:
        return GateDecision(NO_CONTEXT, top, threshold)
    if top is not None and top < threshold:
        return GateDecision(LOW_SCORE, top, threshold)
    return GateDecision(PASS, top, threshold)
