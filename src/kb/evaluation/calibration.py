"""Gate calibration (Phase 7): which `KB_NOT_FOUND_SCORE` separates answerable from unanswerable questions?

The gate refuses without calling the LLM when the top rerank score is below the threshold. An
answer evaluation report (`kb eval-answers`) records, per golden question, the top score and what
happened; from it every candidate threshold can be replayed without running the models again:

    question with top score < t  →  refused by the gate at t (an answer it had is lost)
    question with top score ≥ t  →  the outcome recorded in the report

Exact for thresholds at or above the one the report was run with. Below it, questions the gate
refused at run time would now reach the LLM, with an outcome the report cannot know ("unmeasured").

Answers are re-scored against the current golden set (reviewed expected answers and must_include),
so a report from before the review still counts correctly. By default only reviewed questions count.

Real use adds evidence: a 👎 with the reason "should have answered" on a gate refusal says the
threshold is too high for that question; "should have refused" on an answer scoring below a
candidate threshold says the gate would have caught it (summarize_feedback).
"""

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

from kb.evaluation.answers import contains
from kb.llm.prompts import NOT_FOUND

DEFAULT_THRESHOLDS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HIGH_SCORE = 0.5        # refusals above this are generation problems, not the gate's


@dataclass
class Outcome:
    """One golden question in one report, re-scored against the current golden set."""
    qid: str
    qtype: str
    answerable: bool
    reviewed: bool
    top: float | None       # top rerank score (None: nothing found / not reranked)
    answered: bool
    refused_by: str | None  # gate | llm | None
    retried: bool           # answered only on the refusal retry
    good: bool              # answered with every must_include of the current golden set


@dataclass
class Report:
    """An answer evaluation report: where it came from, how it was run, and its outcomes."""
    path: Path
    model: str              # answer model profile (or the LLM model for reports from before Phase 6)
    threshold: float        # the gate threshold it was run with
    compare: bool | None
    refusal_retry: bool | None
    outcomes: list[Outcome] = field(default_factory=list)


@dataclass
class Row:
    """What one candidate threshold would have done to the selected questions."""
    threshold: float
    answers_lost: list[str]          # answerable questions answered in the report, refused by the gate at t
    good_lost: list[str]             # ... of which the answer had every key fact
    wrong_refusals: int              # answerable questions not answered at t (gate or LLM)
    unanswerable_gate: int           # unanswerable refused by the gate at t
    unanswerable_llm: int            # unanswerable reaching the LLM and refused by it
    let_through: list[str]           # unanswerable questions answered at t
    llm_calls_saved: int             # questions refused by the gate (no LLM call)
    retry_rescues: int               # answered only on the retry, and still reaching the LLM at t
    unmeasured: list[str]            # gate-refused at run time but not at t: outcome unknown


def load_report(path: Path, golden: dict[str, dict]) -> Report:
    """Read a `kb eval-answers` report and re-score its answers against the current golden set."""
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("config", {})
    report = Report(path=path, model=config.get("answer_model") or config.get("llm_model") or "unknown",
                    threshold=float(config.get("not_found_score", 0.1)), compare=config.get("compare"),
                    refusal_retry=config.get("refusal_retry"))
    for q in data.get("questions", []):
        gq = golden.get(q["qid"])
        if gq is None:
            continue
        answered = q["status"] == "answered"
        must = gq.get("must_include", [])
        report.outcomes.append(Outcome(
            qid=q["qid"], qtype=gq["type"], answerable=gq["type"] != "unanswerable",
            reviewed=gq.get("status") == "reviewed", top=q.get("top_score"), answered=answered,
            refused_by=q.get("refused_by"), retried=bool(q.get("retried")),
            good=answered and all(contains(q.get("answer", ""), s) for s in must)))
    return report


def latest_reports(eval_dir: Path, golden: dict[str, dict], *, reviewed_only: bool) -> list[Report]:
    """For each answer model, the most recent report covering every question in scope (a full run)."""
    needed = {qid for qid, q in golden.items() if q.get("status") == "reviewed" or not reviewed_only}
    chosen: dict[str, Report] = {}
    for path in sorted(glob.glob(str(eval_dir / "answers-*.json")), reverse=True):
        report = load_report(Path(path), golden)
        if report.model not in chosen and needed <= {o.qid for o in report.outcomes}:
            chosen[report.model] = report
    return list(chosen.values())


def in_scope(report: Report, reviewed_only: bool) -> list[Outcome]:
    """The report's outcomes that count: reviewed questions only, or all."""
    return [o for o in report.outcomes if o.reviewed or not reviewed_only]


def below(o: Outcome, t: float) -> bool:
    """Would the gate refuse this question at threshold t? (No score: the gate passes it.)"""
    return o.top is not None and o.top < t


def replay(outcomes: list[Outcome], t: float, run_threshold: float) -> Row:
    """What threshold t would have done, given outcomes recorded at run_threshold."""
    gate = [o for o in outcomes if below(o, t)]
    llm = [o for o in outcomes if not below(o, t)]
    unmeasured = [o.qid for o in llm if o.refused_by == "gate" and below(o, run_threshold)]
    reached = [o for o in llm if o.qid not in unmeasured]
    answerable_gate = [o for o in gate if o.answerable]
    return Row(
        threshold=t,
        answers_lost=[o.qid for o in answerable_gate if o.answered],
        good_lost=[o.qid for o in answerable_gate if o.good],
        wrong_refusals=len(answerable_gate) + sum(1 for o in reached if o.answerable and not o.answered),
        unanswerable_gate=sum(1 for o in gate if not o.answerable),
        unanswerable_llm=sum(1 for o in reached if not o.answerable and not o.answered),
        let_through=[o.qid for o in reached if not o.answerable and o.answered],
        llm_calls_saved=len(gate),
        retry_rescues=sum(1 for o in reached if o.retried and o.answered),
        unmeasured=unmeasured,
    )


@dataclass
class Recommendation:
    """The recommended threshold and the evidence behind it."""
    threshold: float | None
    lowest_answered: Outcome | None          # the answerable question with the lowest score that was answered
    highest_catchable: Outcome | None        # the highest-scoring unanswerable question below it
    reason: str


def recommend(outcomes: list[Outcome], current: float) -> Recommendation:
    """The midpoint between the lowest-scoring answered answerable question and the highest-scoring
    unanswerable question below it: the widest margin on both sides that loses no answer."""
    answered = sorted((o for o in outcomes if o.answerable and o.answered and o.top is not None), key=lambda o: o.top)
    if not answered:
        return Recommendation(None, None, None, "no answered answerable question with a score: nothing to calibrate on")
    lowest = answered[0]
    catchable = sorted((o for o in outcomes if not o.answerable and o.top is not None and o.top < lowest.top),
                       key=lambda o: o.top)
    if not catchable:
        return Recommendation(min(current, round(lowest.top / 2, 2)), lowest, None,
                              f"no unanswerable question scores below the lowest answered one ({lowest.qid} "
                              f"{lowest.top:.3f}): the gate cannot catch any without losing answers")
    high = catchable[-1]
    value = round((high.top + lowest.top) / 2, 2)
    return Recommendation(value, lowest, high,
                          f"midpoint of {high.qid} {high.top:.3f} (unanswerable) and {lowest.qid} {lowest.top:.3f} "
                          f"(lowest answered); margin {value - high.top:.3f} below, {lowest.top - value:.3f} above")


def high_score_refusals(outcomes: list[Outcome]) -> list[Outcome]:
    """Answerable questions the model refused although retrieval scored high: not fixable by the gate."""
    return sorted((o for o in outcomes if o.answerable and o.refused_by == "llm" and (o.top or 0) >= HIGH_SCORE),
                  key=lambda o: o.top)


def live_scores(conn) -> list[float | None]:
    """Top scores of real questions (chat and `kb ask`, not evaluation) that reached the gate."""
    rows = conn.execute("SELECT top_rerank_score FROM traces WHERE route IN ('answer', 'compare') AND cache_hit = 0 "
                        "AND COALESCE(user_id, '') != 'eval'").fetchall()
    return [r[0] for r in rows]


@dataclass
class Rating:
    """One rated answer: the rating, the 👎 reason, and what the gate saw."""
    rating: int
    reason: str | None
    top: float | None
    gate: str | None           # pass | low_score | no_context (None: not recorded, e.g. a cache hit)
    answered: bool             # the answer was not the "not found" sentence


def feedback_ratings(conn) -> list[Rating]:
    """Every rated answer, with its 👎 reason and the trace's top score, gate decision and outcome."""
    rows = conn.execute("""
        SELECT f.rating, f.reason, t.top_rerank_score, t.gate_decision, t.answer
        FROM feedback f JOIN traces t USING (trace_id)
    """).fetchall()
    return [Rating(r[0], r[1], r[2], r[3], bool(r[4]) and not r[4].startswith(NOT_FOUND)) for r in rows]


@dataclass
class FeedbackSummary:
    """What the 👎 reasons say about the threshold t."""
    ratings: int
    thumbs_down: int
    by_reason: dict[str, int]
    gate_refused_should_answer: list[float]    # "should have answered" that the gate refused (threshold too high)
    llm_refused_should_answer: int             # "should have answered" the model refused (generation, TD-23)
    should_refuse_catchable: list[float]       # "should have refused" answered with a score below t (would be caught)
    should_refuse_above: list[float]           # "should have refused" answered with a score at or above t


def summarize_feedback(ratings: list[Rating], t: float) -> FeedbackSummary:
    """Count 👎 reasons and place the two "not found" reasons against threshold t."""
    down = [r for r in ratings if r.rating < 0]
    by_reason: dict[str, int] = {}
    for r in down:
        key = r.reason or "no reason"
        by_reason[key] = by_reason.get(key, 0) + 1
    answer = [r for r in down if r.reason == "should_have_answered"]
    refuse = [r for r in down if r.reason == "should_have_refused" and r.answered and r.top is not None]
    return FeedbackSummary(
        ratings=len(ratings), thumbs_down=len(down), by_reason=by_reason,
        gate_refused_should_answer=sorted(r.top for r in answer if r.gate == "low_score" and r.top is not None),
        llm_refused_should_answer=sum(1 for r in answer if r.gate == "pass" and not r.answered),
        should_refuse_catchable=sorted(r.top for r in refuse if r.top < t),
        should_refuse_above=sorted(r.top for r in refuse if r.top >= t),
    )
