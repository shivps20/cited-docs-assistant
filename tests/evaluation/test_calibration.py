import json
import tempfile
from pathlib import Path

from kb.evaluation.calibration import (
    Rating,
    high_score_refusals,
    in_scope,
    latest_reports,
    load_report,
    recommend,
    replay,
    summarize_feedback,
)

GOLDEN = {
    "q1": {"id": "q1", "type": "lookup", "must_include": ["20300"], "status": "reviewed"},
    "q2": {"id": "q2", "type": "howto", "must_include": ["restart"], "status": "reviewed"},
    "q3": {"id": "q3", "type": "lookup", "must_include": ["9040"], "status": "reviewed"},
    "q4": {"id": "q4", "type": "unanswerable", "must_include": [], "status": "reviewed"},
    "q5": {"id": "q5", "type": "unanswerable", "must_include": [], "status": "reviewed"},
    "q6": {"id": "q6", "type": "unanswerable", "must_include": [], "status": "reviewed"},
    "q7": {"id": "q7", "type": "lookup", "must_include": ["x"], "status": "draft"},
}


def question(qid, top, status="answered", refused_by=None, answer="", retried=False):
    """One question entry as `kb eval-answers` writes it."""
    return {"qid": qid, "top_score": top, "status": status, "refused_by": refused_by, "answer": answer,
            "retried": retried}


QUESTIONS = [
    question("q1", 0.15, answer="Port 20300 [1].", retried=True),           # lowest answered, on the retry
    question("q2", 0.70, answer="Stop the service [1]."),                   # answered, key fact missing
    question("q3", 0.90, "not_found", "llm"),                               # refused although retrieval scored high
    question("q4", 0.05, "not_found", "gate"),                              # gate-refused at run time
    question("q5", 0.40, "not_found", "llm"),
    question("q6", 0.95, "answered", answer="It costs 100 EUR."),           # unanswerable, let through
    question("q7", 0.20, answer="x"),                                       # not reviewed
]


def write_report(tmp_path, name, model="local-qwen", questions=QUESTIONS, threshold=0.1):
    """An answer evaluation report file."""
    path = tmp_path / name
    path.write_text(json.dumps({"config": {"answer_model": model, "not_found_score": threshold, "compare": True,
                                           "refusal_retry": True}, "questions": questions}), encoding="utf-8")
    return path


def load_report_from(questions):
    """A Report built from question entries (written to a temporary file and read back)."""
    with tempfile.TemporaryDirectory() as d:
        return load_report(write_report(Path(d), "answers.json", questions=questions), GOLDEN)


def test_answers_are_rescored_against_the_current_golden_set(tmp_path):
    report = load_report(write_report(tmp_path, "answers-1.json"), GOLDEN)
    by_id = {o.qid: o for o in report.outcomes}
    assert by_id["q1"].good and not by_id["q2"].good and by_id["q2"].answered
    assert (report.model, report.threshold, report.refusal_retry) == ("local-qwen", 0.1, True)
    assert [o.qid for o in in_scope(report, True)] == ["q1", "q2", "q3", "q4", "q5", "q6"]
    assert len(in_scope(report, False)) == 7


def test_replay_above_the_run_threshold_is_exact():
    outcomes = in_scope(load_report_from(QUESTIONS), True)
    row = replay(outcomes, 0.5, 0.1)
    assert (row.answers_lost, row.good_lost) == (["q1"], ["q1"])
    assert row.wrong_refusals == 2                        # q1 by the gate, q3 by the model
    assert (row.unanswerable_gate, row.unanswerable_llm, row.let_through) == (2, 0, ["q6"])
    assert (row.llm_calls_saved, row.retry_rescues, row.unmeasured) == (3, 0, [])
    keep = replay(outcomes, 0.1, 0.1)
    assert (keep.answers_lost, keep.retry_rescues, keep.unanswerable_gate) == ([], 1, 1)


def test_below_the_run_threshold_gate_refusals_are_unmeasured():
    row = replay(in_scope(load_report_from(QUESTIONS), True), 0.0, 0.1)
    assert row.unmeasured == ["q4"] and row.unanswerable_gate == 0 and row.unanswerable_llm == 1


def test_recommendation_is_the_midpoint_that_loses_no_answer():
    rec = recommend(in_scope(load_report_from(QUESTIONS), True), current=0.1)
    assert rec.threshold == 0.1 and rec.lowest_answered.qid == "q1" and rec.highest_catchable.qid == "q4"
    assert "margin 0.050 below, 0.050 above" in rec.reason
    nothing_below = [question("q1", 0.6, answer="20300"), question("q5", 0.8, "not_found", "llm")]
    rec = recommend(in_scope(load_report_from(nothing_below), True), current=0.1)
    assert rec.threshold == 0.1 and rec.highest_catchable is None and "cannot catch" in rec.reason


def test_high_score_refusals_are_listed_as_generation_problems():
    assert [o.qid for o in high_score_refusals(in_scope(load_report_from(QUESTIONS), True))] == ["q3"]


def test_latest_full_report_per_model(tmp_path):
    write_report(tmp_path, "answers-20261001-000000.json")
    write_report(tmp_path, "answers-20261002-000000.json", questions=QUESTIONS[:2])      # partial run: skipped
    write_report(tmp_path, "answers-20261003-000000.json", model="claude-opus")
    reports = latest_reports(tmp_path, GOLDEN, reviewed_only=True)
    assert sorted((r.model, r.path.name) for r in reports) == [
        ("claude-opus", "answers-20261003-000000.json"), ("local-qwen", "answers-20261001-000000.json")]


def test_a_report_from_before_phase_6_is_not_a_model_of_its_own(tmp_path):
    old = tmp_path / "answers-20260901-000000.json"            # no answer_model, only the LLM
    old.write_text(json.dumps({"config": {"llm_model": "qwen2.5:7b-instruct", "not_found_score": 0.1},
                               "questions": QUESTIONS}), encoding="utf-8")
    new = tmp_path / "answers-20261001-000000.json"
    new.write_text(json.dumps({"config": {"answer_model": "local-qwen", "llm_model": "qwen2.5:7b-instruct",
                                          "not_found_score": 0.1}, "questions": QUESTIONS}), encoding="utf-8")
    assert [r.path.name for r in latest_reports(tmp_path, GOLDEN, reviewed_only=True)] == [new.name]



def test_feedback_reasons_are_placed_against_the_threshold():
    ratings = [
        Rating(1, None, 0.9, "pass", True),
        Rating(-1, "should_have_answered", 0.08, "low_score", False),     # the gate refused: threshold too high
        Rating(-1, "should_have_answered", 0.85, "pass", False),          # the model refused (TD-23)
        Rating(-1, "should_have_refused", 0.07, "pass", True),            # answered below 0.1: a gate would catch it
        Rating(-1, "should_have_refused", 0.9, "pass", True),             # only the model can refuse it
        Rating(-1, "incomplete", 0.9, "pass", True),
        Rating(-1, None, 0.5, "pass", True),
    ]
    fb = summarize_feedback(ratings, 0.1)
    assert (fb.ratings, fb.thumbs_down) == (7, 6)
    assert fb.by_reason == {"should_have_answered": 2, "should_have_refused": 2, "incomplete": 1, "no reason": 1}
    assert fb.gate_refused_should_answer == [0.08] and fb.llm_refused_should_answer == 1
    assert fb.should_refuse_catchable == [0.07] and fb.should_refuse_above == [0.9]
