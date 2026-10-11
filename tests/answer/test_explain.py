import json
from types import SimpleNamespace

import pytest

from kb.answer import explain
from kb.answer.records import messages_hash, unit_from_record, unit_record
from kb.core.db import connect, migrate
from kb.core.tracing import Tracer
from kb.llm.prompts import build_messages
from kb.retrieve.assemble import ContextUnit, SameText

TEXT = "Run MQL> set system searchindex depth 5; to set the depth. The default port is 20300."


def unit(n="2.1", external_ok=True):
    """A context unit of a fictional guide."""
    return ContextUnit(doc_id="guide", title="Guide", section_id=f"guide#{n}", section_number=n,
                       heading_path=f"{n} Index depth", header=f"Guide > {n} Index depth", page_start=4, page_end=5,
                       text=TEXT, kind="section", score=0.9, tokens=40, release="R2025x", external_ok=external_ok)


@pytest.fixture
def conn(tmp_path):
    """A migrated database."""
    path = tmp_path / "kb.db"
    migrate(connect(path, check_schema=False))
    c = connect(path)
    yield c
    c.close()


def write_trace(conn, answer="Set it with `MQL> set system searchindex depth 5;` [1]. Use `MQL> set depth 6;` [1].",
                units=None, user="guest"):
    """A trace as the answer pipeline writes it: search, rerank, assemble, gate, generate (with context), cite."""
    units = units or [unit()]
    messages = build_messages("What is the index depth?", units, ())
    with Tracer(conn, "What is the index depth?", user_id=user) as t:
        t.add_stage("search", 30.0, mode="hybrid", limit=30, groups=["all"], release=None, category=None,
                    hits=[["other#1#0", 0.6], ["guide#2.1#0", 0.5], ["far#9#0", 0.1]])
        t.add_stage("rerank", 5000.0, candidates=3, top=[["guide#2.1#0", 0.91], ["other#1#0", 0.40]])
        t.add_stage("assemble", 8.0, units=[["guide#2.1", "section", 40]])
        t.add_stage("gate", 0.01, threshold=0.1, decision="pass", top_score=0.91)
        t.add_stage("generate", 9000.0, profile="local-qwen", provider="local-qwen", model="qwen", prompt_tokens=900,
                    output_tokens=40, tokens_per_s=22.0, question="What is the index depth?", sides=[],
                    context=[unit_record(u) for u in units], prompt_sha=messages_hash(messages), raw_output=answer)
        t.add_stage("cite", 0.1, cited=[1], invalid=[], refused=False)
        t.set(route="answer", gate_decision="pass", top_rerank_score=0.91, llm_provider="local-qwen", llm_model="qwen",
              answer=answer)
    return t.trace_id


def test_unit_records_round_trip_and_rebuild_the_same_prompt():
    """A stored unit comes back equal for prompting: the rebuilt messages hash the same."""
    u = unit()
    u.same_text = [SameText("copy", "Copy", "copy#2.1", "2.1", 4, 4, "R2025x")]
    back = unit_from_record(json.loads(json.dumps(unit_record(u))))
    assert (back.section_id, back.text, back.release, back.doc_type) == (u.section_id, u.text, u.release, "pdf")
    assert messages_hash(build_messages("q", [back], ())) == messages_hash(build_messages("q", [u], ()))


def test_explain_shows_scores_context_prompt_and_command_check(conn):
    """Reranked candidates first with both scores; the context, the exact prompt, and the unverified command."""
    view = explain.explain_trace(conn, write_trace(conn))
    assert [(c["chunk_id"], c["rerank_rank"], c["retrieval_rank"], c["in_context"]) for c in view["candidates"]] == [
        ("guide#2.1#0", 1, 2, True), ("other#1#0", 2, 1, False), ("far#9#0", None, 3, False)]
    assert view["gate"] == {"decision": "pass", "top_score": 0.91, "threshold": 0.1}
    assert [u["section_id"] for u in view["context"]] == ["guide#2.1"]
    gen = view["generations"][0]
    assert gen["prompt_identical"] is True and gen["messages"][1]["content"].count(TEXT) == 1
    assert view["checks"]["commands"] == {"total": 2, "unverified": ["MQL> set depth 6;"]}
    assert view["faithfulness"] is None
    assert explain.explain_trace(conn, "nope") is None


class FakeJudge:
    """A judge model that supports the first claim with a real quote and rejects the second."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages, json_format=False, **kwargs):
        """One verdict reply."""
        self.calls += 1
        reply = {"claims": [{"claim": "depth is set with the MQL command", "evidence": "set system searchindex depth 5",
                             "supported": True, "source": 1},
                            {"claim": "depth 6", "evidence": "", "supported": False, "source": None}]}
        return SimpleNamespace(text=json.dumps(reply), seconds=1.0)


def registry(judge, *, external_judge=False):
    """The parts of a model registry the faithfulness check uses."""
    role = SimpleNamespace(name="cloud" if external_judge else "local-qwen", external=external_judge)
    return SimpleNamespace(catalogue=SimpleNamespace(for_role=lambda r: role, fallback="local-qwen"),
                           ready=lambda name: True, provider=lambda name: judge,
                           for_role=lambda r, max_output_tokens=None: judge)


def test_faithfulness_is_judged_once_stored_with_the_trace_and_shown(conn):
    """The verdict becomes a 'judge' stage; asking again returns it without another LLM call unless forced."""
    trace_id, judge = write_trace(conn), FakeJudge()
    result = explain.check_faithfulness(conn, registry(judge), trace_id)
    assert result["faithfulness"] == 0.5 and result["claims"][0]["evidence_found"] is True
    assert explain.check_faithfulness(conn, registry(judge), trace_id)["faithfulness"] == 0.5
    assert judge.calls == 1
    explain.check_faithfulness(conn, registry(judge), trace_id, force=True)
    assert judge.calls == 2
    assert explain.explain_trace(conn, trace_id)["faithfulness"]["model"] == "local-qwen"


def test_internal_context_is_never_judged_by_an_external_model(conn):
    """The privacy policy holds for the judge: an internal source makes it judge locally, with a note."""
    trace_id = write_trace(conn, units=[unit(external_ok=False)])
    result = explain.check_faithfulness(conn, registry(FakeJudge(), external_judge=True), trace_id)
    assert result["model"] == "local-qwen" and "may not be sent to cloud" in result["note"]


def test_a_refused_answer_has_nothing_to_judge(conn):
    """No answer text: a clear error instead of a judge call."""
    with Tracer(conn, "unknown", user_id="guest") as t:
        t.add_stage("gate", 0.01, threshold=0.1, decision="low_score", top_score=0.02)
        t.set(gate_decision="low_score")
    with pytest.raises(ValueError, match="no stored context"):
        explain.check_faithfulness(conn, registry(FakeJudge()), t.trace_id)
