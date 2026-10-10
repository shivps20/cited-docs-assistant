import pytest

from kb.answer.pipeline import Answer, Source
from kb.evaluation.answers import (
    answer_commands,
    contains,
    run_answer_eval,
    unverified_commands,
)
from kb.llm.judge import (
    Claim,
    Verdict,
    distinctive_values,
    evidence_in_sources,
    judge_faithfulness,
    judge_messages,
    parse_verdict,
    split_answer,
)
from kb.llm.prompts import NOT_FOUND
from kb.llm.providers import Generation
from kb.retrieve.assemble import ContextUnit, SameText
from kb.retrieve.gate import GateDecision

GOLDEN = [
    {"id": "q1", "type": "lookup", "question": "ports?", "must_include": ["20300", "launcher.example.com"],
     "expected_answer": "Ports 20300 and 33200; see KB0433809 and https://launcher.example.com/info.",
     "sources": [{"doc_id": "launcher", "pages": [4, 6]}]},
    {"id": "q2", "type": "compare", "question": "mssql vs oracle?", "must_include": ["TomEE"],
     "expected_answer": "Both use TomEE.", "sources": [{"doc_id": "mssql", "pages": [22]},
                                                       {"doc_id": "oracle", "pages": [20]}]},
    {"id": "q3", "type": "unanswerable", "question": "nginx?", "must_include": [], "expected_answer": "",
     "sources": []},
    {"id": "q4", "type": "unanswerable", "question": "price?", "must_include": [], "expected_answer": "",
     "sources": []},
]


def unit(doc, pages):
    return ContextUnit(doc_id=doc, title=doc, section_id=f"{doc}#1", section_number="1", heading_path="1 X",
                       header=doc, page_start=pages[0], page_end=pages[1], text="t", kind="section", score=0.9,
                       tokens=10)


def source(n, doc, pages):
    return Source(n=n, doc_id=doc, title=doc, release="", section="1", heading="1 X", page_start=pages[0],
                  page_end=pages[1], line=f"[{n}] {doc}")


def answer(text, status="answered", sources=(), context=(), top=0.9, notices=(), refused_by_gate=False):
    gen = None if refused_by_gate else Generation(text, "ollama", "qwen", 1.0, output_tokens=50)
    return Answer(question="q", text=text, status=status, gate=GateDecision("pass", top, 0.1),
                  sources=list(sources), context=list(context), candidates=[], trace_id="t",
                  timings_ms={"embed": 10, "search": 20, "rerank": 3000, "generate": 5000},
                  generation=gen, notices=list(notices))


CANNED = {
    "ports?": answer("The Launcher listens on 20300 or 33200 [1]; see `KB0433809` and "
                     "https://launcher.example.com/info [1].", sources=[source(1, "launcher", (5, 5))],
                     context=[unit("launcher", (4, 6))]),
    "mssql vs oracle?": answer("The sources say both use **TomEE** [1].", sources=[source(1, "mssql", (22, 22)),
                                                                                 source(2, "other", (1, 1))],
                               context=[unit("mssql", (22, 22))],
                               notices=["Removed the contradictory 'could not find the answer' sentence ..."]),
    "nginx?": answer(NOT_FOUND, status="not_found", top=0.9),
    "price?": answer("A licence costs 100 EUR.", top=0.9),
}


def test_contains_ignores_case_markdown_and_whitespace():
    assert contains("Use **TomEE**  on\nport `8080`", "tomee on port 8080")
    assert not contains("Use Tomcat", "TomEE")


def test_run_answer_eval_scores_every_dimension():
    judged = []

    def judge_fn(a):
        judged.append(a.text)
        return Verdict(claims=[Claim("listens on 20300", True), Claim("listens on 9999", False)])

    report = run_answer_eval(lambda q: CANNED[q], GOLDEN, judge_fn)
    m = report["summary"]
    assert (m["answerable"], m["unanswerable"]) == (2, 2)
    assert m["answered"] == 1.0 and m["wrong_refusals"] == []
    assert m["must_include"] == pytest.approx(1.0) and m["must_include_all"] == 1.0
    assert (m["article_numbers"], m["urls"]) == (1.0, 1.0)
    assert m["citation_precision"] == pytest.approx(2 / 3)      # 'other' is not a golden source
    assert m["citation_doc_recall"] == pytest.approx(2 / 3)     # oracle never cited
    assert m["context_hit"] == 1.0
    assert m["faithfulness"] == 0.5 and m["fully_faithful"] == 0.0
    assert m["meta_talk"] == 0.5 and m["dropped_not_found"] == 0.5
    assert m["refused"] == 0.5 and m["wrong_answers"] == ["q4"] and m["refused_by_llm"] == 1
    assert len(judged) == 3                                     # refusals are not judged
    q1 = next(q for q in report["questions"] if q["qid"] == "q1")
    assert q1["unsupported"] == ["listens on 9999"] and q1["retrieval_ms"] == 3030


def test_same_text_copy_counts_as_cited_and_in_context():
    kept = unit("oracle", (20, 20))
    kept.same_text = [SameText("mssql", "mssql", "mssql#1", "1", 22, 22)]
    cited = source(1, "oracle", (20, 20))
    cited.same_text = [{"doc_id": "mssql", "title": "mssql", "section": "1", "page_start": 22, "page_end": 22,
                        "release": ""}]
    q2 = {**GOLDEN[1], "sources": [{"doc_id": "mssql", "pages": [22]}]}     # golden names only the MSSQL guide
    report = run_answer_eval(lambda q: answer("Both use TomEE [1].", sources=[cited], context=[kept]), [q2])
    r = report["questions"][0]
    assert (r["cited_correct"], r["golden_docs_cited"], r["context_hit"]) == (1, 1, True)


def test_parse_verdict_reads_claims_and_reports_bad_json():
    v = parse_verdict('{"claims": [{"claim": "depth is 6", "supported": true, "source": 1}, '
                      '{"claim": "depth max 10", "supported": false, "source": null}]}')
    assert v.faithfulness == 0.5 and v.unsupported == ["depth max 10"] and v.claims[0].source == 1
    bad = parse_verdict("Sure! Here are the claims")
    assert bad.error and bad.faithfulness is None


SOURCE_TEXT = ("| indexdepth | Accepted values: integer greater than 1 Default: 6 |\n"
               "- Ensure that the SSO HTTP-POST binding is selected.\n- Apply the configuration.")


def test_evidence_must_occur_word_for_word():
    assert evidence_in_sources("indexdepth | Accepted values: integer greater than 1 Default: 6", SOURCE_TEXT)
    assert evidence_in_sources("Ensure that the SSO HTTP-POST binding ... Apply the configuration", SOURCE_TEXT)
    assert not evidence_in_sources("Ensure that the SSO HTTP-Redirect binding is selected.", SOURCE_TEXT)
    assert not evidence_in_sources("", SOURCE_TEXT)


def test_quotes_decide_support_over_the_judges_label():
    reply = ('{"claims": ['
             '{"claim": "default depth 6", "evidence": "Default: 6", "supported": false},'
             '{"claim": "HTTP-Redirect binding", "evidence": "SSO HTTP-Redirect binding", "supported": true},'
             '{"claim": "restart afterwards", "evidence": "", "supported": false}]}')
    v = parse_verdict(reply, SOURCE_TEXT)
    assert [c.supported for c in v.claims] == [True, False, False]
    assert v.faithfulness == pytest.approx(1 / 3)


def test_lead_in_lines_are_not_claims():
    reply = ('{"claims": [{"claim": "To set these parameters, use the following ACME commands:", '
             '"evidence": "", "supported": false}, {"claim": "default depth 6", "evidence": "Default: 6", '
             '"supported": true}]}')
    v = parse_verdict(reply, SOURCE_TEXT)
    assert [c.text for c in v.claims] == ["default depth 6"] and v.faithfulness == 1.0


def test_judge_sees_sources_and_answer_without_markers():
    messages = judge_messages("Depth is 6 [1].", [unit("index", (20, 22))])
    assert "SOURCES:" in messages[1]["content"] and "Depth is 6 ." in messages[1]["content"]
    assert '"supported"' in messages[0]["content"]


ORA_TEXT = "Errors: ORA-01157 cannot identify data file; ORA-01110 data file 4. Edit config/listener.ora."


def test_values_support_a_claim_without_a_usable_quote():
    reply = ('{"claims": ['
             '{"claim": "The errors ORA-01157 and ORA-01110 appear; edit config/listener.ora", "evidence": ""},'
             '{"claim": "Error ORA-09999 appears", "evidence": ""},'
             '{"claim": "Restart the service", "evidence": ""}]}')
    v = parse_verdict(reply, ORA_TEXT)
    assert [c.supported for c in v.claims] == [True, False, False]
    assert [c.by_values for c in v.claims] == [True, False, False]
    assert distinctive_values("port 9040 and `srvctl start db`") == ["9040", "srvctl start db"]


def test_long_answers_are_split_between_paragraphs_with_code_blocks_whole():
    long = "a" * 900 + "\n\n```\nline 1\n\nline 2\n```\n\n" + "b" * 900
    parts = split_answer(long, limit=1000)
    assert len(parts) == 2 and parts[0].endswith("```") and "line 1\n\nline 2" in parts[0]
    assert parts[1] == "b" * 900 and split_answer("short") == ["short"]


class PartJudge:
    """A judge that returns one supported claim per part it is asked about."""

    def __init__(self):
        """No calls yet."""
        self.calls = 0

    def generate(self, messages, *, on_token=None, json_format=False, json_schema=None):
        """One claim quoting the context."""
        self.calls += 1
        reply = f'{{"claims": [{{"claim": "part {self.calls}", "evidence": "t", "supported": true}}]}}'
        return Generation(reply, "ollama", "qwen", 0.5)


def test_judge_merges_the_claims_of_every_part():
    judge = PartJudge()
    v = judge_faithfulness(judge, "x" * 1000 + "\n\n" + "y" * 1000, [unit("doc", (1, 1))])
    assert judge.calls == 2 and [c.text for c in v.claims] == ["part 1", "part 2"]
    assert v.faithfulness == 1.0 and v.seconds == 1.0 and v.error is None


def test_commands_must_occur_word_for_word_in_the_context():
    text = "Run:\n```bash\nsrvctl start db -d X\nlsnrctl status\n```\nThen set `octreedepth 6`; see `ls`."
    assert answer_commands(text) == ["srvctl start db -d X", "lsnrctl status", "octreedepth 6"]
    total, missing = unverified_commands(text, "srvctl  start db -d X ... `octreedepth 5` ... lsnrctl status")
    assert total == 3 and missing == ["octreedepth 6"]
    # formatting is not drift: prompt spacing, punctuation and line breaks are ignored
    wrapped = "Run:\n```\nMQL > tidy vault vplm;\nFILEGROUP [I1_DATA] (NAME=[I1_DATA],\n```"
    assert unverified_commands(wrapped, "MQL> tidy vault vplm ;  FILEGROUP [I1_DATA]\n(NAME = [I1_DATA],") == (2, [])


def test_command_check_is_summarised():
    context = [unit("launcher", (4, 6))]
    context[0].text = "Run `kb restart --all` to apply."
    a = answer("Run `kb restart --all` [1], then `kb purge --hard` [1].", sources=[source(1, "launcher", (5, 5))],
               context=context)
    report = run_answer_eval(lambda q: a, GOLDEN[:1])
    assert report["summary"]["commands_verified"] == 0.5 and report["summary"]["commands_unverified"] == 1
    assert report["questions"][0]["commands_unverified"] == ["kb purge --hard"]
