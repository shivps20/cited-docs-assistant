import json

import pytest

from kb.agent.compare import Side, decompose, parse_sides, retrieve_sides, side_release
from kb.agent.route import route_question
from kb.agent.tools import KBTools
from kb.answer.pipeline import Answerer
from kb.core.db import connect, migrate
from kb.core.tracing import Tracer
from kb.llm.providers import Generation
from kb.retrieve.assemble import ContextUnit, demote_front_matter, merge_contexts
from kb.retrieve.pipeline import SearchRequest, SearchResult
from kb.retrieve.search import Candidate

# ---------------------------------------------------------------------------------------- routing


@pytest.mark.parametrize("question, kind", [
    ("How does creating the database differ between the MSSQL and Oracle guides?", "compare"),
    ("What is the difference between RPN and Action Priority?", "compare"),
    ("Certificate requirements for SAML on Cloud versus HTTPS on premises?", "compare"),
    ("Compare the MSSQL vs Oracle setup", "compare"),
    ("Is the indexing server still needed in R2024x and in R2026x?", "compare"),           # two releases
    ("Should services use a single FQDN or one FQDN per service?", "compare"),             # a choice
    ("What are the different components for installation?", "answer"),                    # 'different' = list
    ("What delta between ClientBeginRequest and ClientDoneRequest indicates a problem?", "answer"),
    ("Which port does CoreServer use on R2026x?", "answer"),
])
def test_route_question(question, kind):
    assert route_question(question).kind == kind


# ---------------------------------------------------------------------------------- decomposition

def test_parse_sides_accepts_two_or_three_distinct_sides_only():
    good = '{"sides": [{"label": "Oracle", "query": "How is the database created on Oracle?"},' \
           ' {"label": "MSSQL", "query": "How is the database created on MSSQL?"}]}'
    assert [s.label for s in parse_sides(good)] == ["Oracle", "MSSQL"]
    assert parse_sides("not json") == []
    assert parse_sides('{"sides": [{"label": "Oracle", "query": "q"}]}') == []                      # one side
    assert parse_sides('{"sides": [{"label": "A", "query": "same"}, {"label": "B", "query": "Same"}]}') == []
    assert parse_sides('{"sides": [{"label": "A", "query": "' + "word " * 50 + '"}, {"label": "B", "query": "q"}]}') == []
    not_split = '{"sides": [{"label": "MSSQL", "query": "How does it differ between MSSQL and Oracle?"},' \
                ' {"label": "Oracle", "query": "How is it done on Oracle?"}]}'
    assert parse_sides(not_split) == []                         # a side still names the other side


class FakePlanner:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate(self, messages, *, on_token=None, json_format=False):
        self.calls.append(json_format)
        return Generation(self.reply, "ollama", "qwen", 0.5)


def test_decompose_uses_json_mode_and_falls_back_on_a_bad_reply():
    planner = FakePlanner('{"sides": [{"label": "A", "query": "about A?"}, {"label": "B", "query": "about B?"}]}')
    assert [s.query for s in decompose(planner, "A vs B?").sides] == ["about A?", "about B?"] and planner.calls == [True]
    bad = decompose(FakePlanner("{}"), "A vs B?")
    assert bad.sides == [] and "one search" in bad.reason
    assert decompose(None, "A vs B?").sides == []


def test_side_release_only_uses_releases_the_user_named():
    q = "How does the guidance differ between R2024x and R2026x?"
    assert side_release(q, Side("R2024x", "Indexing server guidance in R2024x"), None) == 2024
    assert side_release(q, Side("R2026x", "Indexing server guidance in R2026x"), None) == 2026
    assert side_release(q, Side("other", "guidance in R2025x"), None) is None          # not named by the user
    assert side_release("Oracle vs MSSQL?", Side("Oracle", "on R2026x"), 2026) == 2026  # the request's release


# ------------------------------------------------------------------------------ assembly helpers

def unit(sid, text=None, tokens=100, doc=None):
    doc = doc or sid.split("#")[0]
    return ContextUnit(doc_id=doc, title=doc.upper(), section_id=sid, section_number=sid.split("#")[1],
                       heading_path=f"{sid.split('#')[1]} X", header="h", page_start=1, page_end=1,
                       text=text or f"unique text of {sid} " * 5, kind="section", score=0.9, tokens=tokens)


def test_merge_contexts_reserves_slots_per_side_and_merges_copies():
    shared = "the same paragraph about certificates and their common names " * 4
    a = [unit("a#1"), unit("a#2"), unit("a#3"), unit("a#4"), unit("ora#5", text=shared)]
    b = [unit("sql#5", text=shared), unit("b#1"), unit("a#1"), unit("b#2")]
    merged = merge_contexts([("A", a), ("B", b)], per_side=3, max_tokens=10_000)
    assert [(u.section_id, u.side) for u in merged] == [("a#1", "A"), ("sql#5", "B"), ("a#2", "A"), ("b#1", "B"),
                                                        ("a#3", "A"), ("b#2", "B")]
    assert merge_contexts([("A", [unit("x#1", text=shared)]), ("B", [unit("y#1", text=shared + " extra")])])[0] \
        .same_text[0].doc_id == "y"                                                   # near copy cited, not sent
    assert len(merge_contexts([("A", a), ("B", b)], per_side=3, max_tokens=250)) == 2   # token budget


def cand(chunk_id, rerank, header):
    doc, num, _ = chunk_id.split("#")
    return Candidate(point_id=chunk_id, chunk_id=chunk_id, doc_id=doc, section_id=f"{doc}#{num}", section_number=num,
                     chunk_index=0, title=doc, header=header, text="t", page_start=1, page_end=1, score=0.03,
                     rerank_score=rerank)


def test_executive_summaries_are_demoted_but_introductions_are_not():
    ranked = [cand("d#0#0", 0.99, "Guide [R2026x] > Executive Summary"), cand("d#1#0", 0.97, "Guide > 1 Introduction"),
              cand("d#4#0", 0.90, "Guide > 4 Install > 4.1 Steps"), cand("d#9#0", None, "Guide > 9 Tail")]
    order = [c.chunk_id for c in demote_front_matter(ranked)]
    assert order == ["d#1#0", "d#4#0", "d#0#0", "d#9#0"]      # 0.99 × 0.8 = 0.79 < 0.90; tail keeps its place


# ------------------------------------------------------------------------------------------ tools

@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "kb.db"
    migrate(connect(db, check_schema=False))
    c = connect(db)
    with c:
        for doc, groups, latest in [("public", ["all"], 1), ("secret", ["internal"], 1), ("old", ["all"], 0)]:
            c.execute("INSERT INTO documents (doc_id, source_path, file_hash, title, allowed_groups, is_latest) "
                      "VALUES (?, ?, 'h', ?, ?, ?)", (doc, doc, doc.title(), json.dumps(groups), latest))
            c.execute("INSERT INTO sections (section_id, doc_id, heading_path, ordinal, page_start, page_end, text, "
                      "token_count) VALUES (?, ?, '1 Intro', 0, 1, 2, 'text', 10)", (f"{doc}#1", doc))
    yield c
    c.close()


class RecordingRetriever:
    """Returns a canned result per query and records every request it gets."""

    def __init__(self, results):
        self.results, self.requests = results, []

    def retrieve(self, req, trace, *, side=None):
        self.requests.append((req, side))
        with trace.stage("search", side=side):
            pass
        return self.results[req.query]


def test_get_section_checks_the_users_access(conn):
    with Tracer(conn, "q") as trace:
        public = KBTools(conn, RecordingRetriever({}), SearchRequest("q", groups=["all"]), trace)
        internal = KBTools(conn, RecordingRetriever({}), SearchRequest("q", groups=["internal"]), trace)
        assert public.get_section("public#1").text == "text"
        assert public.get_section("secret#1") is None                 # another group's document
        assert internal.get_section("secret#1").title == "Secret"
        assert public.get_section("old#1") is None                    # superseded revision
        assert public.get_section("missing#1") is None


def test_search_kb_injects_the_users_groups(conn):
    retriever = RecordingRetriever({"x": SearchResult([], [], "t", {})})
    with Tracer(conn, "q") as trace:
        tools = KBTools(conn, retriever, SearchRequest("q", groups=["internal"], release=2026), trace)
        tools.search_kb("x", category="installation", side="A")
    req, side = retriever.requests[0]
    assert (req.query, req.groups, req.release, req.category, side) == ("x", ["all", "internal"], 2026,
                                                                         "installation", "A")


# ------------------------------------------------------------------------------- comparison answer

class FakeLLM:
    name, model = "ollama", "qwen"

    def __init__(self):
        self.messages = None

    def generate(self, messages, *, on_token=None, json_format=False):
        self.messages = messages
        return Generation("Cloud needs a certificate [1]; on premises uses HTTPS [2].", "ollama", "qwen", 1.0)


def side_result(sid, score):
    c = cand(f"{sid}#0", score, "Guide > 2 Certificates")
    c.section_id = sid
    return SearchResult([c], [unit(sid)], "t", {})


def test_comparison_searches_each_side_and_names_the_sides(conn):
    planner = FakePlanner('{"sides": [{"label": "Cloud", "query": "certificates for SAML on Cloud?"},'
                          ' {"label": "On premises", "query": "certificates for HTTPS on premises?"}]}')
    retriever = RecordingRetriever({"certificates for SAML on Cloud?": side_result("cloud#2", 0.95),
                                    "certificates for HTTPS on premises?": side_result("onprem#3", 0.80)})
    llm = FakeLLM()
    answer = Answerer(conn, retriever, {"ollama": llm}, planner=planner).answer(
        SearchRequest("Certificates for SAML on Cloud versus HTTPS on premises?"))
    assert (answer.route, answer.sides, answer.status) == ("compare", ["Cloud", "On premises"], "answered")
    assert [s.doc_id for s in answer.sources] == ["cloud", "onprem"]
    assert [side for _, side in retriever.requests] == ["Cloud", "On premises"]
    prompt = llm.messages[1]["content"]
    assert "found for: Cloud" in prompt and "found for: On premises" in prompt and "The question compares" in prompt
    trace = conn.execute("SELECT route, top_rerank_score FROM traces WHERE trace_id = ?", (answer.trace_id,)).fetchone()
    assert (trace["route"], trace["top_rerank_score"]) == ("compare", 0.95)
    stages = [r["stage"] for r in conn.execute("SELECT stage FROM trace_stages WHERE trace_id = ? ORDER BY seq",
                                               (answer.trace_id,))]
    assert stages[:4] == ["route", "decompose", "search", "search"]


def test_comparison_falls_back_to_one_search_when_it_cannot_be_split(conn):
    question = "Certificates for SAML on Cloud versus HTTPS on premises?"
    retriever = RecordingRetriever({question: side_result("cloud#2", 0.95)})
    answer = Answerer(conn, retriever, {"ollama": FakeLLM()}, planner=FakePlanner("no json")).answer(SearchRequest(question))
    assert answer.route == "answer" and len(retriever.requests) == 1
    assert any("one search" in n for n in answer.notices)
    off = Answerer(conn, retriever, {"ollama": FakeLLM()}, planner=FakePlanner("unused"), compare=False)
    assert off.answer(SearchRequest(question)).notices == [] and off.planner.calls == []


def test_retrieve_sides_puts_reranked_candidates_first(conn):
    low = side_result("a#1", 0.05)
    unreranked = side_result("b#1", None)
    unreranked.candidates[0].score = 0.5
    with Tracer(conn, "q") as trace:
        tools = KBTools(conn, RecordingRetriever({"qa": low, "qb": unreranked}), SearchRequest("q"), trace)
        result = retrieve_sides(tools, "q", [Side("A", "qa"), Side("B", "qb")])
    assert result.candidates[0].rerank_score == 0.05            # the gate still sees the (low) rerank score
