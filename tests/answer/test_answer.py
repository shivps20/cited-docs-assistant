import json
from types import SimpleNamespace

import pytest

from kb.answer.pipeline import Answerer
from kb.core.db import connect, migrate
from kb.core.domain import Domain, use_domain
from kb.llm.prompts import (
    NOT_FOUND,
    build_messages,
    check_citations,
    missing_references,
    source_line,
    system_prompt,
)
from kb.llm.providers import Generation, LLMError, OllamaProvider, select_provider
from kb.retrieve.assemble import ContextUnit, SameText
from kb.retrieve.gate import gate
from kb.retrieve.pipeline import SearchRequest, SearchResult
from kb.retrieve.search import Candidate


def unit(doc="saml", n="2.2.3", pages=(9, 10), external_ok=True, release="R2026x", text="Select Configure Metadata."):
    return ContextUnit(doc_id=doc, title=doc.upper(), section_id=f"{doc}#{n}", section_number=n,
                       heading_path=f"2 Configuring > {n} Configure Metadata", header=doc, page_start=pages[0],
                       page_end=pages[1], text=text, kind="section", score=0.9, tokens=50, release=release,
                       external_ok=external_ok)


def cand(rerank_score):
    return Candidate(point_id="p", chunk_id="saml#2.2.3#0", doc_id="saml", section_id="saml#2.2.3",
                     section_number="2.2.3", chunk_index=0, title="SAML", header="h", text="t", page_start=9,
                     page_end=9, score=0.5, rerank_score=rerank_score)


# ---------------------------------------------------------------------------- prompt + citations

def test_prompt_numbers_sources_with_title_release_section_and_pages():
    messages = build_messages("How do I configure metadata?", [unit(), unit("oracle", "4.5", (49, 49), release="any")])
    system, user = messages[0]["content"], messages[1]["content"]
    assert NOT_FOUND in system and "knowledge-base article numbers (e.g. KB0012345)" in system
    assert "[1] SAML (applies to R2026x) | Section 2 Configuring > 2.2.3 Configure Metadata | pp. 9-10" in user
    assert "[2] ORACLE (applies to all releases)" in user and "p. 49" in user
    assert user.index("Sources:") < user.index("Question: How do I configure metadata?")


def test_check_citations_keeps_valid_drops_invalid_and_splits_lists():
    c = check_citations("Open the Control Center [1]. Select the key [1, 3] and apply [7].", n_sources=3)
    assert c.text == "Open the Control Center [1]. Select the key [1][3] and apply."
    assert (c.cited, c.invalid, c.refused) == ([1, 3], [7], False)


def test_check_citations_detects_refusal():
    c = check_citations("I could not find the answer to this question in the available documents.", 4)
    assert c.refused and c.text == NOT_FOUND
    partial = check_citations("Apache is covered [2], but I could not find the answer for NGINX.", 4)
    assert not partial.refused                    # cites a source: an answer, not a refusal
    stray = check_citations(NOT_FOUND + " [1][2][3]", 4)   # seen from qwen2.5 on q044
    assert (stray.refused, stray.text, stray.cited) == (True, NOT_FOUND, [])


def test_uncited_answer_with_tacked_on_not_found_is_an_answer_with_inferred_sources(conn):
    # Seen from qwen2.5: the right commands, no [n] at all, and the not-found sentence at the end.
    reply = ("The ACME commands to manage the index configuration are:\n\n```acme\n"
             "ACME> set index param depth 5;\n```\n\n" + NOT_FOUND)
    index = unit("index", "3.1.7", (20, 22), text="For example:\n\nACME> set index param depth 5;")
    other = unit("saml", "2.2.3", text="Select Configure Metadata.")
    answerer = Answerer(conn, FakeRetriever(SearchResult([cand(0.25)], [other, index], "t", {})),
                        {"ollama": FakeLLM("ollama", reply=reply)})
    answer = answerer.answer(SearchRequest("What are the different ACME commands for the search index?", rerank_top=10))
    assert answer.status == "answered" and NOT_FOUND not in answer.text
    assert [s.n for s in answer.sources] == [2] and any("identified from the commands" in n for n in answer.notices)
    assert any("contradictory" in n for n in answer.notices)


def test_reworded_refusal_is_still_a_refusal():
    assert check_citations("I could not find the answer to this question in the provided sources.", 3).refused


def test_not_found_sentence_tacked_onto_an_answer_is_removed():
    c = check_citations("Commands:\n\n- `print index params all;` [1]\n\n" + NOT_FOUND, 2)
    assert (c.text, c.refused) == ("Commands:\n\n- `print index params all;` [1]", False)


def test_repeated_and_trailing_citations_are_tidied():
    assert check_citations("The default is 6 [1]. [1]", 2).text == "The default is 6 [1]."
    assert check_citations("Step one [1][1].", 2).text == "Step one [1]."
    assert check_citations("Apply it.\n\nNote: re-import [1].\n\n[1]", 2).text == "Apply it.\n\nNote: re-import [1]."
    kept = check_citations("Apply it [1].\n\n[2]", 2)      # [2] is cited nowhere else: keep it
    assert kept.text.endswith("[2]") and kept.cited == [1, 2]


def test_progress_bars_patched_off():
    import FlagEmbedding.inference.reranker.encoder_only.base as reranker_module

    from kb.core.progress import silence_progress_bars

    silence_progress_bars()
    bar = reranker_module.trange(0, 20, 4, desc="Compute Scores", disable=False)
    assert bar.disable and list(bar) == [0, 4, 8, 12, 16]


def test_source_line():
    assert source_line(2, unit()) == "[2] SAML (R2026x), Section 2.2.3 Configure Metadata, pp. 9-10"


def test_source_line_names_same_text_copies():
    u = unit("oracle", "3.4", (8, 8))
    u.same_text = [SameText("mssql", "MSSQL", "mssql#3.4", "3.4", 8, 9, "R2026x")]
    assert source_line(1, u) == ("[1] ORACLE (R2026x), Section 3.4 Configure Metadata, p. 8"
                                 " · same text: MSSQL (R2026x), Section 3.4, pp. 8-9")
    assert "MSSQL" not in build_messages("q", [u])[1]["content"]       # the LLM sees the text once, unchanged


# ------------------------------------------------------------------------------------------- gate

def test_gate_low_score_no_context_and_unreranked():
    assert gate(SearchResult([cand(0.08)], [unit()], "t", {}), 0.3).decision == "low_score"
    assert gate(SearchResult([], [], "t", {}), 0.3).decision == "no_context"
    assert gate(SearchResult([cand(0.9)], [unit()], "t", {}), 0.3).passed
    assert gate(SearchResult([cand(None)], [unit()], "t", {}), 0.3).passed      # not reranked: LLM decides


# -------------------------------------------------------------------------------------- providers

def test_select_provider_respects_external_ok():
    ok, blocked = [unit()], [unit(), unit("tracing", external_ok=False)]
    assert select_provider("auto", ok, ["ollama"]) == ("ollama", None)            # OpenAI not configured
    assert select_provider("auto", ok, ["ollama", "openai"]) == ("openai", None)
    name, notice = select_provider("auto", blocked, ["ollama", "openai"])
    assert name == "ollama" and "tracing" in notice
    assert select_provider("openai", blocked, ["ollama", "openai"])[0] == "ollama"
    assert select_provider("ollama", ok, ["ollama", "openai"]) == ("ollama", None)
    with pytest.raises(LLMError):
        select_provider("openai", ok, ["ollama"])


def test_ollama_provider_streams_and_reports_tokens():
    parts = [SimpleNamespace(message=SimpleNamespace(content=t), done=False) for t in ("Use ", "port 443 [1].")]
    final = SimpleNamespace(message=SimpleNamespace(content=""), done=True, prompt_eval_count=900, eval_count=12,
                            load_duration=2_000_000_000, eval_duration=500_000_000)
    provider = OllamaProvider("http://127.0.0.1:1", "qwen")
    calls = {}
    provider.client = SimpleNamespace(chat=lambda model, messages, **kw: calls.update(kw) or iter(parts + [final]))
    seen = []
    g = provider.generate([{"role": "user", "content": "q"}], on_token=seen.append)
    assert g.text == "Use port 443 [1]." and seen == ["Use ", "port 443 [1]."]
    assert (g.prompt_tokens, g.output_tokens, g.load_seconds, g.tokens_per_s) == (900, 12, 2.0, 24.0)
    assert calls["stream"] is True and calls["options"]["num_ctx"] == 8192


# --------------------------------------------------------------------------------------- answerer

class FakeRetriever:
    def __init__(self, result):
        self.result = result

    def retrieve(self, req, trace):
        with trace.stage("search"):
            pass
        return self.result


class FakeLLM:
    def __init__(self, name, reply="Select Configure Metadata [1] and apply [5].", fail=False):
        self.name, self.model, self.reply, self.fail, self.calls = name, f"{name}-model", reply, fail, 0

    def generate(self, messages, *, on_token=None):
        self.calls += 1
        if self.fail:
            raise LLMError("down")
        if on_token:
            on_token(self.reply)
        return Generation(self.reply, self.name, self.model, 1.0, prompt_tokens=100, output_tokens=10)


@pytest.fixture
def conn(tmp_path):
    db = tmp_path / "kb.core.db"
    migrate(connect(db, check_schema=False))
    c = connect(db)
    yield c
    c.close()


def test_answer_cites_sources_and_traces_everything(conn):
    llm = FakeLLM("ollama")
    answerer = Answerer(conn, FakeRetriever(SearchResult([cand(0.95)], [unit()], "t", {})), {"ollama": llm})
    answer = answerer.answer(SearchRequest("how?", rerank_top=10))
    assert answer.status == "answered" and answer.text == "Select Configure Metadata [1] and apply."
    assert [s.line for s in answer.sources] == ["[1] SAML (R2026x), Section 2.2.3 Configure Metadata, pp. 9-10"]
    assert answer.invalid_citations == [5]
    assert "Sources:" in answer.formatted()

    trace = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (answer.trace_id,)).fetchone()
    assert (trace["route"], trace["gate_decision"], trace["llm_provider"]) == ("answer", "pass", "ollama")
    assert json.loads(trace["sources"])[0]["section"] == "2.2.3"
    stages = [r["stage"] for r in conn.execute("SELECT stage FROM trace_stages WHERE trace_id = ? ORDER BY seq",
                                               (answer.trace_id,))]
    assert stages == ["route", "search", "gate", "generate", "cite"]


def test_condensed_follow_up_is_traced_with_both_questions(conn):
    answerer = Answerer(conn, FakeRetriever(SearchResult([cand(0.95)], [unit()], "t", {})), {"ollama": FakeLLM("ollama")})
    answer = answerer.answer(SearchRequest("How do I configure SAML metadata on Oracle?", rerank_top=10),
                             asked="and on Oracle?", pre_stages=[("condense", 812.0, {"reason": "follow-up rewritten"})])
    trace = conn.execute("SELECT query, standalone_query FROM traces WHERE trace_id = ?", (answer.trace_id,)).fetchone()
    assert (trace["query"], trace["standalone_query"]) == ("and on Oracle?", "How do I configure SAML metadata on Oracle?")
    first = conn.execute("SELECT stage, duration_ms FROM trace_stages WHERE trace_id = ? ORDER BY seq LIMIT 1",
                         (answer.trace_id,)).fetchone()
    assert (first["stage"], first["duration_ms"]) == ("condense", 812.0)


def test_gate_refuses_without_calling_the_llm(conn):
    llm = FakeLLM("ollama")
    answerer = Answerer(conn, FakeRetriever(SearchResult([cand(0.05)], [unit()], "t", {})), {"ollama": llm})
    answer = answerer.answer(SearchRequest("licence price?", rerank_top=10))
    assert (answer.status, answer.refused_by, answer.text, llm.calls) == ("not_found", "gate", NOT_FOUND, 0)


def test_missing_qa_numbers_and_urls_from_cited_sources_are_appended(conn):
    cited = unit(text="See KB0433809 and https://launcher.example.com/info. Also KB0111111.")
    uncited = unit("other", "9", text="KB0999999 https://example.com/x")
    llm = FakeLLM("ollama", reply="Use port 20300 [1]. See KB0111111 [1].")
    answerer = Answerer(conn, FakeRetriever(SearchResult([cand(0.9)], [cited, uncited], "t", {})), {"ollama": llm})
    answer = answerer.answer(SearchRequest("ports?", rerank_top=10))
    assert answer.references == ["- KB0433809 [1]", "- https://launcher.example.com/info [1]"]
    assert answer.text.endswith("Articles and links in the cited sections:\n- KB0433809 [1]\n"
                                "- https://launcher.example.com/info [1]")
    assert "KB0999999" not in answer.text            # only cited sources contribute


def test_broad_question_with_modest_score_reaches_the_llm(conn):
    # 'What are the different ACME commands for the search index?' scored 0.25 with the right section first.
    llm = FakeLLM("ollama", reply="Use `set index param depth 5;` [1].")
    answerer = Answerer(conn, FakeRetriever(SearchResult([cand(0.25)], [unit()], "t", {})), {"ollama": llm})
    answer = answerer.answer(SearchRequest("What are the different ACME commands for the search index?", rerank_top=10))
    assert (answer.status, llm.calls) == ("answered", 1)


def test_llm_refusal_is_not_found(conn):
    llm = FakeLLM("ollama", reply=NOT_FOUND)
    answerer = Answerer(conn, FakeRetriever(SearchResult([cand(0.9)], [unit()], "t", {})), {"ollama": llm})
    answer = answerer.answer(SearchRequest("NGINX config?", rerank_top=10))
    assert (answer.status, answer.refused_by, answer.sources) == ("not_found", "llm", [])


def test_openai_failure_falls_back_to_ollama(conn):
    local, external = FakeLLM("ollama"), FakeLLM("openai", fail=True)
    answerer = Answerer(conn, FakeRetriever(SearchResult([cand(0.9)], [unit()], "t", {})),
                        {"ollama": local, "openai": external})
    answer = answerer.answer(SearchRequest("how?", rerank_top=10))
    assert answer.generation.provider == "ollama" and external.calls == 1 and local.calls == 1
    assert any("OpenAI failed" in n for n in answer.notices)


def test_domain_rules_fill_the_prompt_and_the_appended_references():
    domain = Domain(reference_patterns=(r"\bDOC-\d{4}\b",), reference_label="document ids",
                    reference_example="DOC-0042", synonym_examples='"start" and "launch"')
    prompt = system_prompt(domain)
    assert '(e.g. "start" and "launch")' in prompt and "document ids (e.g. DOC-0042)" in prompt
    use_domain(domain)
    refs = missing_references("See DOC-0001.", [(1, unit(text="DOC-0001 and DOC-0002, KB0433809"))])
    assert refs == ["- DOC-0002 [1]"]                     # only the domain's pattern counts
