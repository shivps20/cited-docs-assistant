import pytest

from kb.answer.cache import AnswerCache, normalise_question
from kb.answer.pipeline import Answerer
from kb.core.db import connect, migrate
from kb.llm.prompts import NOT_FOUND
from kb.llm.providers import Generation, LLMError
from kb.retrieve.assemble import ContextUnit, SameText
from kb.retrieve.pipeline import SearchRequest, SearchResult
from kb.retrieve.search import Candidate


def unit():
    """One context unit with a same-text copy (so the cache must round-trip nested data)."""
    return ContextUnit(doc_id="saml", title="SAML", section_id="saml#2.2.3", section_number="2.2.3",
                       heading_path="2 Configuring > 2.2.3 Configure Metadata", header="saml", page_start=9,
                       page_end=10, text="Select Configure Metadata.", kind="window", score=0.9, tokens=50,
                       release="R2026x", window=(0, 2), chunk_ids=["saml#2.2.3#0"],
                       same_text=[SameText("saml2", "SAML 2", "saml2#2.2.3", "2.2.3", 9, 9)])


def cand():
    """A candidate that passes the gate."""
    return Candidate(point_id="p", chunk_id="saml#2.2.3#0", doc_id="saml", section_id="saml#2.2.3",
                     section_number="2.2.3", chunk_index=0, title="SAML", header="h", text="t", page_start=9,
                     page_end=9, score=0.5, rerank_score=0.95)


class FakeRetriever:
    """Returns one fixed result and counts the searches."""

    def __init__(self):
        """No searches yet."""
        self.calls = 0

    def retrieve(self, req, trace):
        """The fixed result."""
        self.calls += 1
        return SearchResult([cand()], [unit()], "t", {})


class FakeLLM:
    """Answers with a fixed reply (or fails) and counts the calls."""

    def __init__(self, name="ollama", reply="Select Configure Metadata [1].", fail=False):
        """Name, reply and whether generate() raises."""
        self.name, self.model, self.reply, self.fail, self.calls = name, f"{name}-model", reply, fail, 0

    def generate(self, messages, *, on_token=None):
        """The reply, streamed in one piece."""
        self.calls += 1
        if self.fail:
            raise LLMError("down")
        if on_token:
            on_token(self.reply)
        return Generation(self.reply, self.name, self.model, 1.0, output_tokens=5)


@pytest.fixture
def conn(tmp_path):
    """A migrated database with one indexed document."""
    db = tmp_path / "kb.db"
    migrate(connect(db, check_schema=False))
    c = connect(db)
    with c:
        c.execute("INSERT INTO documents (doc_id, source_path, file_hash, status, allowed_groups) "
                  "VALUES ('saml', 'saml.pdf', 'h1', 'indexed', '[\"all\"]')")
    yield c
    c.close()


def answerer(conn, llm=None, **kwargs):
    """An Answerer with the cache on, a fake retriever and a fake local model."""
    retriever = FakeRetriever()
    return Answerer(conn, retriever, {"ollama": llm or FakeLLM()}, cache=True, **kwargs), retriever


def test_normalised_question_ignores_case_spacing_and_trailing_punctuation():
    assert normalise_question("  How do I  configure SAML?  ") == normalise_question("how do i configure saml") \
        == "how do i configure saml"


def test_a_repeated_question_is_answered_from_the_cache(conn):
    llm = FakeLLM()
    a, retriever = answerer(conn, llm)
    first = a.answer(SearchRequest("How do I configure SAML?", rerank_top=10))
    streamed, shown = [], []
    second = a.answer(SearchRequest("how do I configure  SAML", rerank_top=10), on_token=streamed.append,
                      on_context=shown.append)
    assert (llm.calls, retriever.calls) == (1, 1)
    assert second.cached_from == first.trace_id and second.trace_id != first.trace_id
    assert second.text == first.text and [s.line for s in second.sources] == [s.line for s in first.sources]
    assert second.context[0].same_text[0].doc_id == "saml2" and second.context[0].window == (0, 2)
    assert streamed == [first.text] and len(shown[0]) == 1
    assert any("Answered from the answer cache" in n for n in second.notices)
    trace = conn.execute("SELECT cache_hit, route, answer FROM traces WHERE trace_id = ?", (second.trace_id,)).fetchone()
    assert (trace["cache_hit"], trace["route"], trace["answer"]) == (1, "answer", first.text)


def test_groups_release_corpus_and_model_settings_are_part_of_the_key(conn):
    llm = FakeLLM()
    a, _ = answerer(conn, llm)
    a.answer(SearchRequest("ports?", rerank_top=10))
    a.answer(SearchRequest("ports?", rerank_top=10, groups=["internal"]))      # other access
    a.answer(SearchRequest("ports?", rerank_top=10, release=2024))             # other release
    assert llm.calls == 3
    Answerer(conn, a.retriever, {"ollama": llm}, cache=True, compare=False).answer(SearchRequest("ports?", rerank_top=10))
    assert llm.calls == 4                                                      # other answer settings
    with conn:
        conn.execute("UPDATE documents SET file_hash = 'h2'")                  # re-indexed corpus
    a.answer(SearchRequest("ports?", rerank_top=10))
    assert llm.calls == 5
    assert conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 1   # older corpus dropped on store


def test_refusals_and_fallback_answers_are_not_stored(conn):
    a, _ = answerer(conn, FakeLLM(reply=NOT_FOUND))
    a.answer(SearchRequest("nginx?", rerank_top=10))
    local, external = FakeLLM("ollama"), FakeLLM("openai", fail=True)
    Answerer(conn, FakeRetriever(), {"ollama": local, "openai": external}, cache=True).answer(
        SearchRequest("how?", rerank_top=10))                                   # openai failed, ollama answered
    assert conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 0


def test_no_cache_answers_afresh_and_cache_off_stores_nothing(conn):
    llm = FakeLLM()
    a, _ = answerer(conn, llm)
    a.answer(SearchRequest("how?", rerank_top=10))
    assert a.answer(SearchRequest("how?", rerank_top=10), use_cache=False).cached_from == ""
    assert llm.calls == 2
    Answerer(conn, FakeRetriever(), {"ollama": llm}).answer(SearchRequest("other?", rerank_top=10))
    assert conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 1


def test_a_thumbs_down_on_the_answer_or_a_cached_copy_removes_the_entry(conn):
    llm = FakeLLM()
    a, _ = answerer(conn, llm)
    first = a.answer(SearchRequest("how?", rerank_top=10))
    assert AnswerCache(conn).forget_trace(first.trace_id) == 1
    a.answer(SearchRequest("how?", rerank_top=10))                             # stored again
    copy = a.answer(SearchRequest("how?", rerank_top=10))
    assert copy.cached_from and AnswerCache(conn).forget_trace(copy.trace_id) == 1
    assert llm.calls == 2


def test_stats_and_clear(conn):
    a, _ = answerer(conn)
    a.answer(SearchRequest("how?", rerank_top=10))
    a.answer(SearchRequest("how?", rerank_top=10))
    cache = AnswerCache(conn)
    s = cache.stats()
    assert (s["entries"], s["hits"], s["current"], s["stale"], s["top"]) == (1, 1, 1, 0, [("how", 1)])
    with conn:
        conn.execute("UPDATE documents SET external_ok = 1")                   # manifest change
    assert cache.stats()["stale"] == 1
    assert cache.clear(stale_only=True) == 1 and cache.clear() == 0
