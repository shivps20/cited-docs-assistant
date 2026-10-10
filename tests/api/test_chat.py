import json
import os
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from kb.answer.pipeline import Answer, Source
from kb.api import health, sessions
from kb.api.app import create_app
from kb.api.chat import run_turn
from kb.api.services import Services
from kb.api.users import User, UserDirectory
from kb.core.config import get_settings
from kb.core.db import connect, migrate
from kb.llm.condense import clean_rewrite, condense, needs_condensing
from kb.llm.providers import Generation, LLMError
from kb.llm.registry import ModelRegistry
from kb.retrieve.assemble import ContextUnit
from kb.retrieve.gate import GateDecision
from kb.retrieve.release import resolve_release

HISTORY = [{"role": "user", "content": "Which TomEE port does CoreServer use on MSSQL?"},
           {"role": "assistant", "content": "CoreServer uses TomEE port 9040 [1]."}]


# ---------------------------------------------------------------------------------- condensing

@pytest.mark.parametrize("question, expected", [
    ("and on Oracle?", True),                                   # short + opener
    ("How do I change it?", True),                              # back-reference
    ("What about the FCS port?", True),                         # opener
    ("Can i use any other port?", True),                        # 'other' refers back (seen in a live chat)
    ("Is there an alternative to the external TomEE setup?", True),   # 'alternative' refers back
    ("What is the default index depth for Search Index volumic search?", False),   # self-contained
    (("Which databases are supported by the Acme platform installation on Windows servers "
      "for R2026x and what does that mean for upgrades?"), False),                  # long: assumed standalone
])
def test_needs_condensing(question, expected):
    assert needs_condensing(question, HISTORY) is expected
    assert needs_condensing(question, []) is False              # never without history


def test_clean_rewrite_accepts_one_question_and_rejects_answers():
    assert clean_rewrite('Rewritten question: "Which TomEE port does CoreServer use on Oracle?"\n', "and on Oracle?") \
        == "Which TomEE port does CoreServer use on Oracle?"
    assert clean_rewrite("I could not find the answer to this question.", "and on Oracle?") is None
    assert clean_rewrite("", "x") is None
    assert clean_rewrite(" ".join(["word"] * 60), "and on Oracle?") is None          # an answer, not a question


class FakeCondenser:
    def __init__(self, reply="Which TomEE port does CoreServer use on Oracle?", fail=False):
        self.reply, self.fail, self.calls = reply, fail, 0

    def generate(self, messages, *, on_token=None):
        self.calls += 1
        if self.fail:
            raise LLMError("down")
        assert "9040" in messages[1]["content"]                  # the history reaches the condenser
        return Generation(self.reply, "ollama", "qwen", 0.5)


def test_condense_rewrites_follow_ups_and_falls_back_safely():
    c = condense(FakeCondenser(), "and on Oracle?", HISTORY)
    assert (c.question, c.condensed) == ("Which TomEE port does CoreServer use on Oracle?", True)
    llm = FakeCondenser()
    assert condense(llm, "What is the default index depth for Search Index?", HISTORY).condensed is False
    assert llm.calls == 0                                       # self-contained: no LLM call
    failed = condense(FakeCondenser(fail=True), "and on Oracle?", HISTORY)
    assert (failed.question, failed.condensed) == ("and on Oracle?", False)


# ---------------------------------------------------------------------------------- sticky release

def test_resolve_release():
    assert (resolve_release("Install on R2025x?", None).release, resolve_release("Install on R2025x?", None).sticky) \
        == (2025, "R2025x")
    assert resolve_release("and the ports?", "R2025x").release == 2025                 # sticky
    assert resolve_release("same for any release", "R2025x").sticky is None          # cleared
    several = resolve_release("Compare R2024x and R2026x", "R2025x")
    assert (several.release, several.sticky, several.reason) == (None, "R2025x", "several releases")
    assert resolve_release("How do I install?", None).reason == "none"


# ---------------------------------------------------------------------------------- /api/chat

USERS = UserDirectory({"guest": User("guest", "Guest"), "internal_user": User("internal_user", "Internal", ("internal",))},
                      "guest")


def unit():
    return ContextUnit(doc_id="mssql", title="MSSQL Install", section_id="mssql#4.4", section_number="4.4",
                       heading_path="4 Install > 4.4 Ports", header="h", page_start=22, page_end=22,
                       text="| CoreServer | 9040 |", kind="section", score=0.97, tokens=20, release="R2026x")


class FakeAnswerer:
    def __init__(self, conn, log):
        self.conn, self.log = conn, log

    def answer(self, req, *, provider=None, on_token=None, on_context=None, asked=None, pre_stages=(),
               on_status=None, use_cache=True):
        self.log.append({"query": req.query, "groups": req.groups, "release": req.release, "asked": asked,
                         "pre_stages": [s[0] for s in pre_stages], "model": provider, "use_cache": use_cache})
        if req.query == "boom":
            raise LLMError("Ollama is not reachable")
        if "versus" in req.query:
            on_status("comparing", {})
            for i, side in enumerate(("MSSQL", "Oracle"), start=1):
                on_status("searching_side", {"side": side, "index": i, "total": 2})
        on_context([unit()])
        for piece in ("CoreServer uses ", "port 9040 [1]."):
            on_token(piece)
        src = Source(1, "mssql", "MSSQL Install", "R2026x", "4.4", "4.4 Ports", 22, 22, "[1] MSSQL Install …")
        return Answer(question=req.query, text="CoreServer uses port 9040 [1].", status="answered",
                      gate=GateDecision("pass", 0.97, 0.1), sources=[src], context=[unit()], candidates=[],
                      trace_id="trace-1", timings_ms={"generate": 900.0},
                      generation=Generation("CoreServer uses port 9040 [1].", "ollama", "qwen", 1.0, output_tokens=8))


class FakeServices(Services):
    def make_answerer(self, conn):
        return FakeAnswerer(conn, self.calls)


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    db = tmp_path / "kb.db"
    c = connect(db, check_schema=False)
    migrate(c)
    c.close()
    monkeypatch.setattr(health, "ollama_status", lambda host, model: {"ok": True, "detail": "fake"})
    svc = FakeServices(settings=get_settings().model_copy(update={"db_path": db, "models_path": tmp_path / "no-models.yaml"}),
                       users=USERS,
                       client=SimpleNamespace(), embedder=object(), reranker=object(),
                       models=ModelRegistry.from_providers({"ollama": SimpleNamespace(model="qwen")}), condenser=FakeCondenser())
    svc.calls = []
    with TestClient(create_app(svc)) as client:
        yield client, svc, db


def test_chat_streams_events_and_stores_both_messages(app_env):
    client, svc, db = app_env
    events = parse_sse(client.post("/api/chat", json={"question": "Which TomEE port does CoreServer use on R2026x?"},
                                   headers={"X-KB-User": "internal_user"}).text)
    names = [e for e, _ in events]
    assert names[0] == "session" and names[-1] == "final"
    assert names[1:3] == ["status", "context"] and names.count("token") == 2
    final = events[-1][1]
    assert final["answer"] == "CoreServer uses port 9040 [1]." and final["trace_id"] == "trace-1"
    assert final["release"] == "R2026x" and final["sources"][0]["section"] == "4.4"
    assert svc.calls[0]["groups"] == ["all", "internal"] and svc.calls[0]["release"] == 2026

    sid = events[0][1]["session_id"]
    conn = connect(db)
    stored = sessions.get_session(conn, sid, "internal_user")
    conn.close()
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
    assert stored["messages"][1]["trace_id"] == "trace-1" and stored["sticky_release"] == "R2026x"


def test_follow_up_is_condensed_and_release_stays_sticky(app_env):
    client, svc, _ = app_env
    sid = parse_sse(client.post("/api/chat", json={"question": "Which TomEE port does CoreServer use on R2026x?"}).text)[0][1]["session_id"]
    events = parse_sse(client.post("/api/chat", json={"question": "and on Oracle?", "session_id": sid}).text)
    assert [d["stage"] for e, d in events if e == "status"] == ["condensing", "searching"]
    second = svc.calls[1]
    assert second["query"] == "Which TomEE port does CoreServer use on Oracle?" and second["asked"] == "and on Oracle?"
    assert second["release"] == 2026 and second["pre_stages"] == ["condense"]
    assert events[-1][1]["standalone_query"] == "Which TomEE port does CoreServer use on Oracle?"


def test_comparison_progress_is_streamed_per_side(app_env):
    client, _, _ = app_env
    events = parse_sse(client.post("/api/chat", json={"question": "Database creation on MSSQL versus Oracle?"}).text)
    statuses = [d for e, d in events if e == "status"]
    assert [d["stage"] for d in statuses] == ["searching", "comparing", "searching_side", "searching_side"]
    assert statuses[3] == {"stage": "searching_side", "side": "Oracle", "index": 2, "total": 2}


def test_chat_rejects_other_users_sessions_and_blank_questions(app_env):
    client, _, _ = app_env
    sid = parse_sse(client.post("/api/chat", json={"question": "Which ports?"},
                                headers={"X-KB-User": "internal_user"}).text)[0][1]["session_id"]
    assert client.post("/api/chat", json={"question": "and?", "session_id": sid}).status_code == 404
    assert client.post("/api/chat", json={"question": "   "}).status_code == 422
    unknown = client.post("/api/chat", json={"question": "x", "model": "gpt"})
    assert unknown.status_code == 400 and "unknown model 'gpt'" in unknown.json()["detail"]
    assert client.post("/api/chat", json={"question": "x", "provider": "gpt"}).status_code == 400   # older field


def test_turn_waiting_for_the_model_lock_reports_queued(app_env):
    _, svc, _ = app_env
    conn = svc.connect()
    sid = sessions.create_session(conn, USERS.resolve("guest"))["session_id"]
    conn.close()
    events, seen_queued = [], threading.Event()

    def emit(name, data):
        events.append((name, data))
        if data.get("stage") == "queued":
            seen_queued.set()

    svc.model_lock.acquire()                                    # another turn is running
    worker = threading.Thread(target=run_turn, args=(svc, USERS.resolve("guest"), sid, "Which ports?", None, emit))
    worker.start()
    assert seen_queued.wait(5)
    assert not any(name == "final" for name, _ in events)       # nothing runs while the lock is held
    svc.model_lock.release()
    worker.join(5)
    assert events[0] == ("status", {"stage": "queued"}) and events[-1][0] == "final"
    assert not svc.model_lock.locked()                          # released after the turn


def test_llm_failure_becomes_an_error_event(app_env):
    client, _, _ = app_env
    events = parse_sse(client.post("/api/chat", json={"question": "boom"}).text)
    assert events[-1] == ("error", {"message": "Ollama is not reachable"})


# ---------------------------------------------------------------------------------- feedback, release, UI

def add_trace(db, trace_id, user_id, sources=()):
    conn = connect(db)
    with conn:
        conn.execute("INSERT INTO traces (trace_id, user_id, query, sources, gate_decision) VALUES (?, ?, 'q', ?, 'pass')",
                     (trace_id, user_id, json.dumps(list(sources))))
    conn.close()


def test_feedback_is_stored_once_per_user_and_only_for_own_answers(app_env):
    client, _, db = app_env
    add_trace(db, "trace-own", "guest")
    add_trace(db, "trace-other", "internal_user")
    assert client.post("/api/feedback", json={"trace_id": "trace-own", "rating": 1}).json()["rating"] == 1
    again = client.post("/api/feedback", json={"trace_id": "trace-own", "rating": -1, "comment": "  wrong port  "})
    assert again.json() == {"trace_id": "trace-own", "rating": -1, "reason": None, "comment": "wrong port"}
    conn = connect(db)
    assert tuple(conn.execute("SELECT COUNT(*), MAX(rating) FROM feedback").fetchone()) == (1, -1)  # replaced, not added
    conn.close()
    assert client.post("/api/feedback", json={"trace_id": "trace-other", "rating": 1}).status_code == 404
    assert client.post("/api/feedback", json={"trace_id": "nope", "rating": 1}).status_code == 404
    assert client.post("/api/feedback", json={"trace_id": "trace-own", "rating": 5}).status_code == 422


def test_reopened_session_shows_sources_and_rating(app_env):
    client, _, db = app_env
    sid = client.post("/api/sessions").json()["session_id"]
    add_trace(db, "trace-9", "guest", sources=[{"n": 1, "line": "[1] MSSQL Install, Section 4.4, p. 22"}])
    conn = connect(db)
    sessions.add_message(conn, sid, "user", "Which port?")
    sessions.add_message(conn, sid, "assistant", "9040 [1].", trace_id="trace-9")
    conn.close()
    client.post("/api/feedback", json={"trace_id": "trace-9", "rating": 1, "comment": "good"})
    answer = client.get(f"/api/sessions/{sid}").json()["messages"][1]
    assert answer["sources"][0]["line"].startswith("[1] MSSQL") and (answer["rating"], answer["comment"]) == (1, "good")


def test_release_can_be_set_and_cleared(app_env):
    client, _, _ = app_env
    sid = client.post("/api/sessions").json()["session_id"]
    assert client.put(f"/api/sessions/{sid}/release", json={"release": "r2025x"}).json()["sticky_release"] == "R2025x"
    assert client.get(f"/api/sessions/{sid}").json()["sticky_release"] == "R2025x"
    assert client.put(f"/api/sessions/{sid}/release", json={"release": None}).json()["sticky_release"] is None
    assert client.put(f"/api/sessions/{sid}/release", json={"release": "2025"}).status_code == 422
    assert client.put("/api/sessions/unknown/release", json={"release": None}).status_code == 404


def test_ui_page_is_served(app_env):
    client, _, _ = app_env
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "/api/chat" in page.text and "X-KB-User" in page.text


def test_models_endpoint_and_model_choice_reach_the_answerer(app_env):
    client, svc, _ = app_env
    listed = client.get("/api/models").json()
    assert listed["default"] == "ollama" and listed["fallback"] == "ollama"
    assert listed["models"] == [{"name": "ollama", "model": "qwen", "adapter": "ollama", "location": "local", "ready": True}]
    parse_sse(client.post("/api/chat", json={"question": "Which ports?", "model": "ollama"}).text)
    parse_sse(client.post("/api/chat", json={"question": "Which ports?"}).text)
    assert [c["model"] for c in svc.calls[-2:]] == ["ollama", None]


def test_models_yaml_is_reloaded_when_it_changes_and_a_broken_edit_is_ignored(tmp_path):


    path = tmp_path / "models.yaml"
    settings = get_settings().model_copy(update={"models_path": path, "db_path": tmp_path / "kb.db"})
    svc = Services(settings=settings, users=USERS, client=None, embedder=None, reranker=None,
                   models=ModelRegistry.from_providers({"ollama": SimpleNamespace(model="qwen")}))
    assert svc.current_models().catalogue.source == "providers"           # no file: what it started with
    path.write_text("fallback: small\nroles: {answer: small}\nmodels:\n  small: {adapter: ollama, model: qwen-small}\n",
                    encoding="utf-8")
    reloaded = svc.current_models()
    assert reloaded.catalogue.roles["answer"] == "small" and svc.condenser is not None
    path.write_text("models: [not, a, mapping]\n", encoding="utf-8")
    os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 5))      # a newer edit, with an error
    assert svc.current_models() is reloaded                                # the previous catalogue stays


def test_thumbs_down_removes_the_answer_from_the_cache(app_env):
    client, _, db = app_env
    add_trace(db, "trace-cached", "guest")
    conn = connect(db)
    with conn:
        conn.execute("INSERT INTO answer_cache (cache_key, query_norm, user_groups, index_version, model_id, answer) "
                     "VALUES ('k', 'which port', '[]', 'c', 'm', ?)", ('{"trace_id": "trace-cached"}',))
    client.post("/api/feedback", json={"trace_id": "trace-cached", "rating": 1})
    assert conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 1      # 👍 keeps it
    client.post("/api/feedback", json={"trace_id": "trace-cached", "rating": -1})
    assert conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 0
    conn.close()


def test_chat_can_skip_the_cache(app_env):
    client, services, _ = app_env
    client.post("/api/chat", json={"question": "Which port?", "cache": False})
    assert services.calls[-1]["use_cache"] is False


def test_thumbs_down_reason_is_stored_validated_and_shown_again(app_env):
    client, _, db = app_env
    sid = client.post("/api/sessions").json()["session_id"]
    add_trace(db, "trace-r", "guest")
    conn = connect(db)
    sessions.add_message(conn, sid, "assistant", "Not found.", trace_id="trace-r")
    conn.close()
    saved = client.post("/api/feedback", json={"trace_id": "trace-r", "rating": -1, "reason": "should_have_answered"})
    assert saved.json()["reason"] == "should_have_answered"
    assert client.get(f"/api/sessions/{sid}").json()["messages"][0]["reason"] == "should_have_answered"
    assert client.post("/api/feedback", json={"trace_id": "trace-r", "rating": 1, "reason": "wrong"}).status_code == 422
    assert client.post("/api/feedback", json={"trace_id": "trace-r", "rating": -1, "reason": "rude"}).status_code == 422
    client.post("/api/feedback", json={"trace_id": "trace-r", "rating": 1})          # a thumbs up clears the reason
    assert client.get(f"/api/sessions/{sid}").json()["messages"][0]["reason"] is None
