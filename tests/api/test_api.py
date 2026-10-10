from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from kb.api import health, sessions
from kb.api.app import create_app
from kb.api.services import Services
from kb.api.users import UnknownUser, User, UserDirectory, UsersError, load_users
from kb.core.config import ROOT, get_settings
from kb.core.db import connect, migrate
from kb.llm.registry import ModelRegistry

USERS = UserDirectory({"guest": User("guest", "Guest"), "internal_user": User("internal_user", "Internal", ("internal",))},
                      "guest")


# ------------------------------------------------------------------------------------- users.yaml

def test_load_users_and_resolve(tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text("default_user: guest\nusers:\n  guest: {name: Guest}\n"
                    "  internal_user: {name: Internal, groups: [internal, all]}\n", encoding="utf-8")
    users = load_users(path)
    assert users.resolve(None).user_id == "guest"
    assert users.resolve("internal_user").search_groups == ["all", "internal"]
    assert users.resolve("internal_user").groups == ("internal",)     # 'all' is implicit, not stored
    with pytest.raises(UnknownUser):
        users.resolve("mallory")


def test_load_users_reports_every_problem(tmp_path):
    path = tmp_path / "users.yaml"
    path.write_text("default_user: nobody\nusers:\n  Bad Id: {}\n  ok: {groups: [Not Valid]}\n", encoding="utf-8")
    with pytest.raises(UsersError) as e:
        load_users(path)
    assert len(e.value.errors) == 3
    assert load_users(ROOT / "config" / "users.example.yaml").default_user == "guest"   # the committed example is valid


# ----------------------------------------------------------------------------------------- sessions

@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "kb.db"
    conn = connect(path, check_schema=False)
    migrate(conn)
    conn.close()
    return path


def test_sessions_are_per_user_and_keep_messages(db_path):
    conn = connect(db_path)
    guest, other = USERS.resolve("guest"), USERS.resolve("internal_user")
    s = sessions.create_session(conn, guest)
    assert s["user_groups"] == ["all"] and s["messages"] == []
    sessions.add_message(conn, s["session_id"], "user", "Which ports does the Launcher use?")
    sessions.add_message(conn, s["session_id"], "assistant", "20300 [1]", trace_id="t1")
    sessions.add_message(conn, s["session_id"], "user", "and on Linux?", standalone_query="Launcher ports on Linux?")
    assert sessions.get_session(conn, s["session_id"], other.user_id) is None          # not another user's
    listed = sessions.list_sessions(conn, "guest")
    assert listed[0]["title"] == "Which ports does the Launcher use?" and listed[0]["message_count"] == 3
    assert [m["role"] for m in sessions.recent_turns(conn, s["session_id"], turns=2)] == ["assistant", "user"]
    sessions.set_sticky_release(conn, s["session_id"], "R2025x")
    assert sessions.get_session(conn, s["session_id"], "guest")["sticky_release"] == "R2025x"
    conn.close()


# ---------------------------------------------------------------------------------------------- API

class FakeQdrant:
    def collection_exists(self, name):
        return True

    def count(self, name, exact=True):
        return SimpleNamespace(count=861)


@pytest.fixture
def client(db_path, monkeypatch):
    monkeypatch.setattr(health, "ollama_status", lambda host, model: {"ok": True, "detail": "fake"})
    settings = get_settings().model_copy(update={"db_path": db_path, "models_path": db_path.parent / "no-models.yaml"})
    svc = Services(settings=settings, users=USERS, client=FakeQdrant(), embedder=object(), reranker=object(),
                   models=ModelRegistry.from_providers({"ollama": SimpleNamespace(model="qwen")}), load_seconds=1.5)
    with TestClient(create_app(svc)) as c:
        yield c


def test_health_reports_every_check(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body["checks"]) == {"sqlite", "qdrant", "ollama", "models"}
    assert "861 points" in body["checks"]["qdrant"]["detail"]


def test_user_comes_from_header_and_groups_from_config(client):
    assert client.get("/api/me").json()["user_id"] == "guest"                         # default user
    me = client.get("/api/me", headers={"X-KB-User": "internal_user"}).json()
    assert me["groups"] == ["all", "internal"]
    assert client.get("/api/me", headers={"X-KB-User": "mallory"}).status_code == 403
    assert {u["user_id"] for u in client.get("/api/users").json()["users"]} == {"guest", "internal_user"}


def test_session_endpoints_hide_other_users_sessions(client):
    created = client.post("/api/sessions", headers={"X-KB-User": "internal_user"})
    assert created.status_code == 201
    sid = created.json()["session_id"]
    assert created.json()["user_groups"] == ["all", "internal"]
    assert client.get(f"/api/sessions/{sid}", headers={"X-KB-User": "internal_user"}).json()["messages"] == []
    assert client.get(f"/api/sessions/{sid}").status_code == 404                      # guest cannot see it
    assert client.get("/api/sessions").json()["sessions"] == []
    listed = client.get("/api/sessions", headers={"X-KB-User": "internal_user"}).json()["sessions"]
    assert [s["session_id"] for s in listed] == [sid] and listed[0]["title"] == "New conversation"
