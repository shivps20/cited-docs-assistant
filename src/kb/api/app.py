"""FastAPI application: health, users, chat sessions and the streaming chat endpoint.

Run with `uv run kb serve`. Models are loaded once at startup (lifespan); every request gets its
own SQLite connection. The user comes from the X-KB-User header (default user if absent) and is
looked up in users.yaml; unknown users get 403.

    GET  /api/health                  SQLite, Qdrant, Ollama, models
    GET  /api/users                   configured users (for the UI's user picker)
    GET  /api/me                      the current user and their access groups
    POST /api/sessions                start a conversation
    GET  /api/sessions                the current user's conversations, most recent first
    GET  /api/sessions/{session_id}   one conversation with its messages (404 if not the user's)
    PUT  /api/sessions/{id}/release   set or clear the conversation's release filter
    POST /api/chat                    ask a question; server-sent events (see kb.api.chat)
    POST /api/feedback                rate an answer (+1 / -1) with an optional comment
    GET  /                            the chat UI (kb/api/static/index.html)
"""

import sqlite3
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from kb.answer.cache import AnswerCache
from kb.api import sessions
from kb.api.chat import stream_turn
from kb.api.health import check_health
from kb.api.services import Services, load_services
from kb.api.users import UnknownUser, User
from kb.core.config import get_settings
from kb.ingest.manifest import parse_release
from kb.llm.providers import LLMError

MAX_QUESTION_CHARS = 2000
MAX_COMMENT_CHARS = 1000
STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    """Body of POST /api/chat. Without session_id a new conversation is started."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = None
    model: str | None = Field(None, max_length=64)       # a catalogue model name; None: the answer role
    provider: str | None = Field(None, max_length=64)    # older name for `model` (auto / ollama / openai)
    cache: bool = True                                   # False: answer afresh, not from the answer cache

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Strip the question and reject one that is only whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("question is empty")
        return value


class FeedbackRequest(BaseModel):
    """Body of POST /api/feedback: a thumbs up (1) or down (-1) for one answer, optionally with a comment."""

    trace_id: str = Field(min_length=1)
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=MAX_COMMENT_CHARS)


class ReleaseRequest(BaseModel):
    """Body of PUT /api/sessions/{id}/release: a release such as 'R2025x', or null for all releases."""

    release: str | None = None

    @field_validator("release")
    @classmethod
    def _valid_release(cls, value: str | None) -> str | None:
        """Normalise 'r2025x' to 'R2025x'; reject anything that is not a release."""
        if value is None or not value.strip():
            return None
        return f"R{parse_release(value)}x"


def create_app(services: Services | None = None) -> FastAPI:
    """Build the app; with `services` given (tests), nothing is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load shared services once when the server starts."""
        app.state.services = services or load_services(get_settings())
        if services is None:                         # real start-up: say plainly what is not reachable
            report = check_health(app.state.services)
            for name, check in report["checks"].items():
                status = "ok  " if check["ok"] else "DOWN"
                print(f"startup check {status} {name}: {check['detail']}", flush=True)
            if report["status"] != "ok":
                print("WARN some dependencies are down: questions will fail until they are started "
                      "(GET /api/health shows the current state)", flush=True)
        yield

    app = FastAPI(title="Knowledge-Base Assistant", version="0.4", lifespan=lifespan)

    def get_services(request: Request) -> Services:
        """The shared services loaded at startup."""
        return request.app.state.services

    Svc = Annotated[Services, Depends(get_services)]

    def get_db(svc: Svc) -> Iterator[sqlite3.Connection]:
        """A SQLite connection for this request, closed afterwards."""
        conn = svc.connect()
        try:
            yield conn
        finally:
            conn.close()

    def current_user(svc: Svc, x_kb_user: Annotated[str | None, Header()] = None) -> User:
        """The user named by the X-KB-User header (or the default user); 403 if unknown."""
        try:
            return svc.users.resolve(x_kb_user)
        except UnknownUser as e:
            raise HTTPException(status_code=403, detail=str(e)) from e

    Db = Annotated[sqlite3.Connection, Depends(get_db)]
    CurrentUser = Annotated[User, Depends(current_user)]

    @app.get("/api/health")
    def health(svc: Svc) -> dict:
        """Status of SQLite, Qdrant, Ollama and the loaded models."""
        return check_health(svc)

    @app.get("/api/users")
    def users(svc: Svc) -> dict:
        """Configured users for the UI's user picker, and the default one."""
        return {"default_user": svc.users.default_user,
                "users": [{"user_id": u.user_id, "name": u.name, "groups": list(u.groups)}
                          for u in svc.users.users.values()]}

    @app.get("/api/me")
    def me(user: CurrentUser) -> dict:
        """The current user and the groups their searches are filtered by."""
        return {"user_id": user.user_id, "name": user.name, "groups": user.search_groups}

    @app.post("/api/sessions", status_code=201)
    def new_session(user: CurrentUser, conn: Db) -> dict:
        """Start a new conversation for the current user."""
        return sessions.create_session(conn, user)

    @app.get("/api/sessions")
    def my_sessions(user: CurrentUser, conn: Db) -> dict:
        """The current user's conversations, most recent first."""
        return {"sessions": sessions.list_sessions(conn, user.user_id)}

    @app.get("/api/sessions/{session_id}")
    def one_session(session_id: str, user: CurrentUser, conn: Db) -> dict:
        """One conversation with its messages; 404 if it does not exist or is another user's."""
        session = sessions.get_session(conn, session_id, user.user_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return session

    @app.get("/api/models")
    def models(svc: Svc) -> dict:
        """The configured models for the UI's model menu: the answer role's model, and per model whether
        it can be used (key set) and whether it runs outside the machine (only sees external_ok sources)."""
        registry = svc.current_models()
        catalogue = registry.catalogue
        return {"default": catalogue.roles["answer"], "fallback": catalogue.fallback,
                "models": [{"name": p.name, "model": p.model, "adapter": p.adapter, "location": p.location,
                            "ready": registry.ready(p.name)} for p in catalogue.models.values()]}

    @app.post("/api/chat")
    def chat(body: ChatRequest, svc: Svc, user: CurrentUser, conn: Db) -> StreamingResponse:
        """Answer a question in a conversation (new one if no session_id), streamed as server-sent events."""
        model = body.model or body.provider
        if model:
            try:
                svc.current_models().resolve(model)
            except LLMError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        if body.session_id:
            session = sessions.get_session(conn, body.session_id, user.user_id)
            if session is None:
                raise HTTPException(status_code=404, detail="session not found")
        else:
            session = sessions.create_session(conn, user)
        return StreamingResponse(stream_turn(svc, user, session, body.question, model, body.cache),
                                 media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.put("/api/sessions/{session_id}/release")
    def session_release(session_id: str, body: ReleaseRequest, user: CurrentUser, conn: Db) -> dict:
        """Set or clear the conversation's release filter (also changed by naming a release in a question)."""
        if sessions.get_session(conn, session_id, user.user_id) is None:
            raise HTTPException(status_code=404, detail="session not found")
        sessions.set_sticky_release(conn, session_id, body.release)
        return {"session_id": session_id, "sticky_release": body.release}

    @app.post("/api/feedback")
    def feedback(body: FeedbackRequest, user: CurrentUser, conn: Db) -> dict:
        """Rate one of the current user's answers; rating again replaces the earlier rating."""
        comment = body.comment.strip() if body.comment and body.comment.strip() else None
        saved = sessions.save_feedback(conn, body.trace_id, user.user_id, body.rating, comment)
        if saved is None:
            raise HTTPException(status_code=404, detail="answer not found")
        if body.rating < 0:     # a bad answer is not served from the cache again
            AnswerCache(conn).forget_trace(body.trace_id)
        return saved

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    def ui() -> FileResponse:
        """The chat page."""
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app
