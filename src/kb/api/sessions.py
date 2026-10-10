"""Chat sessions and their messages in SQLite (tables `sessions` and `messages`).

A session belongs to one user; lookups always filter by user, so one user can never read another
user's conversation. The user's groups are stored as a snapshot when the session starts (for
auditing); access checks always use the current users.yaml.
"""

import json
import sqlite3
import uuid

from kb.api.users import User
from kb.ingest.manifest import display_path

TITLE_CHARS = 80    # sessions are listed by their first question, cut to this length
# Why an answer got a thumbs down: the last two say the "not found" decision was wrong (kb calibrate).
FEEDBACK_REASONS = ("wrong", "incomplete", "should_have_answered", "should_have_refused")

_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


def create_session(conn: sqlite3.Connection, user: User) -> dict:
    """Start a new session for `user` and return it."""
    session_id = uuid.uuid4().hex
    with conn:
        conn.execute("INSERT INTO sessions (session_id, user_id, user_groups) VALUES (?, ?, ?)",
                     (session_id, user.user_id, json.dumps(user.search_groups)))
    return get_session(conn, session_id, user.user_id)


def get_session(conn: sqlite3.Connection, session_id: str, user_id: str) -> dict | None:
    """The session with its messages, or None if it does not exist or belongs to another user."""
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id)).fetchone()
    if row is None:
        return None
    session = dict(row)
    session["user_groups"] = json.loads(session["user_groups"])
    session["messages"] = list_messages(conn, session_id, user_id)
    return session


def list_sessions(conn: sqlite3.Connection, user_id: str, limit: int = 30) -> list[dict]:
    """The user's sessions, most recently active first, titled by their first question."""
    rows = conn.execute(f"""
        SELECT s.session_id, s.sticky_release, s.created_at, s.last_active_at,
               (SELECT content FROM messages m WHERE m.session_id = s.session_id AND m.role = 'user'
                ORDER BY m.id LIMIT 1) AS first_question,
               (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) AS message_count
        FROM sessions s WHERE s.user_id = ? ORDER BY s.last_active_at DESC, s.rowid DESC LIMIT {int(limit)}
    """, (user_id,)).fetchall()
    sessions = []
    for r in rows:
        title = (r["first_question"] or "New conversation").strip()
        sessions.append({**dict(r), "title": title if len(title) <= TITLE_CHARS else title[:TITLE_CHARS - 1] + "…"})
    return sessions


def list_messages(conn: sqlite3.Connection, session_id: str, user_id: str | None = None) -> list[dict]:
    """All messages of a session in order. Assistant messages also carry their sources and gate decision
    (from the trace) and, when user_id is given, that user's feedback rating, reason and comment."""
    rows = conn.execute("""
        SELECT m.id, m.role, m.content, m.standalone_query, m.trace_id, m.created_at,
               t.sources, t.gate_decision, f.rating, f.reason, f.comment
        FROM messages m
        LEFT JOIN traces t ON t.trace_id = m.trace_id
        LEFT JOIN feedback f ON f.trace_id = m.trace_id AND f.user_id = ?
        WHERE m.session_id = ? ORDER BY m.id
    """, (user_id, session_id)).fetchall()
    messages = []
    for r in rows:
        m = dict(r)
        m["sources"] = json.loads(m["sources"]) if m["sources"] else []
        for source in m["sources"]:                         # stored in the manifest form; shown per KB_SOURCE_PATH
            if isinstance(source, dict) and source.get("path"):
                source["path"] = display_path(source["path"])
        messages.append(m)
    return messages


def add_message(conn: sqlite3.Connection, session_id: str, role: str, content: str, *,
                standalone_query: str | None = None, trace_id: str | None = None) -> int:
    """Append a message (role 'user' or 'assistant') and mark the session active; returns its id."""
    with conn:
        cur = conn.execute("INSERT INTO messages (session_id, role, content, standalone_query, trace_id) "
                           "VALUES (?, ?, ?, ?, ?)", (session_id, role, content, standalone_query, trace_id))
        conn.execute(f"UPDATE sessions SET last_active_at = {_NOW} WHERE session_id = ?", (session_id,))
    return cur.lastrowid


def recent_turns(conn: sqlite3.Connection, session_id: str, turns: int = 6) -> list[dict]:
    """The last `turns` messages (oldest first), the conversation history for follow-up questions."""
    rows = conn.execute("SELECT role, content, standalone_query FROM messages WHERE session_id = ? "
                        "ORDER BY id DESC LIMIT ?", (session_id, turns)).fetchall()
    return [dict(r) for r in reversed(rows)]


def save_feedback(conn: sqlite3.Connection, trace_id: str, user_id: str, rating: int,
                  comment: str | None = None, reason: str | None = None) -> dict | None:
    """Store (or replace) the user's rating of one answer; None if the trace is not that user's answer.
    reason (one of FEEDBACK_REASONS) is kept only with a thumbs down; a thumbs up clears it."""
    owner = conn.execute("SELECT user_id FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
    if owner is None or owner["user_id"] != user_id:
        return None
    if reason is not None and reason not in FEEDBACK_REASONS:
        raise ValueError(f"unknown feedback reason {reason!r}")
    reason = reason if rating < 0 else None
    with conn:
        conn.execute(f"""
            INSERT INTO feedback (trace_id, user_id, rating, comment, reason) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (trace_id, user_id) DO UPDATE SET rating = excluded.rating, comment = excluded.comment,
                reason = excluded.reason, created_at = {_NOW}
        """, (trace_id, user_id, rating, comment, reason))
    return {"trace_id": trace_id, "rating": rating, "reason": reason, "comment": comment}


def set_sticky_release(conn: sqlite3.Connection, session_id: str, release: str | None) -> None:
    """Remember (or clear, with None) the release that filters the rest of the session, e.g. 'R2025x'."""
    with conn:
        conn.execute("UPDATE sessions SET sticky_release = ? WHERE session_id = ?", (release, session_id))
