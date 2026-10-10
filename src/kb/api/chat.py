"""One chat turn as a stream of server-sent events (POST /api/chat).

The turn runs in a worker thread (it blocks on the models and the LLM) and pushes events into a
queue; the HTTP response streams them as they arrive:

    session   {session_id, sticky_release}                       always first
    status    {stage: queued | condensing | searching, ...}      progress (queued: waiting for another turn)
              {stage: comparing | searching_side | reading, ...} a comparison: being split, each side
                                                                 searched ({side, index, total}), more
                                                                 sections read ({sections})
    context   {sources: [...]}                                   context sent to the LLM (for the UI panel)
    token     {text}                                             streamed answer pieces (raw model output)
    final     {answer, status, sources, references, cached, ...} the cleaned answer: replaces the streamed text
    error     {message}                                          instead of final, if the turn failed

Turn order: read the recent history → resolve the release (sticky per session) → condense a
follow-up into a standalone question → store the user message → answer (retrieve, gate, LLM,
citations) → store the assistant message with its trace id → final event.
"""

import json
import logging
import queue
import threading
from collections.abc import Callable, Iterator

from kb.answer.pipeline import Answer
from kb.api import sessions
from kb.api.services import Services
from kb.api.users import User
from kb.ingest.manifest import display_path
from kb.llm.condense import HISTORY_TURNS, condense, needs_condensing
from kb.llm.providers import LLMError
from kb.retrieve.assemble import ContextUnit
from kb.retrieve.pipeline import SearchRequest
from kb.retrieve.release import resolve_release

log = logging.getLogger(__name__)
_DONE = object()

Emit = Callable[[str, dict], None]


def sse(event: str, data: dict) -> str:
    """One server-sent event: 'event: <name>' and a JSON 'data:' line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def context_payload(context: list[ContextUnit]) -> dict:
    """The numbered context units, as the UI's context panel shows them."""
    return {"sources": [{"n": n, "citation": u.citation, "doc_id": u.doc_id, "title": u.title,
                         "section": u.section_number, "heading": u.heading, "pages": u.pages, "release": u.release,
                         "score": round(u.score, 4), "tokens": u.tokens, "text": u.text,
                         "path": display_path(u.source_path), "same_text": [s.citation for s in u.same_text]}
                        for n, u in enumerate(context, start=1)]}


def final_payload(answer: Answer, *, standalone: str | None, condense_reason: str, release: str | None,
                  release_reason: str, message_id: int) -> dict:
    """Everything the UI needs after the stream: the cleaned answer and how it was produced."""
    g = answer.generation
    return {
        "answer": answer.text, "status": answer.status, "refused_by": answer.refused_by,
        "gate": {"decision": answer.gate.decision, "top_score": answer.gate.top_score, "explain": answer.gate.explain()},
        "sources": [{"n": s.n, "line": s.line, "path": display_path(s.path), "doc_id": s.doc_id, "title": s.title,
                     "section": s.section,
                     "heading": s.heading, "page_start": s.page_start, "page_end": s.page_end, "release": s.release,
                     "same_text": s.same_text}
                    for s in answer.sources],
        "references": answer.references, "notices": answer.notices, "invalid_citations": answer.invalid_citations,
        "standalone_query": standalone, "condense_reason": condense_reason,
        "route": answer.route, "sides": answer.sides,
        "release": release, "release_reason": release_reason,
        "provider": g.provider if g else None, "model": g.model if g else None,
        "model_profile": answer.model_profile or None,
        "tokens_per_s": round(g.tokens_per_s, 1) if g and g.tokens_per_s else None,
        "timings_ms": answer.timings_ms, "trace_id": answer.trace_id, "message_id": message_id,
        "cached": bool(answer.cached_from),
    }


def run_turn(services: Services, user: User, session_id: str, question: str, provider: str | None,
             emit: Emit, use_cache: bool = True) -> None:
    """Answer one question in a session, reporting progress through `emit` (runs in a worker thread).
    use_cache=False: answer afresh instead of from the answer cache."""
    if not services.model_lock.acquire(blocking=False):   # one turn at a time on the shared models and GPU
        emit("status", {"stage": "queued"})
        services.model_lock.acquire()
    try:
        _run_turn_locked(services, user, session_id, question, provider, emit, use_cache)
    finally:
        services.model_lock.release()


def _run_turn_locked(services: Services, user: User, session_id: str, question: str, provider: str | None,
                     emit: Emit, use_cache: bool = True) -> None:
    """The turn itself, with the model lock held."""
    conn = services.connect()
    try:
        session = sessions.get_session(conn, session_id, user.user_id)
        history = sessions.recent_turns(conn, session_id, HISTORY_TURNS)

        choice = resolve_release(question, session["sticky_release"])
        if choice.sticky != session["sticky_release"]:
            sessions.set_sticky_release(conn, session_id, choice.sticky)

        if services.condenser is not None and needs_condensing(question, history):
            emit("status", {"stage": "condensing"})
        condensed = condense(services.condenser, question, history)
        standalone = condensed.question if condensed.condensed else None
        sessions.add_message(conn, session_id, "user", question, standalone_query=standalone)

        release = f"R{choice.release}x" if choice.release else None
        emit("status", {"stage": "searching", "standalone_query": standalone, "release": release,
                        "release_reason": choice.reason, "sticky_release": choice.sticky})
        pre_stages = [("condense", condensed.seconds * 1000,
                       {"reason": condensed.reason, "standalone": standalone})] if condensed.seconds else []
        request = SearchRequest(condensed.question, groups=user.search_groups, release=choice.release,
                                user_id=user.user_id, session_id=session_id)
        answer = services.make_answerer(conn).answer(
            request, provider=provider, asked=question, pre_stages=pre_stages,
            on_token=lambda piece: emit("token", {"text": piece}),
            on_context=lambda units: emit("context", context_payload(units)),
            on_status=lambda stage, data: emit("status", {"stage": stage, **data}), use_cache=use_cache)

        message_id = sessions.add_message(conn, session_id, "assistant", answer.text, trace_id=answer.trace_id)
        emit("final", final_payload(answer, standalone=standalone, condense_reason=condensed.reason,
                                    release=release, release_reason=choice.reason, message_id=message_id))
    except LLMError as e:
        emit("error", {"message": str(e)})
    except Exception as e:
        log.exception("chat turn failed")
        emit("error", {"message": f"{type(e).__name__}: {e}"})
    finally:
        conn.close()


def stream_turn(services: Services, user: User, session: dict, question: str,
                provider: str | None, use_cache: bool = True) -> Iterator[str]:
    """Server-sent events for one turn: the session event, then whatever the worker thread emits."""
    events: queue.Queue = queue.Queue()

    def work() -> None:
        """Run the turn, then signal the end of the stream."""
        try:
            run_turn(services, user, session["session_id"], question, provider, lambda e, d: events.put((e, d)),
                     use_cache)
        finally:
            events.put(_DONE)

    yield sse("session", {"session_id": session["session_id"], "sticky_release": session["sticky_release"]})
    threading.Thread(target=work, name="chat-turn", daemon=True).start()
    while (item := events.get()) is not _DONE:
        yield sse(*item)
