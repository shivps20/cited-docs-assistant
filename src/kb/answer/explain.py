"""Behind the scenes of one answer, from its trace: candidates with retrieval and rerank scores, the gate, the
context and the exact prompt sent to the model, the checks, the timings, and an on-demand faithfulness check.

Used by `kb trace` and by the chat's "Behind the scenes" panel (GET /api/traces/{id},
POST /api/traces/{id}/faithfulness). Everything is rebuilt from the `traces` / `trace_stages` rows, so it works
for any traced answer: the generate stage stores the context units it sent (records below) and a hash of the
messages, and the prompt is rebuilt from them with the current prompt code; when the code has changed since,
the view says so. Traces written before this module show what they recorded (no prompt, top 10 rerank scores).
"""

import json
import sqlite3
import time

from kb.answer.records import messages_hash, unit_from_record
from kb.core.tracing import append_stage
from kb.evaluation.answers import unverified_commands
from kb.llm.judge import judge_faithfulness
from kb.llm.prompts import build_messages
from kb.llm.providers import LLMError

JUDGE_MAX_TOKENS = 3000      # as in kb eval-answers: the judge lists every claim with a quote


def load_trace(conn: sqlite3.Connection, trace_id: str) -> tuple[sqlite3.Row, list[dict]] | None:
    """The trace row and its stages (data parsed from JSON), or None when there is no such trace."""
    row = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
    if row is None:
        return None
    stages = [{"seq": r["seq"], "stage": r["stage"], "duration_ms": r["duration_ms"],
               "data": json.loads(r["data"] or "{}")}
              for r in conn.execute("SELECT seq, stage, duration_ms, data FROM trace_stages WHERE trace_id = ? "
                                    "ORDER BY seq", (trace_id,))]
    return row, stages


def latest_trace_id(conn: sqlite3.Connection, user_id: str | None = None) -> str | None:
    """The most recent trace (of one user, when given)."""
    sql, args = "SELECT trace_id FROM traces", ()
    if user_id is not None:
        sql, args = sql + " WHERE user_id = ?", (user_id,)
    row = conn.execute(sql + " ORDER BY created_at DESC LIMIT 1", args).fetchone()
    return row["trace_id"] if row else None


def _section_of(chunk_id: str) -> str:
    """'doc#2.1#0' -> 'doc#2.1'."""
    return chunk_id.rsplit("#", 1)[0]


def _candidates(conn: sqlite3.Connection, stages: list[dict], context_sections: set[str]) -> list[dict]:
    """Every search hit with its retrieval (fusion) score and rank, its rerank score and rank when reranked,
    whether its section reached the context, and where it is (document, section, pages)."""
    out: list[dict] = []
    reranks: dict[tuple[str, str], tuple[int, float]] = {}
    for s in stages:
        if s["stage"] == "rerank":
            side = s["data"].get("side", "")
            for rank, (chunk_id, score) in enumerate(s["data"].get("top", []), start=1):
                reranks[(side, chunk_id)] = (rank, score)
    for s in stages:
        if s["stage"] != "search":
            continue
        side = s["data"].get("side", "")
        for rank, (chunk_id, score) in enumerate(s["data"].get("hits", []), start=1):
            rr = reranks.get((side, chunk_id))
            out.append({"side": side, "chunk_id": chunk_id, "retrieval_rank": rank, "retrieval_score": score,
                        "rerank_rank": rr[0] if rr else None, "rerank_score": rr[1] if rr else None,
                        "in_context": _section_of(chunk_id) in context_sections})
    if out:
        ids = sorted({c["chunk_id"] for c in out})
        rows = {}
        for i in range(0, len(ids), 500):
            batch = ids[i:i + 500]
            rows.update({r["chunk_id"]: r for r in conn.execute(
                "SELECT c.chunk_id, c.page_start, c.page_end, c.header, d.title, d.doc_type FROM chunks c "
                f"JOIN documents d ON d.doc_id = c.doc_id WHERE c.chunk_id IN ({','.join('?' * len(batch))})", batch)})
        for c in out:
            r = rows.get(c["chunk_id"])
            c.update(title=r["title"] if r else None, section=r["header"].split(" > ", 1)[-1] if r else None,
                     page_start=r["page_start"] if r else None, page_end=r["page_end"] if r else None,
                     doc_type=r["doc_type"] if r else None)
    # reranked first in rerank order, then the rest in search order (per side)
    return sorted(out, key=lambda c: (c["side"], c["rerank_rank"] is None, c["rerank_rank"] or c["retrieval_rank"]))


def _generation(stage: dict, question: str) -> dict:
    """One generate stage: model and usage, the context units and the prompt rebuilt from them."""
    d = stage["data"]
    units = [unit_from_record(r) for r in d.get("context", [])]
    gen = {k: d.get(k) for k in ("profile", "provider", "model", "prompt_tokens", "output_tokens", "tokens_per_s",
                                 "load_s", "prompt_chars", "fallback_from", "context_fitted", "raw_output", "error")}
    gen["duration_ms"] = stage["duration_ms"]
    gen["units"] = [{k: v for k, v in r.items() if k != "text"} | {"chars": len(r.get("text") or "")}
                    for r in d.get("context", [])]
    if units:
        messages = build_messages(d.get("question", question), units, d.get("sides") or ())
        gen["messages"] = messages
        gen["prompt_identical"] = d.get("prompt_sha") == messages_hash(messages) if d.get("prompt_sha") else None
    else:
        gen["messages"] = None          # traced before the context was stored with the generate stage
        gen["prompt_identical"] = None
    return gen


def explain_trace(conn: sqlite3.Connection, trace_id: str) -> dict | None:
    """Everything recorded about one answer, ready to show; None when the trace does not exist."""
    loaded = load_trace(conn, trace_id)
    if loaded is None:
        return None
    row, stages = loaded
    by_name: dict[str, list[dict]] = {}
    for s in stages:
        by_name.setdefault(s["stage"], []).append(s)
    question = row["standalone_query"] or row["query"]
    generations = [_generation(s, question) for s in by_name.get("generate", [])]
    final_units = list(next((g["units"] for g in reversed(generations) if g["units"]), []))
    context_sections = {u["section_id"] for u in final_units}
    if not context_sections:                       # older traces: the assemble stage lists the units
        listed = [u for s in by_name.get("assemble", []) for u in s["data"].get("units", [])]
        context_sections = {u[0] for u in listed}
        for section_id, kind, tokens in listed:
            r = conn.execute("SELECT s.heading_path, s.page_start, s.page_end, d.title, d.doc_type FROM sections s "
                             "JOIN documents d ON d.doc_id = s.doc_id WHERE s.section_id = ?", (section_id,)).fetchone()
            final_units.append({"section_id": section_id, "kind": kind, "tokens": tokens,
                                "title": r["title"] if r else section_id, "heading_path": r["heading_path"] if r else "",
                                "page_start": r["page_start"] if r else None, "page_end": r["page_end"] if r else None,
                                "doc_type": r["doc_type"] if r else None, "from_assemble": True})
    gate = by_name.get("gate", [{}])[-1].get("data", {})
    cite = by_name.get("cite", [{}])[-1].get("data", {})
    cache = by_name.get("cache", [{}])[-1].get("data", {})
    judge = by_name.get("judge", [{}])[-1].get("data") if "judge" in by_name else None

    commands = None
    final = next((s for s in reversed(by_name.get("generate", [])) if s["data"].get("context")), None)
    if row["answer"] and final is not None:
        context_text = "\n".join(r.get("text") or "" for r in final["data"]["context"])
        total, unverified = unverified_commands(row["answer"], context_text)
        commands = {"total": total, "unverified": unverified}

    return {
        "trace_id": row["trace_id"], "created_at": row["created_at"], "user_id": row["user_id"],
        "session_id": row["session_id"], "query": row["query"], "standalone_query": row["standalone_query"],
        "route": row["route"], "release_filter": row["release_filter"], "answer": row["answer"],
        "sources": json.loads(row["sources"]) if row["sources"] else [],
        "model": {"provider": row["llm_provider"], "model": row["llm_model"]},
        "cache": {"hit": bool(row["cache_hit"]), "served_from": cache.get("served_from")},
        "gate": {"decision": row["gate_decision"] or gate.get("decision"), "top_score": row["top_rerank_score"],
                 "threshold": gate.get("threshold")},
        "filters": next(({k: s["data"].get(k) for k in ("mode", "limit", "groups", "release", "category")}
                         for s in by_name.get("search", [])), None),
        "candidates": _candidates(conn, stages, context_sections),
        "rerank": [{k: s["data"].get(k) for k in ("candidates", "top", "max_length", "side")} | {"stored": len(s["data"].get("top", []))}
                   for s in by_name.get("rerank", [])],
        "context": final_units,
        "generations": generations,
        "checks": {"cited": cite.get("cited"), "invalid_citations": cite.get("invalid"), "refused": cite.get("refused"),
                   "retry": by_name.get("retry", [{}])[-1].get("data") if "retry" in by_name else None,
                   "read": by_name.get("read", [{}])[-1].get("data") if "read" in by_name else None,
                   "commands": commands},
        "faithfulness": judge,
        "timings": [{"seq": s["seq"], "stage": s["stage"], "duration_ms": round(s["duration_ms"], 2),
                     "side": s["data"].get("side")} for s in stages],
        "total_ms": row["total_ms"], "error": row["error"],
    }


def check_faithfulness(conn: sqlite3.Connection, models, trace_id: str, *, force: bool = False) -> dict:
    """Run the faithfulness judge on a traced answer against the context it was given, store the result as a
    'judge' stage of the trace, and return it. An earlier result is returned unless force. The judge model is
    the catalogue's judge role, or the local fallback when the context may not leave the machine (external_ok)."""
    loaded = load_trace(conn, trace_id)
    if loaded is None:
        raise LookupError(f"no trace {trace_id}")
    row, stages = loaded
    if not force:
        earlier = [s for s in stages if s["stage"] == "judge"]
        if earlier:
            return earlier[-1]["data"]
    final = next((s for s in reversed(stages) if s["stage"] == "generate" and s["data"].get("context")), None)
    if not row["answer"] or final is None:
        raise ValueError("this answer has no stored context to check (refused, served from the cache, or traced "
                         "before contexts were stored)")
    units = [unit_from_record(r) for r in final["data"]["context"]]
    role = models.catalogue.for_role("judge")
    note = None
    if role.external and not all(u.external_ok for u in units):
        # the privacy policy applies to the judge as to answering: internal text stays on the machine
        note = f"judged locally: the context may not be sent to {role.name}"
        name, judge = models.catalogue.fallback, models.provider(models.catalogue.fallback)
    else:
        name = role.name if models.ready(role.name) else models.catalogue.fallback
        judge = models.for_role("judge", max_output_tokens=JUDGE_MAX_TOKENS)
    start = time.perf_counter()
    try:
        verdict = judge_faithfulness(judge, row["answer"], units)
    except LLMError as e:
        verdict = None
        error = str(e)
    data = {"model": name, "note": note}
    if verdict is None:
        data |= {"faithfulness": None, "error": error, "claims": []}
    else:
        data |= {"faithfulness": verdict.faithfulness, "error": verdict.error,
                 "claims": [{"claim": c.text, "supported": c.supported, "source": c.source, "evidence": c.evidence,
                             "evidence_found": c.evidence_found, "by_values": c.by_values} for c in verdict.claims]}
    append_stage(conn, trace_id, "judge", (time.perf_counter() - start) * 1000, data)
    return data
