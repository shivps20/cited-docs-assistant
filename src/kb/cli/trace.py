"""`kb trace`: behind the scenes of one answer (candidates and scores, gate, context, prompt, checks, timings)."""

import json
import sys


def _fmt(value, spec: str = ".4f") -> str:
    """A number for the table, or '-' when missing."""
    return "-" if value is None else format(value, spec)


def _pages(c: dict) -> str:
    """'p. 4-6' / 'slide 9' / '' (Word) for a candidate or unit."""
    start, end, kind = c.get("page_start"), c.get("page_end"), c.get("doc_type")
    if start is None or kind == "docx":
        return ""
    unit = "slide" if kind == "pptx" else "p."
    return f"{unit} {start}" if start == end else f"{unit} {start}-{end}"


def trace_command(args) -> int:
    """Print the trace view of one answer (default: the latest trace); --faithfulness runs the judge on it."""
    from kb.answer.explain import check_faithfulness, explain_trace, latest_trace_id
    from kb.core.db import connect

    sys.stdout.reconfigure(encoding="utf-8")
    conn = connect()
    trace_id = latest_trace_id(conn) if args.trace_id in (None, "last") else args.trace_id
    if trace_id is None:
        print("no traces yet")
        return 1
    if len(trace_id) < 32:                       # a prefix, as shown in the chat ("trace 7b93a3da")
        rows = conn.execute("SELECT trace_id FROM traces WHERE trace_id LIKE ? LIMIT 2", (trace_id + "%",)).fetchall()
        if len(rows) != 1:
            print(f"{'no' if not rows else 'more than one'} trace starting with {trace_id!r}")
            return 1
        trace_id = rows[0]["trace_id"]
    if args.faithfulness:
        from kb.llm.registry import ModelRegistry

        print("checking faithfulness (one LLM call, 20-60 s) …", flush=True)
        try:
            check_faithfulness(conn, ModelRegistry.load(), trace_id, force=args.force)
        except (LookupError, ValueError) as e:
            print(e)
            return 1
    view = explain_trace(conn, trace_id)
    conn.close()
    if view is None:
        print(f"no trace {trace_id}")
        return 1
    if args.json:
        print(json.dumps(view, ensure_ascii=False, indent=2, default=str))
        return 0
    _print(view, show_all=args.all, show_prompt=args.prompt)
    return 0


def _print(v: dict, *, show_all: bool, show_prompt: bool) -> None:
    """The trace view as text sections."""
    print(f"trace {v['trace_id']}  {v['created_at']}  user {v['user_id']}")
    print(f"question: {v['query']}")
    if v["standalone_query"]:
        print(f"searched as: {v['standalone_query']}")
    f = v["filters"] or {}
    print(f"route {v['route']} · release {v['release_filter'] or 'any'} · groups {f.get('groups')} · "
          f"model {v['model']['model'] or '-'} · {v['total_ms'] / 1000:.1f} s" if v["total_ms"] else "")
    if v["cache"]["hit"]:
        print(f"served from the answer cache (original answer: trace {v['cache']['served_from']}); "
              f"run `kb trace {v['cache']['served_from']}` for its details")
    g = v["gate"]
    print(f"\n== gate: {g['decision']} (top rerank score {_fmt(g['top_score'], '.3f')}, threshold {g['threshold']})")

    cands = v["candidates"] if show_all else [c for c in v["candidates"] if c["rerank_rank"] or c["in_context"]]
    hidden = len(v["candidates"]) - len(cands)
    print(f"\n== candidates ({len(v['candidates'])} from search; reranked first"
          f"{f', {hidden} not reranked hidden (--all)' if hidden else ''})")
    print(f"{'side':<8} {'rr#':>3} {'rerank':>7} {'ret#':>4} {'retrieval':>9} {'ctx':>3}  location")
    for c in cands:
        where = " · ".join(filter(None, [c.get("title"), c.get("section"), _pages(c)])) or c["chunk_id"]
        print(f"{(c['side'] or '-')[:8]:<8} {c['rerank_rank'] or '':>3} {_fmt(c['rerank_score']):>7} "
              f"{c['retrieval_rank']:>4} {_fmt(c['retrieval_score']):>9} {'yes' if c['in_context'] else '':>3}  "
              f"{where[:110]}")
    for r in v["rerank"]:
        if r["stored"] < (r["candidates"] or 0) and r["stored"] <= 10:
            print("  (traced before all rerank scores were stored: only the top 10 are known)")

    print(f"\n== context sent to the LLM ({len(v['context'])} units)")
    for n, u in enumerate(v["context"], start=1):
        print(f"[{n}] {u['title']} · {u['heading_path']} · {_pages(u)} · {u['kind']} · {u['tokens']} tokens"
              f"{' · side ' + u['side'] if u.get('side') else ''}")

    for i, gen in enumerate(v["generations"], start=1):
        label = "generation" if len(v["generations"]) == 1 else f"generation {i} of {len(v['generations'])}"
        print(f"\n== {label}: {gen['profile']} ({gen['model']}) · prompt {gen['prompt_tokens']} tokens · "
              f"answer {gen['output_tokens']} tokens · {_fmt(gen['tokens_per_s'], '.1f')} tok/s · "
              f"{gen['duration_ms'] / 1000:.1f} s")
        if gen.get("fallback_from"):
            print(f"  fell back from: {', '.join(gen['fallback_from'])}")
        if gen.get("context_fitted"):
            print(f"  context fitted to the model: {gen['context_fitted'][0]} → {gen['context_fitted'][1]} units")
        if gen["messages"] is None:
            print("  prompt: not stored (traced before prompts were kept)")
        else:
            same = {True: "identical to what was sent", False: "rebuilt with today's prompt code: it has changed "
                    "since this answer", None: "rebuilt"}[gen["prompt_identical"]]
            print(f"  prompt: {sum(len(m['content']) for m in gen['messages'])} characters, {same}"
                  f"{'' if show_prompt else ' (--prompt shows it)'}")
            if show_prompt:
                for m in gen["messages"]:
                    print(f"\n--- {m['role']} ---\n{m['content']}")

    c = v["checks"]
    print("\n== checks")
    print(f"citations kept {c['cited']} · removed (no such source) {c['invalid_citations'] or []} · "
          f"refused {c['refused']}")
    if c["retry"]:
        print(f"refusal retry with {len(c['retry'].get('units', []))} units")
    if c["commands"]:
        bad = c["commands"]["unverified"]
        print(f"commands in the answer: {c['commands']['total']}, not found word for word in the context: {len(bad)}"
              + "".join(f"\n  - {b}" for b in bad))
    fa = v["faithfulness"]
    if fa is None:
        print("faithfulness: not checked (kb trace --faithfulness)")
    elif fa.get("error"):
        print(f"faithfulness: failed ({fa['error']})")
    else:
        claims = fa.get("claims", [])
        print(f"faithfulness: {_fmt(fa['faithfulness'], '.2f')} ({sum(x['supported'] for x in claims)} of "
              f"{len(claims)} claims supported; judge {fa['model']}{'; ' + fa['note'] if fa.get('note') else ''})")
        for x in claims:
            if not x["supported"]:
                print(f"  unsupported: {x['claim']}")

    print("\n== timings")
    print("  " + " · ".join(f"{t['stage']}{'(' + t['side'] + ')' if t.get('side') else ''} "
                            f"{t['duration_ms'] / 1000:.2f} s" if t["duration_ms"] >= 100 else
                            f"{t['stage']} {t['duration_ms']:.0f} ms" for t in v["timings"]))
