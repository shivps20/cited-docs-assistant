"""`kb search` and `kb ask`: retrieval and cited answers from the command line."""

import sys

from kb.core.config import get_settings


def _parse_groups(values: list[str] | None) -> list[str]:
    """Access groups from repeated and/or comma-separated --groups values (default: all)."""
    groups = [g.strip() for v in (values or []) for g in v.split(",") if g.strip()]
    return groups or ["all"]


def search_command(args) -> int:
    """`kb search`: run retrieval only and print the ranked chunks, the assembled context and timings."""
    import time

    from kb.core.db import connect
    from kb.ingest.manifest import parse_release
    from kb.retrieve.pipeline import Retriever, SearchRequest
    from kb.retrieve.rerank import BgeReranker
    from kb.store.embed import BgeM3Embedder
    from kb.store.vectorstore import get_client

    sys.stdout.reconfigure(encoding="utf-8")
    try:
        release = parse_release(args.release) if args.release else None
    except ValueError as e:
        print(e)
        return 1
    s = get_settings()
    start = time.perf_counter()
    embedder = BgeM3Embedder(device="cpu")
    reranker = None if args.no_rerank else BgeReranker(device="cpu", max_length=args.max_length)
    print(f"models loaded on CPU in {time.perf_counter() - start:.1f} s\n")

    conn = connect()
    retriever = Retriever(conn, get_client(), s.qdrant_collection, embedder, reranker)
    rerank_top = s.rerank_top if args.rerank_top is None else (args.rerank_top or None)
    request = SearchRequest(args.query, groups=_parse_groups(args.groups), release=release, mode=args.mode,
                            rerank=not args.no_rerank, candidates=args.candidates, rerank_top=rerank_top,
                            min_context_score=args.min_score)
    result = retriever.search(request)

    print(f"query: {args.query}")
    print(f"filters: groups={request.groups} release={args.release or 'any'} latest only; mode={args.mode}\n")
    print(f"{'#':>2} {'rerank':>6} {'retr':>6}  {'doc_id':<32} {'section':<10} {'pages':>7}  title")
    for rank, c in enumerate(result.candidates[:args.show], start=1):
        rr = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "-"
        title = c.header.rsplit(" > ", 1)[-1]
        print(f"{rank:>2} {rr:>6} {c.score:>6.3f}  {c.doc_id[:32]:<32} {c.section_number[:10]:<10} "
              f"{c.page_start:>3}-{c.page_end:<3}  {title[:60]}")

    print(f"\nassembled context: {len(result.context)} units, {sum(u.tokens for u in result.context)} tokens")
    for i, u in enumerate(result.context, start=1):
        print(f"  [{i}] {u.kind:<7} {u.tokens:>5} tok  {u.citation}")
        if args.context:
            print(f"\n{u.header}\n\n{u.text}\n")
    timings = "  ".join(f"{k} {v:.0f} ms" for k, v in result.timings_ms.items())
    print(f"\ntimings: {timings}  (trace {result.trace_id})")
    conn.close()
    return 0


def ask_command(args) -> int:
    """`kb ask`: answer a question with citations, streaming the answer, then print sources and timings."""
    import os
    import time

    os.environ["TQDM_DISABLE"] = "1"

    from kb.answer.pipeline import Answerer
    from kb.core.db import connect
    from kb.ingest.manifest import parse_release
    from kb.llm.prompts import REFERENCES_HEADING
    from kb.llm.providers import LLMError, build_providers
    from kb.retrieve.pipeline import Retriever, SearchRequest
    from kb.retrieve.rerank import BgeReranker
    from kb.store.embed import BgeM3Embedder
    from kb.store.vectorstore import get_client

    sys.stdout.reconfigure(encoding="utf-8")
    try:
        release = parse_release(args.release) if args.release else None
    except ValueError as e:
        print(e)
        return 1
    s = get_settings()
    start = time.perf_counter()
    embedder = BgeM3Embedder(device="cpu")
    reranker = None if args.no_rerank else BgeReranker(device="cpu")
    print(f"models loaded on CPU in {time.perf_counter() - start:.1f} s\n")

    conn = connect()
    retriever = Retriever(conn, get_client(), s.qdrant_collection, embedder, reranker)
    answerer = Answerer(conn, retriever, build_providers(s), not_found_score=s.not_found_score,
                        provider=args.provider or s.llm_provider)
    rerank_top = s.rerank_top if args.rerank_top is None else (args.rerank_top or None)
    request = SearchRequest(args.query, groups=_parse_groups(args.groups), release=release, mode=args.mode,
                            rerank=not args.no_rerank, rerank_top=rerank_top, user_id="cli")
    print(f"question: {args.query}")
    print(f"filters: groups={request.groups} release={args.release or 'any'}\n")

    streamed = False

    def on_token(piece: str) -> None:
        """Print each streamed piece of the answer as it arrives."""
        nonlocal streamed
        streamed = True
        print(piece, end="", flush=True)

    def on_context(context) -> None:
        """Print the numbered context units before generation starts (--show-context)."""
        print("--- context sent to the LLM ---")
        for i, u in enumerate(context, start=1):
            print(f"\n[{i}] {u.citation}  ({u.tokens} tokens)\n{u.text}")
        print("\n--- end of context ---\n")

    try:
        answer = answerer.answer(request, on_token=None if args.no_stream else on_token,
                                 on_context=on_context if args.show_context else None)
    except LLMError as e:
        print(f"\nERROR {e}")
        conn.close()
        return 1

    print("\n" if streamed else answer.text + "\n")
    if streamed and answer.references:   # added after generation, so not part of the stream
        print("\n".join([REFERENCES_HEADING, *answer.references]) + "\n")
    if answer.sources:
        print("Sources:")
        for src in answer.sources:
            print(f"  {src.line}")
        print()
    for notice in answer.notices:
        print(f"NOTE {notice}")
    if answer.invalid_citations:
        print(f"NOTE removed citation(s) {answer.invalid_citations}: not among the {len(answer.context)} sources")

    t = answer.timings_ms
    retrieval_ms = sum(t.get(k, 0) for k in ("embed", "search", "rerank", "assemble"))
    status = answer.status if answer.status == "answered" else f"not found (by {answer.refused_by})"
    print(f"\nstatus: {status}   gate: {answer.gate.explain()}")
    line = f"retrieval {retrieval_ms / 1000:.1f} s (rerank {t.get('rerank', 0) / 1000:.1f} s)"
    g = answer.generation
    if g:
        speed = f", {g.tokens_per_s:.0f} tok/s" if g.tokens_per_s else ""
        load = f", model load {g.load_seconds:.1f} s" if g.load_seconds and g.load_seconds > 0.5 else ""
        line += (f" | generation {t.get('generate', 0) / 1000:.1f} s with {g.provider} {g.model} "
                 f"({g.prompt_tokens} prompt + {g.output_tokens} answer tokens{speed}{load})")
    print(f"timings: {line} | total {sum(t.values()) / 1000:.1f} s")
    print(f"trace: {answer.trace_id}")
    conn.close()
    return 0
