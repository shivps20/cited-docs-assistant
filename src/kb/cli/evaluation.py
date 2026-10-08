"""`kb eval` and `kb eval-answers`: retrieval and answer evaluation against the golden set."""

import sys

from kb.core.config import get_settings
from kb.ingest.manifest import load_manifest


def eval_command(args) -> int:
    """`kb eval`: run the golden questions through each retrieval configuration and print the metrics."""
    import json as _json
    import os
    import time

    os.environ["TQDM_DISABLE"] = "1"   # no per-call progress bars from FlagEmbedding

    from kb.core.db import connect
    from kb.evaluation.retrieval import (
        CONFIGS,
        full_access,
        load_golden,
        restricted_docs,
        run_eval,
    )
    from kb.ingest.manifest import RELEASE_ANY_MIN
    from kb.retrieve.pipeline import Retriever, SearchRequest
    from kb.retrieve.rerank import BgeReranker
    from kb.retrieve.search import PUBLIC_GROUP
    from kb.store.embed import BgeM3Embedder
    from kb.store.vectorstore import get_client

    sys.stdout.reconfigure(encoding="utf-8")
    names = [n.strip() for n in args.configs.split(",")] if args.configs else [c.name for c in CONFIGS]
    unknown = set(names) - {c.name for c in CONFIGS}
    if unknown:
        print(f"unknown config(s): {', '.join(sorted(unknown))}")
        return 1
    configs = [c for c in CONFIGS if c.name in names]
    golden = load_golden([q.strip() for q in args.questions.split(",")] if args.questions else None)

    s = get_settings()
    docs = load_manifest(s.manifest_path, s.docs_dir)
    access = full_access(docs)
    start = time.perf_counter()
    embedder = BgeM3Embedder(device="cpu")
    reranker = BgeReranker(device="cpu") if any(c.rerank for c in configs) else None
    print(f"models loaded on CPU in {time.perf_counter() - start:.1f} s; "
          f"{len(golden)} questions x {len(configs)} configs\n", flush=True)
    conn = connect()
    client = get_client()
    retriever = Retriever(conn, client, s.qdrant_collection, embedder, reranker)

    def search_fn(question, config, groups=None, release=None):
        """Run one golden question with one configuration (all groups and no release filter by default)."""
        if reranker is not None:
            reranker.max_length = config.max_length
        return retriever.search(SearchRequest(question, groups=list(groups or access), release=release, mode=config.mode,
                                              rerank=config.rerank, rerank_top=config.rerank_top, user_id="eval"))

    run_start = {}

    def progress(msg: str) -> None:
        """Print one dot per question."""
        name = msg.split(":")[0]
        if name not in run_start:
            run_start[name] = time.perf_counter()
            print(f"running {name} ", end="", flush=True)
        print(".", end="", flush=True)
        if msg.endswith(golden[-1]["id"]):
            print(f" {time.perf_counter() - run_start[name]:.0f} s", flush=True)

    report = run_eval(search_fn, configs, golden, progress)

    print(f"\n{'config':<17} {'R@1':>5} {'R@5':>5} {'R@10':>5} {'MRR':>5} {'ctx':>5} {'all@5':>5} "
          f"{'p50 ms':>7} {'p95 ms':>7} {'rerank p50':>10}")
    for name, r in report.items():
        m = r["summary"]
        print(f"{name:<17} {m['recall@1']:>5.2f} {m['recall@5']:>5.2f} {m['recall@10']:>5.2f} {m['mrr']:>5.2f} "
              f"{m['context_recall']:>5.2f} {m['all_docs@5']:>5.2f} {m['latency_ms_p50']:>7.0f} "
              f"{m['latency_ms_p95']:>7.0f} {m['rerank_ms_p50']:>10.0f}")

    print("\nRecall@5 / MRR by question type")
    for name, r in report.items():
        parts = [f"{t} {v['recall@5']:.2f}/{v['mrr']:.2f} (n={v['n']})" for t, v in r["summary"]["by_type"].items()]
        print(f"  {name:<17} " + "   ".join(parts))

    reranked = [(n, r) for n, r in report.items() if r["config"]["rerank"]]
    if reranked:
        print("\nTop rerank score: answerable (rank-1 hit) vs unanswerable")
        for name, r in reranked:
            m = r["summary"]
            ok, bad = m["correct_top_score"], m["unanswerable_top_score"]
            fmt = lambda v: "-" if v is None else f"{v:.3f}"
            print(f"  {name:<17} answerable min {fmt(ok['min'])} median {fmt(ok['median'])} | "
                  f"unanswerable max {fmt(bad['max'])}  {[round(x, 3) for x in bad['scores']]}")

    if args.misses:
        for name, r in report.items():
            misses = [q for q in r["questions"] if q["qtype"] != "unanswerable"
                      and (q["first_hit"] is None or q["first_hit"] > 5)]
            print(f"\nMisses (no hit in top 5) for {name}: {len(misses)}")
            for q in misses:
                print(f"  {q['qid']} first hit: {q['first_hit'] or '-'}  top: {', '.join(q['top_results'][:3])}")

    # Access and release filtering checks, derived from the manifest and the golden set.
    hybrid = next(c for c in CONFIGS if c.name == "hybrid")
    all_golden = load_golden()
    hidden = {d.doc_id for d in restricted_docs(docs)}
    probe = next((q for q in all_golden if any(src["doc_id"] in hidden for src in q["sources"])), None)
    acl_ok, acl_note = True, "skipped: no golden question cites a restricted document"
    if probe:
        leaked = sorted({c.doc_id for c in search_fn(probe["question"], hybrid, groups=[PUBLIC_GROUP]).candidates}
                        & hidden)
        acl_ok = not leaked
        acl_note = f"{probe['id']} as a public user: restricted documents " + \
                   (f"LEAKED ({', '.join(leaked)})" if leaked else "hidden")
    years = sorted({d.release_min for d in docs if d.release_min > RELEASE_ANY_MIN})
    first = next((q for q in all_golden if q["sources"]), None)
    rel_ok, rel_note = True, "skipped: no release ranges in the manifest"
    if years and first:
        year = years[len(years) // 2]
        rel = search_fn(first["question"], hybrid, release=year)
        rel_ok = all(c.payload["release_min"] <= year <= c.payload["release_max"] for c in rel.candidates)
        rel_note = f"{first['id']} filtered to R{year}x: only documents that apply to it"
    print(f"\nchecks: access filter ({acl_note}) {'PASS' if acl_ok else 'FAIL'}; "
          f"release filter ({rel_note}) {'PASS' if rel_ok else 'FAIL'}")

    out_dir = s.db_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"retrieval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(_json.dumps(report, indent=2), encoding="utf-8")
    print(f"full report: {out}")
    conn.close()
    return 0 if acl_ok and rel_ok else 1


def eval_answers_command(args) -> int:
    """`kb eval-answers`: answer every golden question, score the answers, optionally judge them."""
    import hashlib
    import json as _json
    import time

    from kb.answer.pipeline import Answerer
    from kb.core.db import connect
    from kb.evaluation.answers import run_answer_eval
    from kb.evaluation.retrieval import full_access, load_golden
    from kb.llm.judge import judge_faithfulness
    from kb.llm.prompts import system_prompt
    from kb.llm.providers import LLMError, OllamaProvider, build_providers
    from kb.retrieve.pipeline import Retriever, SearchRequest
    from kb.retrieve.rerank import BgeReranker
    from kb.store.embed import BgeM3Embedder
    from kb.store.vectorstore import get_client

    sys.stdout.reconfigure(encoding="utf-8")
    golden = load_golden([q.strip() for q in args.questions.split(",")] if args.questions else None)
    if args.types:
        types = {t.strip() for t in args.types.split(",")}
        golden = [q for q in golden if q["type"] in types]
    if not golden:
        print("no golden questions selected")
        return 1

    s = get_settings()
    access = full_access(load_manifest(s.manifest_path, s.docs_dir))   # answer as a user who sees everything
    start = time.perf_counter()
    embedder, reranker = BgeM3Embedder(device="cpu"), BgeReranker(device="cpu")
    print(f"models loaded on CPU in {time.perf_counter() - start:.1f} s; {len(golden)} questions; "
          f"judge {'off' if args.no_judge else 'on'}\n", flush=True)
    conn = connect()
    retriever = Retriever(conn, get_client(), s.qdrant_collection, embedder, reranker)
    providers = build_providers(s)
    answerer = Answerer(conn, retriever, providers, not_found_score=s.not_found_score,
                        provider=args.provider or s.llm_provider)
    judge = None if args.no_judge else OllamaProvider(
        s.ollama_host, s.llm_model, num_ctx=s.llm_num_ctx, keep_alive=s.llm_keep_alive, temperature=0.0,
        max_tokens=1500)

    def answer_fn(question: str):
        """Answer one golden question with full access and no release filter."""
        return answerer.answer(SearchRequest(question, groups=access, user_id="eval"))

    def judge_fn(answer):
        """Faithfulness verdict for one answer against the context it was given."""
        return judge_faithfulness(judge, answer.text, answer.context)

    def pct(found: int, total: int) -> str:
        """'3/4' style count, or '-' when there is nothing to find."""
        return f"{found}/{total}" if total else "-"

    def progress(r) -> None:
        """One line per question as soon as it is scored."""
        verdict = "ok " if r.status_ok else "BAD"
        status = r.status if r.status == "answered" else f"refused/{r.refused_by}"
        faith = "-" if r.faithfulness is None else f"{r.faithfulness:.2f}"
        flags = "".join(f for f, on in (("M", r.no_markers), ("N", r.dropped_not_found), ("T", r.meta_talk)) if on)
        print(f"{r.qid} {r.qtype:<12} {verdict} {status:<14} must {pct(r.must_found, r.must_total):>5} "
              f"art {pct(r.ref_found, r.ref_total):>4} url {pct(r.urls_found, r.urls_total):>4} "
              f"cite {pct(r.cited_correct, r.cited):>4} faith {faith:>4} {flags:<3} {r.total_ms / 1000:5.1f} s",
              flush=True)
        if args.details:
            for m in r.missing:
                print(f"      missing: {m!r}")
            for c in r.unsupported:
                print(f"      unsupported: {c}")
            if r.judge_error:
                print(f"      judge error: {r.judge_error}")

    print(f"{'qid':<4} {'type':<12} {'':3} {'status':<14} {'must':>10} {'art':>7} {'url':>8} {'cite':>9} "
          f"{'faith':>10}  flags (M no [n], N not-found removed, T talks about sources)")
    try:
        report = run_answer_eval(answer_fn, golden, None if args.no_judge else judge_fn, progress)
    except LLMError as e:
        print(f"\nERROR {e}")
        conn.close()
        return 1

    m = report["summary"]

    def f(v, fmt=".2f") -> str:
        """Format a metric that may be None."""
        return "-" if v is None else format(v, fmt)

    print(f"\nAnswerable ({m['answerable']}): answered {f(m['answered'])}  must_include {f(m['must_include'])} "
          f"(all present {f(m['must_include_all'])})  article numbers {f(m['article_numbers'])}  URLs {f(m['urls'])}")
    print(f"  citations: precision {f(m['citation_precision'])}  document recall {f(m['citation_doc_recall'])}  "
          f"context had a golden source {f(m['context_hit'])}")
    print(f"  faithfulness {f(m['faithfulness'])} (fully faithful {f(m['fully_faithful'])}, "
          f"judge errors {m['judge_errors']})")
    print(f"  style: no [n] markers {f(m['no_markers'])}  not-found sentence removed {f(m['dropped_not_found'])}  "
          f"talks about sources {f(m['meta_talk'])}  invalid citations {m['invalid_citations']}")
    if m["wrong_refusals"]:
        print(f"  WRONG REFUSALS: {', '.join(m['wrong_refusals'])}")
    print(f"Unanswerable ({m['unanswerable']}): refused {f(m['refused'])} "
          f"(gate {m['refused_by_gate']}, LLM {m['refused_by_llm']})")
    if m["wrong_answers"]:
        print(f"  ANSWERED WHEN IT SHOULD REFUSE: {', '.join(m['wrong_answers'])}")
    print("By type: " + "   ".join(f"{t} answered {f(v['answered'])} must {f(v['must_include'])} "
                                   f"faith {f(v['faithfulness'])} (n={v['n']})" for t, v in m["by_type"].items()))
    print(f"Latency per question: total p50 {m['total_ms_p50'] / 1000:.1f} s, p95 {m['total_ms_p95'] / 1000:.1f} s "
          f"(retrieval {m['retrieval_ms_p50'] / 1000:.1f} s, generation {m['generation_ms_p50'] / 1000:.1f} s, "
          f"judge {m['judge_ms_p50'] / 1000:.1f} s)")

    report["config"] = {
        "llm_model": s.llm_model, "provider": args.provider or s.llm_provider, "temperature": s.llm_temperature,
        "rerank_top": s.rerank_top, "not_found_score": s.not_found_score, "judge": not args.no_judge,
        "prompt_sha": hashlib.sha256(system_prompt().encode()).hexdigest()[:12],
    }
    out_dir = s.db_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"answers-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(_json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"full report (answers, raw outputs, unsupported claims): {out}")
    conn.close()
    return 0
