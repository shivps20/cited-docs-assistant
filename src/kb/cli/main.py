"""Command-line entry point: `uv run kb <command>` (argument parsing and dispatch)."""

import argparse
import sys

from kb.cli.evaluation import eval_answers_command, eval_command
from kb.cli.ingest import (
    chunk_documents,
    index_documents,
    inspect_chunks,
    inspect_document,
    manifest_scan,
    manifest_validate,
    parse_documents,
    show_coverage,
    show_status,
)
from kb.cli.search import ask_command, search_command
from kb.cli.serve import serve_command


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `kb` command: parse arguments and run the chosen subcommand."""
    parser = argparse.ArgumentParser(prog="kb", description="Knowledge-base assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="document manifest").add_subparsers(dest="action", required=True)
    manifest.add_parser("validate", help="validate the manifest and list documents")
    manifest.add_parser("scan", help="append draft rows for files not yet in the manifest")
    parse = commands.add_parser("parse", help="parse documents with Docling and cache the result")
    parse.add_argument("--doc", action="append", metavar="DOC_ID", help="only this document (repeatable)")
    parse.add_argument("--force", action="store_true", help="re-parse even if a cached result exists")
    commands.add_parser("status", help="ingestion status and parse statistics per document")
    inspect = commands.add_parser("inspect", help="show a document's rebuilt section structure")
    inspect.add_argument("doc_id")
    inspect.add_argument("--section", metavar="NUMBER", help="print the content of one section, e.g. 3.1.7")
    inspect.add_argument("--details", action="store_true",
                         help="also list unmatched TOC entries, demoted headings and removed lines")
    inspect.add_argument("--chunks", action="store_true",
                         help="show the chunks (all, or of --section) as they will be embedded")
    chunk = commands.add_parser("chunk", help="build sections + chunks from parsed documents and store them")
    chunk.add_argument("--doc", action="append", metavar="DOC_ID", help="only this document (repeatable)")
    coverage = commands.add_parser("coverage", help="check golden facts are present in stored chunks")
    coverage.add_argument("--all", action="store_true", help="list passing questions too")
    index = commands.add_parser("index", help="embed chunks and write them to Qdrant")
    index.add_argument("--doc", action="append", metavar="DOC_ID", help="only this document (repeatable)")
    index.add_argument("--force", action="store_true", help="re-embed even if already indexed")
    index.add_argument("--prune", action="store_true",
                       help="delete points and rows of documents no longer in the manifest")
    search_cmd = commands.add_parser("search", help="retrieve chunks and assembled context for a question")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--groups", action="append", metavar="GROUP",
                            help="user access group(s), repeatable or comma-separated (default: all)")
    search_cmd.add_argument("--release", metavar="RELEASE", help="restrict to documents for a release, e.g. R2024x")
    search_cmd.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    search_cmd.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder reranker")
    search_cmd.add_argument("--candidates", type=int, default=30, help="chunks retrieved before reranking")
    search_cmd.add_argument("--show", type=int, default=10, help="ranked results to print")
    search_cmd.add_argument("--context", action="store_true", help="print the full assembled context")
    search_cmd.add_argument("--rerank-top", type=int,
                            help="rerank only the first N candidates; 0 = all (default: KB_RERANK_TOP, 20)")
    search_cmd.add_argument("--max-length", type=int, default=512, help="reranker input length in tokens")
    search_cmd.add_argument("--min-score", type=float, help="drop context chunks with rerank score below this")
    serve = commands.add_parser("serve", help="run the chat API and web UI")
    serve.add_argument("--host", help="interface to listen on (default: KB_API_HOST, 127.0.0.1)")
    serve.add_argument("--port", type=int, help="port (default: KB_API_PORT, 8000)")
    ask = commands.add_parser("ask", help="answer a question with citations (retrieval + LLM)")
    ask.add_argument("query")
    ask.add_argument("--groups", action="append", metavar="GROUP",
                     help="user access group(s), repeatable or comma-separated (default: all)")
    ask.add_argument("--release", metavar="RELEASE", help="restrict to documents for a release, e.g. R2024x")
    ask.add_argument("--provider", choices=["auto", "ollama", "openai"],
                     help="LLM provider (default: KB_LLM_PROVIDER, auto)")
    ask.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    ask.add_argument("--no-rerank", action="store_true", help="skip the reranker (and the score gate)")
    ask.add_argument("--rerank-top", type=int,
                     help="rerank only the first N candidates; 0 = all (default: KB_RERANK_TOP, 20)")
    ask.add_argument("--show-context", action="store_true", help="print the context sent to the LLM first")
    ask.add_argument("--no-stream", action="store_true", help="print the answer only when it is complete")
    eval_answers = commands.add_parser("eval-answers", help="evaluate generated answers against the golden set")
    eval_answers.add_argument("--questions", help="comma-separated golden ids, e.g. q001,q012 (default: all)")
    eval_answers.add_argument("--types", help="comma-separated question types: lookup, howto, compare, unanswerable")
    eval_answers.add_argument("--no-judge", action="store_true", help="skip the LLM faithfulness judge (faster)")
    eval_answers.add_argument("--provider", choices=["auto", "ollama", "openai"],
                              help="LLM provider for the answers (default: KB_LLM_PROVIDER, auto)")
    eval_answers.add_argument("--details", action="store_true",
                              help="also print missing strings and unsupported claims per question")
    eval_cmd = commands.add_parser("eval", help="evaluate retrieval configurations against the golden set")
    eval_cmd.add_argument("--configs", help="comma-separated subset of: dense, sparse, hybrid, hybrid+rr10, "
                                            "hybrid+rr15@256, hybrid+rr30 (default: all)")
    eval_cmd.add_argument("--questions", help="comma-separated golden ids, e.g. q001,q012 (default: all)")
    eval_cmd.add_argument("--misses", action="store_true", help="list questions without a hit in the top 5")

    args = parser.parse_args(argv)
    from kb.core.perf import disable_power_throttling

    disable_power_throttling()       # keep CPU model work at full speed when the window is in the background
    if args.command == "manifest":
        return {"validate": manifest_validate, "scan": manifest_scan}[args.action]()
    if args.command == "parse":
        return parse_documents(args.doc, args.force)
    if args.command == "status":
        return show_status()
    if args.command == "inspect":
        if args.chunks:
            return inspect_chunks(args.doc_id, args.section)
        return inspect_document(args.doc_id, args.section, args.details)
    if args.command == "chunk":
        return chunk_documents(args.doc)
    if args.command == "coverage":
        return show_coverage(args.all)
    if args.command == "index":
        return index_documents(args.doc, args.force, args.prune)
    if args.command == "search":
        return search_command(args)
    if args.command == "serve":
        return serve_command(args)
    if args.command == "ask":
        return ask_command(args)
    if args.command == "eval":
        return eval_command(args)
    if args.command == "eval-answers":
        return eval_answers_command(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
