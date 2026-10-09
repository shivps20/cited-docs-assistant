"""`kb cache stats` and `kb cache clear`: the exact-match answer cache (kb.answer.cache)."""

import sys

from kb.core.db import connect


def cache_stats() -> int:
    """Print the number of entries, hits, stale entries and the most-asked questions."""
    from kb.answer.cache import AnswerCache

    sys.stdout.reconfigure(encoding="utf-8")
    conn = connect()
    try:
        s = AnswerCache(conn).stats()
    finally:
        conn.close()
    print(f"entries {s['entries']} (current corpus {s['current']}, stale {s['stale']}), "
          f"answers served from the cache {s['hits']}")
    if s["oldest"]:
        print(f"oldest entry {s['oldest'][:16].replace('T', ' ')} UTC")
    if s["top"]:
        print("most served:")
        for question, hits in s["top"]:
            print(f"  {hits:4d}  {question}")
    return 0


def cache_clear(stale_only: bool) -> int:
    """Remove every entry, or with --stale only those of an older corpus."""
    from kb.answer.cache import AnswerCache

    conn = connect()
    try:
        removed = AnswerCache(conn).clear(stale_only=stale_only)
    finally:
        conn.close()
    print(f"removed {removed} {'stale ' if stale_only else ''}cache entr{'y' if removed == 1 else 'ies'}")
    return 0
