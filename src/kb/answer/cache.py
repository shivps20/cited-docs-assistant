"""Exact-match answer cache (Phase 7): the same question, asked again under the same conditions, is
answered from SQLite without search, reranking or the LLM.

The key covers everything that can change the answer:
- the question, normalised (case, spacing and trailing punctuation ignored); for a chat follow-up
  this is the condensed standalone question;
- the user's access groups, the release filter and the retrieval options;
- the corpus fingerprint: a hash of every indexed document's file hash, manifest metadata (groups,
  external_ok, release range) and index version, so re-indexing or a manifest change retires every
  entry at once;
- the answer model's catalogue profile, the system prompt and the answer settings (comparison path,
  refusal retry, gate threshold).

Only clean answers are stored: answered (never "not found", so a refusal is never repeated from the
cache), written by the requested model itself (no fallback after an error, no model swapped by the
privacy policy). A 👎 on an answer, or on a cached copy of it, removes the entry.

Uses the `answer_cache` table of schema v1: `index_version` holds the corpus fingerprint, `model_id`
the model and settings, `answer` the whole answer as JSON, `sources` its cited sources.
"""

import dataclasses
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass

from kb.llm.providers import Generation
from kb.retrieve.assemble import ContextUnit, SameText
from kb.retrieve.gate import GateDecision

_NOW = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


def normalise_question(question: str) -> str:
    """The question as the cache compares it: lower case, single spaces, no trailing ?!. or spaces."""
    return re.sub(r"\s+", " ", question).strip().rstrip("?!. ").lower()


def _sha(value: object) -> str:
    """Short sha256 of a JSON-serialisable value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def corpus_fingerprint(conn: sqlite3.Connection) -> str:
    """Hash of the indexed corpus: changes when a document is added, removed, re-indexed or its
    manifest entry (groups, external_ok, releases) changes."""
    rows = conn.execute("""
        SELECT doc_id, file_hash, metadata_hash, index_version, allowed_groups, external_ok, is_latest
        FROM documents WHERE status = 'indexed' ORDER BY doc_id
    """).fetchall()
    return _sha([tuple(r) for r in rows])


@dataclass
class CacheKey:
    """The parts of a cache key, kept apart so the table stays readable (`kb cache stats`)."""
    question: str          # normalised question
    groups: str            # JSON array, sorted
    release: str | None
    retrieval: str         # hash of the retrieval options
    corpus: str            # corpus fingerprint
    model: str             # profile name + hash of profile, prompt and answer settings

    @property
    def digest(self) -> str:
        """The primary key: sha256 of all parts."""
        return hashlib.sha256(json.dumps(dataclasses.astuple(self)).encode()).hexdigest()


def cache_key(conn: sqlite3.Connection, question: str, *, groups: list[str], release: str | None,
              retrieval: dict, model: str, settings: dict) -> CacheKey:
    """The key for `question` under these conditions. settings: the profile, prompt hash and answer
    switches that decide how the answer model answers."""
    return CacheKey(question=normalise_question(question), groups=json.dumps(sorted(set(groups))),
                    release=release, retrieval=_sha(retrieval), corpus=corpus_fingerprint(conn),
                    model=f"{model}:{_sha(settings)}")


def _unit(data: dict) -> ContextUnit:
    """A context unit from its stored JSON."""
    data = dict(data)
    data["same_text"] = [SameText(**s) for s in data.get("same_text", [])]
    if data.get("window") is not None:
        data["window"] = tuple(data["window"])
    return ContextUnit(**data)


@dataclass
class CachedAnswer:
    """A stored answer: everything needed to rebuild the Answer, plus when it was first given."""
    payload: dict
    created_at: str
    hit_count: int

    @property
    def trace_id(self) -> str:
        """The trace of the answer that was stored (feedback on it, or on a copy, removes the entry)."""
        return self.payload["trace_id"]

    def context(self) -> list[ContextUnit]:
        """The context the answer was written from."""
        return [_unit(u) for u in self.payload["context"]]

    def gate(self) -> GateDecision:
        """The gate decision of the original answer."""
        return GateDecision(**self.payload["gate"])

    def generation(self) -> Generation:
        """The original generation's provider and model (no timings: nothing was generated now)."""
        g = self.payload["generation"]
        return Generation(self.payload["text"], g["provider"], g["model"], 0.0)


class AnswerCache:
    """Reads and writes the answer_cache table through one SQLite connection."""

    def __init__(self, conn: sqlite3.Connection):
        """Use `conn` for every read and write."""
        self.conn = conn

    def get(self, key: CacheKey) -> CachedAnswer | None:
        """The stored answer for `key` (and count the hit), or None."""
        row = self.conn.execute("SELECT answer, created_at, hit_count FROM answer_cache WHERE cache_key = ?",
                                (key.digest,)).fetchone()
        if row is None:
            return None
        with self.conn:
            self.conn.execute(f"UPDATE answer_cache SET hit_count = hit_count + 1, last_hit_at = {_NOW} "
                              "WHERE cache_key = ?", (key.digest,))
        return CachedAnswer(json.loads(row["answer"]), row["created_at"], row["hit_count"] + 1)

    def put(self, key: CacheKey, payload: dict) -> None:
        """Store an answer under `key` (replacing an older one) and drop entries of an older corpus."""
        with self.conn:
            self.conn.execute("DELETE FROM answer_cache WHERE index_version != ?", (key.corpus,))
            self.conn.execute("""
                INSERT OR REPLACE INTO answer_cache
                    (cache_key, query_norm, user_groups, release_filter, index_version, model_id, answer, sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (key.digest, key.question, key.groups, key.release, key.corpus, key.model,
                  json.dumps(payload, ensure_ascii=False, default=str),
                  json.dumps(payload["sources"], ensure_ascii=False, default=str)))

    def forget_trace(self, trace_id: str) -> int:
        """Remove the entry stored from `trace_id`, or the one a cache hit with that trace was served
        from (a 👎 on either); returns the number of entries removed."""
        hit = self.conn.execute("SELECT data FROM trace_stages WHERE trace_id = ? AND stage = 'cache'",
                                (trace_id,)).fetchone()
        served_from = json.loads(hit["data"]).get("served_from") if hit else None
        with self.conn:
            cur = self.conn.execute("DELETE FROM answer_cache WHERE json_extract(answer, '$.trace_id') IN (?, ?)",
                                    (trace_id, served_from or trace_id))
        return cur.rowcount

    def stats(self) -> dict:
        """Entries, hits, entries of the current corpus, and the most-hit questions."""
        current = corpus_fingerprint(self.conn)
        row = self.conn.execute("""
            SELECT COUNT(*) AS entries, COALESCE(SUM(hit_count), 0) AS hits,
                   COALESCE(SUM(index_version = ?), 0) AS current, MIN(created_at) AS oldest
            FROM answer_cache
        """, (current,)).fetchone()
        top = self.conn.execute("SELECT query_norm, hit_count FROM answer_cache WHERE hit_count > 0 "
                                "ORDER BY hit_count DESC LIMIT 10").fetchall()
        return {**dict(row), "stale": row["entries"] - row["current"],
                "top": [(r["query_norm"], r["hit_count"]) for r in top]}

    def clear(self, *, stale_only: bool = False) -> int:
        """Remove every entry, or only those of an older corpus; returns the number removed."""
        with self.conn:
            if stale_only:
                cur = self.conn.execute("DELETE FROM answer_cache WHERE index_version != ?",
                                        (corpus_fingerprint(self.conn),))
            else:
                cur = self.conn.execute("DELETE FROM answer_cache")
        return cur.rowcount
