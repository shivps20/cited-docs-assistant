"""SQLite access: schema migrations and the connection helper used by all app code.

Always open the database through connect(): SQLite resets PRAGMA foreign_keys on
every connection, so a raw sqlite3.connect() silently skips the cascade/foreign-key
rules the schema relies on.
"""

import sqlite3
from pathlib import Path

from kb.core.config import get_settings

NOW = "(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"  # ISO-8601 UTC

# Each entry migrates the schema from version i to i+1. Never edit an applied
# migration; append a new one instead.
MIGRATIONS: list[str] = [
    # v1: initial schema
    f"""
    -- Ingest registry: one row per source file.
    CREATE TABLE documents (
        doc_id          TEXT PRIMARY KEY,
        source_path     TEXT NOT NULL UNIQUE,
        title           TEXT,
        doc_type        TEXT,                               -- pdf / pptx / docx
        file_hash       TEXT NOT NULL,                      -- sha256; unchanged file => skip
        page_count      INTEGER,
        release_version TEXT,                               -- e.g. R2026x
        revision        TEXT,
        is_latest       INTEGER NOT NULL DEFAULT 1 CHECK (is_latest IN (0, 1)),
        category        TEXT,
        allowed_groups  TEXT NOT NULL DEFAULT '[]',         -- JSON array
        external_ok     INTEGER NOT NULL DEFAULT 0 CHECK (external_ok IN (0, 1)),
        status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'parsed', 'chunked', 'indexed', 'failed')),
        error           TEXT,
        chunk_count     INTEGER,
        index_version   TEXT,
        parsed_at       TEXT,
        indexed_at      TEXT,
        created_at      TEXT NOT NULL DEFAULT {NOW},
        updated_at      TEXT NOT NULL DEFAULT {NOW}
    ) STRICT;
    CREATE INDEX idx_documents_status ON documents (status);

    -- Parent sections: full text returned by context assembly and get_section().
    CREATE TABLE sections (
        section_id        TEXT PRIMARY KEY,
        doc_id            TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
        parent_section_id TEXT,
        heading           TEXT,
        heading_path      TEXT,                             -- "Install > Database > Oracle"
        level             INTEGER,
        ordinal           INTEGER NOT NULL,                 -- order within the document
        page_start        INTEGER,
        page_end          INTEGER,
        text              TEXT NOT NULL,
        token_count       INTEGER NOT NULL
    ) STRICT;
    CREATE INDEX idx_sections_doc ON sections (doc_id, ordinal);

    -- Chat sessions and turns.
    CREATE TABLE sessions (
        session_id     TEXT PRIMARY KEY,
        user_id        TEXT NOT NULL,
        user_groups    TEXT NOT NULL DEFAULT '[]',          -- JSON array, snapshot at session start
        sticky_release TEXT,                                -- release detected earlier in the session
        created_at     TEXT NOT NULL DEFAULT {NOW},
        last_active_at TEXT NOT NULL DEFAULT {NOW}
    ) STRICT;

    CREATE TABLE messages (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id       TEXT NOT NULL REFERENCES sessions (session_id) ON DELETE CASCADE,
        role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content          TEXT NOT NULL,
        standalone_query TEXT,                              -- condensed follow-up, user turns only
        trace_id         TEXT,
        created_at       TEXT NOT NULL DEFAULT {NOW}
    ) STRICT;
    CREATE INDEX idx_messages_session ON messages (session_id, id);

    -- One row per query.
    CREATE TABLE traces (
        trace_id         TEXT PRIMARY KEY,
        session_id       TEXT REFERENCES sessions (session_id) ON DELETE SET NULL,
        user_id          TEXT,
        query            TEXT NOT NULL,
        standalone_query TEXT,
        route            TEXT,                              -- single / agentic / cache
        release_filter   TEXT,
        top_rerank_score REAL,
        gate_decision    TEXT,                              -- pass / broaden / escalate / refuse
        llm_provider     TEXT,
        llm_model        TEXT,
        answer           TEXT,
        sources          TEXT,                              -- JSON array of cited chunks
        cache_hit        INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0, 1)),
        index_version    TEXT,
        total_ms         REAL,
        error            TEXT,
        created_at       TEXT NOT NULL DEFAULT {NOW}
    ) STRICT;
    CREATE INDEX idx_traces_created ON traces (created_at);
    CREATE INDEX idx_traces_session ON traces (session_id);

    -- One row per pipeline stage per query (condense, route, embed, search, rerank, ...).
    CREATE TABLE trace_stages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id    TEXT NOT NULL REFERENCES traces (trace_id) ON DELETE CASCADE,
        seq         INTEGER NOT NULL,
        stage       TEXT NOT NULL,
        duration_ms REAL,
        data        TEXT,                                   -- JSON: candidates, scores, params
        created_at  TEXT NOT NULL DEFAULT {NOW}
    ) STRICT;
    CREATE INDEX idx_trace_stages_trace ON trace_stages (trace_id, seq);

    CREATE TABLE feedback (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id   TEXT NOT NULL REFERENCES traces (trace_id) ON DELETE CASCADE,
        user_id    TEXT,
        rating     INTEGER NOT NULL CHECK (rating IN (-1, 1)),
        comment    TEXT,
        created_at TEXT NOT NULL DEFAULT {NOW},
        UNIQUE (trace_id, user_id)
    ) STRICT;

    -- Exact-match answer cache. Key = sha256(query + groups + release + index_version + model_id).
    CREATE TABLE answer_cache (
        cache_key      TEXT PRIMARY KEY,
        query_norm     TEXT NOT NULL,
        user_groups    TEXT NOT NULL,
        release_filter TEXT,
        index_version  TEXT NOT NULL,
        model_id       TEXT NOT NULL,
        answer         TEXT NOT NULL,
        sources        TEXT,
        hit_count      INTEGER NOT NULL DEFAULT 0,
        created_at     TEXT NOT NULL DEFAULT {NOW},
        last_hit_at    TEXT
    ) STRICT;
    CREATE INDEX idx_answer_cache_index_version ON answer_cache (index_version);
    """,
    # v2: manifest release ranges and revision families (release_version keeps the display label)
    """
    ALTER TABLE documents ADD COLUMN family TEXT;
    ALTER TABLE documents ADD COLUMN release_min INTEGER;
    ALTER TABLE documents ADD COLUMN release_max INTEGER;
    CREATE INDEX idx_documents_family ON documents (family);
    """,
    # v3: parse statistics per document (shown by `kb status`)
    """
    ALTER TABLE documents ADD COLUMN parse_seconds REAL;      -- last real parse (not cache hits)
    ALTER TABLE documents ADD COLUMN text_items INTEGER;
    ALTER TABLE documents ADD COLUMN headings INTEGER;
    ALTER TABLE documents ADD COLUMN tables INTEGER;
    ALTER TABLE documents ADD COLUMN pictures INTEGER;
    ALTER TABLE documents ADD COLUMN empty_pages INTEGER;     -- pages with no body text or table
    ALTER TABLE documents ADD COLUMN furniture INTEGER;       -- page headers/footers separated by Docling
    """,
    # v4: chunks (search units) produced from sections; embedded and indexed in later steps
    """
    CREATE TABLE chunks (
        chunk_id     TEXT PRIMARY KEY,                      -- "<doc_id>#<section number>#<chunk_index>"
        doc_id       TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
        section_id   TEXT NOT NULL REFERENCES sections (section_id) ON DELETE CASCADE,
        chunk_index  INTEGER NOT NULL,                      -- position within its section (for +/-1 neighbours)
        header       TEXT NOT NULL,                         -- "doc title [release] > 3.1 ... > 3.1.7 ..."
        text         TEXT NOT NULL,
        content_type TEXT NOT NULL CHECK (content_type IN ('text', 'list', 'table', 'code', 'mixed')),
        page_start   INTEGER,
        page_end     INTEGER,
        token_count  INTEGER NOT NULL                       -- of header + text, as embedded
    ) STRICT;
    CREATE INDEX idx_chunks_section ON chunks (section_id, chunk_index);
    CREATE INDEX idx_chunks_doc ON chunks (doc_id);
    ALTER TABLE documents ADD COLUMN chunker_version INTEGER;
    """,
    # v5: indexing bookkeeping
    """
    ALTER TABLE documents ADD COLUMN metadata_hash TEXT;      -- manifest metadata last written to Qdrant
    ALTER TABLE documents ADD COLUMN embed_seconds REAL;      -- last full embedding run
    """,
    # v6: why an answer got a thumbs down (Phase 7: input for re-calibrating the "not found" gate)
    """
    ALTER TABLE feedback ADD COLUMN reason TEXT
        CHECK (reason IN ('wrong', 'incomplete', 'should_have_answered', 'should_have_refused'));
    """,
]
SCHEMA_VERSION = len(MIGRATIONS)


class SchemaError(RuntimeError):
    """The database is missing or its schema is older than this code expects."""


def connect(path: Path | None = None, *, check_schema: bool = True) -> sqlite3.Connection:
    """Open the database with foreign keys on, Row results, and a busy timeout for concurrent writers."""
    path = path or get_settings().db_path
    if check_schema and not path.exists():
        raise SchemaError(f"{path} does not exist; run `uv run python scripts/init_db.py`")
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    if check_schema:
        version = schema_version(conn)
        if version != SCHEMA_VERSION:
            conn.close()
            raise SchemaError(f"{path} is at schema v{version}, code expects v{SCHEMA_VERSION}; "
                              "run `uv run python scripts/init_db.py`")
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    """Schema version of the database (SQLite `user_version`; 0 for a new file)."""
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply pending migrations; returns the versions applied (empty if already current)."""
    current = schema_version(conn)
    if current > SCHEMA_VERSION:
        raise SchemaError(f"database is at schema v{current}, newer than this code (v{SCHEMA_VERSION})")
    conn.execute("PRAGMA journal_mode = WAL")  # persistent in the file: readers don't block the writer
    applied = []
    for version in range(current, SCHEMA_VERSION):
        # executescript commits first, so wrap each migration in its own transaction.
        conn.executescript(f"BEGIN;\n{MIGRATIONS[version]}\nPRAGMA user_version = {version + 1};\nCOMMIT;")
        applied.append(version + 1)
    return applied
