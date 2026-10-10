"""Ingestion step 3: turn sections into search chunks and store sections + chunks in SQLite.

* Parents: every section's own text goes to the `sections` table; context assembly returns
  the whole section when it is small enough (<= 800 tokens) and get_section() reads it.
* Children: each section is split into chunks of about TARGET_TOKENS (bge-m3 tokens), only at
  block boundaries and never across sections.
  - Tables and code are atomic up to MAX_ATOMIC_TOKENS; larger ones are split by rows/lines
    (tables repeat their header row in every piece).
  - Consecutive command lines (product prompts, SQL, shell, XML/config lines) that Docling left as separate
    text items are merged into one code block first, so examples stay together.
  - A short lead-in line ending in ':' ("For example:") moves with the block it introduces.
  - A very small last chunk is merged into the previous one.
* Every chunk carries a contextual header (document title, release, section path). It is
  embedded with the text so "default index depth" finds a chunk whose body never names the
  section, and it gives the LLM the citation context.
"""

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

from kb.core.config import get_settings
from kb.core.domain import get_domain
from kb.ingest.manifest import Document
from kb.ingest.structure import Block, Section, Structure

# Bump when chunking rules change; stored per document so stale chunks can be detected.
CHUNKER_VERSION = 1

TARGET_TOKENS = 256        # body tokens per chunk (the header comes on top)
MAX_ATOMIC_TOKENS = 512    # tables / code up to this size are never split
MIN_TAIL_TOKENS = 64       # a smaller last chunk is merged into the previous one
LEAD_IN_MAX_TOKENS = 40    # "For example:" style lines that must stay with what follows

TokenCounter = Callable[[str], int]

# Generic command / configuration lines. Products' own prompts and tools come from the domain
# file (command_patterns, see kb.core.domain).
_GENERIC_COMMAND_PATTERNS = (
    r"[A-Za-z]:\\\S*>",                                       # C:\Apache24\bin> prompt
    r"<[A-Za-z/!?][^>]*>?",                                   # XML / config lines
    (r"(?:impdp|expdp|openssl|wmic|sqlplus|httpd\.exe|tmsh|jstack|pstack|wget|export|set|cd|spool|"
     r"EXEC|USE|CREATE|ALTER|GRANT|DROP|SELECT|INSERT|UPDATE|DELETE|"
     r"LoadModule|KeepAlive|KeepAliveTimeout|MaxKeepAliveRequests|ServerRoot|"
     r"SSLEngine|SSLProxyEngine|SSLCertificateFile|SSLCertificateKeyFile|ErrorLog|TransferLog|LogLevel)\b"),
)


@cache
def _command_line(extra: tuple[str, ...]) -> re.Pattern:
    """One regex matching a command line: the domain's patterns plus the generic ones."""
    return re.compile("^(?:" + "|".join((*extra, *_GENERIC_COMMAND_PATTERNS)) + ")")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@cache
def bge_m3_token_counter() -> TokenCounter:
    """Token counter using bge-m3's own tokenizer (no special tokens)."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(get_settings().embed_model_path / "tokenizer.json"))
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


@dataclass
class Chunk:
    """One retrievable piece of a section, as stored in SQLite and embedded into Qdrant."""
    chunk_id: str
    doc_id: str
    section_id: str
    section_number: str
    chunk_index: int
    header: str
    text: str
    content_type: str
    page_start: int
    page_end: int
    token_count: int

    @property
    def embed_text(self) -> str:
        """Text that is embedded: the contextual header, a blank line, then the chunk text."""
        return f"{self.header}\n\n{self.text}"


def section_id(doc_id: str, number: str) -> str:
    """Stable section id, e.g. 'install-guide#3.1.7'."""
    return f"{doc_id}#{number}"


def section_label(section: Section, doc_type: str) -> str:
    """Label for one level of the header path: '3.1 Title', or 'Slide 4: Title' for PPTX."""
    if doc_type == "pptx":
        slides = (f"Slide {section.page_start}" if section.page_start == section.page_end
                  else f"Slides {section.page_start}-{section.page_end}")
        return f"{slides}: {section.title}"
    return section.title if section.number == "0" else f"{section.number} {section.title}"


def section_header(doc: Document, structure: Structure, section: Section) -> str:
    """Contextual header for a section's chunks, e.g. 'Title [R2026x] > 3 Setup > 3.1 Ports'."""
    path = " > ".join(section_label(s, doc.doc_type) for s in structure.path(section))
    return f"{header_prefix(doc)}{path}"


# ---------------------------------------------------------------------------- block preparation

def is_command(text: str) -> bool:
    """Is every non-empty line a command, query or config line (product prompts, SQL, shell, XML ...)?"""
    pattern = _command_line(get_domain().command_patterns)
    return all(pattern.match(line.strip()) for line in text.strip().splitlines() if line.strip())


def merge_commands(blocks: list[Block]) -> list[Block]:
    """Join consecutive command/code lines into single code blocks."""
    merged: list[Block] = []
    for block in blocks:
        code_like = block.kind == "code" or (block.kind == "text" and is_command(block.text))
        if code_like and merged and merged[-1].kind == "code":
            prev = merged[-1]
            merged[-1] = Block("code", f"{prev.text}\n{block.text}", prev.page_start, block.page_end)
        elif code_like:
            merged.append(Block("code", block.text, block.page_start, block.page_end))
        else:
            merged.append(block)
    return merged


def _pack_units(units: list[str], joiner: str, count: TokenCounter, limit: int,
                prefix: str = "") -> list[str]:
    """Greedily pack text units into pieces of at most `limit` tokens (a single oversized unit stays whole)."""
    pieces, current = [], []
    for unit in units:
        candidate = joiner.join(current + [unit])
        if current and count(prefix + candidate) > limit:
            pieces.append(prefix + joiner.join(current))
            current = [unit]
        else:
            current.append(unit)
    if current:
        pieces.append(prefix + joiner.join(current))
    return pieces


def split_block(block: Block, count: TokenCounter) -> list[Block]:
    """Split one block that is too large for a chunk, at the safest boundaries for its kind."""
    tokens = count(block.text)
    limit = MAX_ATOMIC_TOKENS if block.kind in ("table", "code") else TARGET_TOKENS
    if tokens <= limit:
        return [block]

    if block.kind == "table":
        lines = block.text.split("\n")
        head = "\n".join(lines[:2]) + "\n"          # header row + separator, repeated in every piece
        pieces = _pack_units(lines[2:], "\n", count, TARGET_TOKENS, prefix=head)
    elif block.kind in ("code", "list"):
        pieces = _pack_units(block.text.split("\n"), "\n", count, TARGET_TOKENS)
    else:
        sentences = _SENTENCE_END.split(block.text)
        units = []
        for sentence in sentences:                  # a single huge sentence: split by words
            if count(sentence) > TARGET_TOKENS:
                units.extend(_pack_units(sentence.split(), " ", count, TARGET_TOKENS))
            else:
                units.append(sentence)
        pieces = _pack_units(units, " ", count, TARGET_TOKENS)
    return [Block(block.kind, p, block.page_start, block.page_end) for p in pieces]


def _is_lead_in(block: Block, count: TokenCounter) -> bool:
    """A short text block ending in ':' that introduces what follows and must stay with it."""
    return block.kind == "text" and block.text.rstrip().endswith(":") and count(block.text) <= LEAD_IN_MAX_TOKENS


def pack_blocks(blocks: list[Block], count: TokenCounter) -> list[list[Block]]:
    """Group blocks into chunks of about TARGET_TOKENS without splitting any block."""
    groups: list[list[Block]] = []
    current: list[Block] = []
    current_tokens = 0
    for block in blocks:
        tokens = count(block.text)
        if current and current_tokens + tokens > TARGET_TOKENS:
            carry = [current.pop()] if len(current) > 1 and _is_lead_in(current[-1], count) else []
            groups.append(current)
            current, current_tokens = carry, sum(count(b.text) for b in carry)
        current.append(block)
        current_tokens += tokens
    if current:
        groups.append(current)

    if len(groups) >= 2:
        tail, prev = groups[-1], groups[-2]
        tail_tokens = sum(count(b.text) for b in tail)
        if tail_tokens < MIN_TAIL_TOKENS and sum(count(b.text) for b in prev) + tail_tokens <= TARGET_TOKENS * 1.5:
            groups[-2:] = [prev + tail]
    return groups


# ---------------------------------------------------------------------------- chunks

def chunk_section(doc: Document, structure: Structure, section: Section, count: TokenCounter) -> list[Chunk]:
    """Split one section into chunks of about TARGET_TOKENS, keeping tables, code and lists whole."""
    blocks = [piece for block in merge_commands(section.blocks) for piece in split_block(block, count)]
    if not blocks:
        return []
    header = section_header(doc, structure, section)
    sid = section_id(doc.doc_id, section.number)
    chunks = []
    for index, group in enumerate(pack_blocks(blocks, count)):
        text = "\n\n".join(b.text for b in group)
        kinds = {b.kind for b in group}
        chunks.append(Chunk(
            chunk_id=f"{sid}#{index}", doc_id=doc.doc_id, section_id=sid, section_number=section.number,
            chunk_index=index, header=header, text=text, content_type=kinds.pop() if len(kinds) == 1 else "mixed",
            page_start=min(b.page_start for b in group), page_end=max(b.page_end for b in group),
            token_count=count(f"{header}\n\n{text}"),
        ))
    return chunks


def chunk_document(doc: Document, structure: Structure, count: TokenCounter) -> list[Chunk]:
    """Chunks for every section of the document, in document order."""
    return [c for s in structure.sections for c in chunk_section(doc, structure, s, count)]


# ---------------------------------------------------------------------------- storage

def header_prefix(doc: Document) -> str:
    """The start of every chunk header of the document: 'Title [R2026x] > ' (no release part for 'any')."""
    release = "" if doc.release_label == "any" else f" [{doc.release_label}]"
    return f"{doc.title}{release} > "


def chunks_up_to_date(conn: sqlite3.Connection, doc: Document) -> bool:
    """Can `kb chunk` skip this document? Yes when it is chunked or indexed from the current parse (a changed
    file resets the status to 'parsed'), with this CHUNKER_VERSION, and its stored chunk headers still start
    with the manifest's current title and release. Code or domain.yaml changes need `kb chunk --force`."""
    row = conn.execute("SELECT status, chunker_version, chunk_count FROM documents WHERE doc_id = ?",
                       (doc.doc_id,)).fetchone()
    if row is None or row["status"] not in ("chunked", "indexed") or row["chunker_version"] != CHUNKER_VERSION:
        return False
    if row["chunk_count"] == 0:                 # nothing to compare (e.g. an image-only PDF): chunked is final
        return True
    header = conn.execute("SELECT header FROM chunks WHERE doc_id = ? LIMIT 1", (doc.doc_id,)).fetchone()
    return header is not None and header["header"].startswith(header_prefix(doc))


def store_document(conn: sqlite3.Connection, doc: Document, structure: Structure, chunks: list[Chunk],
                   count: TokenCounter) -> None:
    """Replace the document's sections and chunks and mark it chunked, in one transaction."""
    if conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc.doc_id,)).fetchone() is None:
        raise ValueError(f"{doc.doc_id} has no documents row; run `kb parse` first")
    section_rows = []
    for ordinal, s in enumerate(structure.sections):
        text = "\n\n".join(b.text for b in s.blocks)
        section_rows.append((
            section_id(doc.doc_id, s.number), doc.doc_id,
            section_id(doc.doc_id, s.parent) if s.parent else None, s.title,
            " > ".join(section_label(p, doc.doc_type) for p in structure.path(s)),
            s.level, ordinal, s.page_start, s.page_end, text, count(text),
        ))
    with conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.doc_id,))
        conn.execute("DELETE FROM sections WHERE doc_id = ?", (doc.doc_id,))
        conn.executemany(
            "INSERT INTO sections (section_id, doc_id, parent_section_id, heading, heading_path, level, ordinal, "
            "page_start, page_end, text, token_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", section_rows)
        conn.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, section_id, chunk_index, header, text, content_type, "
            "page_start, page_end, token_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(c.chunk_id, c.doc_id, c.section_id, c.chunk_index, c.header, c.text, c.content_type,
              c.page_start, c.page_end, c.token_count) for c in chunks])
        conn.execute(
            "UPDATE documents SET status = 'chunked', chunk_count = ?, chunker_version = ?, error = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE doc_id = ?",
            (len(chunks), CHUNKER_VERSION, doc.doc_id))
