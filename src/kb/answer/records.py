"""Context units as stored in a trace's generate stage, and the hash of the messages sent to a model.

Kept free of heavy imports: the answer pipeline writes these records, `kb.answer.explain` reads them back.
"""

import hashlib
import json
from dataclasses import asdict

from kb.retrieve.assemble import ContextUnit

UNIT_FIELDS = ("doc_id", "title", "section_id", "section_number", "heading_path", "header", "page_start", "page_end",
               "text", "kind", "tokens", "release", "external_ok", "side", "source_path", "doc_type")


def unit_record(unit: ContextUnit) -> dict:
    """A context unit as stored in the generate stage: everything needed to rebuild the prompt and judge it."""
    record = {k: getattr(unit, k) for k in UNIT_FIELDS}
    record["same_text"] = [asdict(s) for s in unit.same_text]
    return record


def unit_from_record(record: dict) -> ContextUnit:
    """The context unit back from its stored record (score and chunk ids are not needed any more)."""
    fields = {k: record[k] for k in UNIT_FIELDS if k in record}
    return ContextUnit(score=0.0, **fields)


def messages_hash(messages: list[dict]) -> str:
    """Short hash of the exact messages sent to a model (to tell whether a rebuilt prompt is identical)."""
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
