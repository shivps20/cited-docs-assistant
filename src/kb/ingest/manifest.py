"""Document manifest: the single source of truth for per-document metadata.

One CSV row per source file. Metadata that cannot be inferred reliably from the
files themselves (access groups, release applicability, revision lineage) lives
here, is validated on load, and is copied into every chunk at ingestion.

Columns:
    doc_id          unique slug, e.g. "install-guide"
    path            file path relative to the documents directory
    title           human-readable title, used in citations
    family          logical document across revisions (defaults to doc_id)
    version         document revision, e.g. "3.2"; the highest per family is latest
    release_min     first release the document applies to, e.g. "R2015x" (blank = any)
    release_max     last release it applies to (blank = open-ended)
    allowed_groups  ";"-separated access groups; "all" = everyone
    external_ok     true/false: may chunks be sent to an external LLM (OpenAI)
    category        one of CATEGORIES
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ["doc_id", "path", "title", "family", "version", "release_min", "release_max",
           "allowed_groups", "external_ok", "category"]
CATEGORIES = {"installation", "administration", "authentication", "infrastructure",
              "upgrade", "performance", "applications", "functional", "troubleshooting"}
DOC_TYPES = {".pdf": "pdf", ".pptx": "pptx", ".docx": "docx"}

# Releases are stored as their year (R2026x -> 2026) so ranges can be filtered numerically.
RELEASE_ANY_MIN = 0
RELEASE_ANY_MAX = 9999

_RELEASE = re.compile(r"^(?:V6)?R(\d{4})x$", re.IGNORECASE)
_RELEASE_IN_NAME = re.compile(r"R(20\d\d)x", re.IGNORECASE)
_DOC_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_GROUP = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VERSION = re.compile(r"^\d+(\.\d+)*$")
_BOOL = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}


class ManifestError(Exception):
    """Raised with every problem found in the manifest, so all can be fixed in one pass."""

    def __init__(self, errors: list[str]):
        """Keep the list of problems and build a message listing them all."""
        super().__init__(f"{len(errors)} manifest error(s):\n" + "\n".join(errors))
        self.errors = errors


def parse_release(text: str) -> int:
    """'R2026x' or 'V6R2013x' -> 2026 / 2013."""
    match = _RELEASE.match(text.strip())
    if not match:
        raise ValueError(f"invalid release {text!r} (expected e.g. R2026x)")
    return int(match.group(1))


def release_label(release_min: int, release_max: int) -> str:
    """Human-readable release range, e.g. 'R2015x+', 'R2021x-R2026x', 'up to R2024x' or 'any'."""
    if release_min == RELEASE_ANY_MIN and release_max == RELEASE_ANY_MAX:
        return "any"
    if release_max == RELEASE_ANY_MAX:
        return f"R{release_min}x+"
    if release_min == RELEASE_ANY_MIN:
        return f"up to R{release_max}x"
    if release_min == release_max:
        return f"R{release_min}x"
    return f"R{release_min}x-R{release_max}x"


@dataclass(frozen=True)
class Document:
    """One manifest row, validated, plus `is_latest` computed across its family."""
    doc_id: str
    path: Path
    title: str
    family: str
    version: str
    release_min: int
    release_max: int
    allowed_groups: tuple[str, ...]
    external_ok: bool
    category: str
    doc_type: str
    is_latest: bool

    @property
    def release_label(self) -> str:
        """This document's release range as text (see release_label())."""
        return release_label(self.release_min, self.release_max)

    def applies_to(self, release: int) -> bool:
        """Does the document apply to the given release year?"""
        return self.release_min <= release <= self.release_max


def _version_key(version: str) -> tuple[int, ...]:
    """Version string as a tuple of integers, so '1.10' sorts after '1.9'."""
    return tuple(int(part) for part in version.split("."))


def _parse_row(row: dict[str, str], line: int, docs_dir: Path, errors: list[str]) -> dict | None:
    """Validate one CSV row; appends problems to `errors` and returns the parsed fields, or None if invalid.
    """
    def err(msg: str) -> None:
        """Record a problem for this row's line number."""
        errors.append(f"line {line}: {msg}")

    n_errors = len(errors)
    doc_id = row["doc_id"].strip()
    if not _DOC_ID.match(doc_id):
        err(f"doc_id {doc_id!r} must be a lowercase slug (a-z, 0-9, '-')")

    rel_path = row["path"].strip()
    path = docs_dir / rel_path
    doc_type = DOC_TYPES.get(path.suffix.lower())
    if not rel_path:
        err("path is empty")
    elif doc_type is None:
        err(f"unsupported file type {path.suffix!r} (supported: {', '.join(sorted(DOC_TYPES))})")
    elif not path.is_file():
        err(f"file not found: {path}")

    title = row["title"].strip()
    if not title:
        err("title is empty")

    version = row["version"].strip()
    if not _VERSION.match(version):
        err(f"version {version!r} must look like 1.0 or 3.2")

    try:
        release_min = parse_release(row["release_min"]) if row["release_min"].strip() else RELEASE_ANY_MIN
        release_max = parse_release(row["release_max"]) if row["release_max"].strip() else RELEASE_ANY_MAX
        if release_min > release_max:
            err(f"release_min {row['release_min']} is after release_max {row['release_max']}")
    except ValueError as e:
        err(str(e))
        release_min = release_max = 0

    groups = tuple(g.strip() for g in row["allowed_groups"].split(";") if g.strip())
    if not groups:
        err("allowed_groups is empty (use 'all' for everyone)")
    for group in groups:
        if not _GROUP.match(group):
            err(f"group {group!r} must be lowercase (a-z, 0-9, '_', '-')")

    external_ok = _BOOL.get(row["external_ok"].strip().lower())
    if external_ok is None:
        err(f"external_ok {row['external_ok']!r} must be true or false")

    category = row["category"].strip()
    if category not in CATEGORIES:
        err(f"category {category!r} must be one of: {', '.join(sorted(CATEGORIES))}")

    if len(errors) > n_errors:
        return None
    return {
        "doc_id": doc_id, "path": path.resolve(), "title": title, "family": row["family"].strip() or doc_id,
        "version": version, "release_min": release_min, "release_max": release_max, "allowed_groups": groups,
        "external_ok": external_ok, "category": category, "doc_type": doc_type,
    }


def load_manifest(manifest_path: Path, docs_dir: Path) -> list[Document]:
    """Load and validate the manifest. Raises ManifestError listing every problem found."""
    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if header != COLUMNS:
            raise ManifestError([f"header must be exactly: {','.join(COLUMNS)} (got: {','.join(header)})"])
        rows = list(reader)

    errors: list[str] = []
    parsed = []
    seen_ids: dict[str, int] = {}
    seen_paths: dict[Path, int] = {}
    for line, row in enumerate(rows, start=2):
        if not any((v or "").strip() for v in row.values()):
            continue  # blank line
        item = _parse_row(row, line, docs_dir, errors)
        if item is None:
            continue
        if item["doc_id"] in seen_ids:
            errors.append(f"line {line}: duplicate doc_id {item['doc_id']!r} (first on line {seen_ids[item['doc_id']]})")
            continue
        if item["path"] in seen_paths:
            errors.append(f"line {line}: file already listed on line {seen_paths[item['path']]}")
            continue
        seen_ids[item["doc_id"]] = seen_paths[item["path"]] = line
        parsed.append((line, item))

    # Latest revision per family; two rows with the same family and version are ambiguous.
    latest: dict[str, tuple[int, ...]] = {}
    for line, item in parsed:
        key = _version_key(item["version"])
        if latest.get(item["family"]) == key:
            errors.append(f"line {line}: family {item['family']!r} already has version {item['version']}")
        latest[item["family"]] = max(key, latest.get(item["family"], key))

    if errors:
        raise ManifestError(errors)
    return [Document(**item, is_latest=_version_key(item["version"]) == latest[item["family"]])
            for _, item in parsed]


def find_unlisted(manifest_path: Path, docs_dir: Path) -> list[Path]:
    """Supported files under docs_dir that have no manifest row."""
    listed = set()
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8-sig", newline="") as f:
            listed = {(docs_dir / row["path"].strip()).resolve() for row in csv.DictReader(f)}
    return sorted(p for p in docs_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in DOC_TYPES and p.resolve() not in listed)


def draft_row(path: Path, docs_dir: Path) -> dict[str, str]:
    """Best-guess manifest row for a new file; category is left blank so validation forces a decision."""
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:60].rstrip("-")
    release = _RELEASE_IN_NAME.search(path.stem)
    release_text = f"R{release.group(1)}x" if release else ""
    return {
        "doc_id": slug, "path": path.relative_to(docs_dir).as_posix(),
        "title": re.sub(r"[_]+", " ", path.stem).strip(), "family": slug, "version": "1.0",
        "release_min": release_text, "release_max": release_text,
        "allowed_groups": "all", "external_ok": "false", "category": "",
    }


def append_rows(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    """Append rows to the manifest CSV, writing the header first if the file is new or empty."""
    new_file = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
