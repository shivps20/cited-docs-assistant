"""Document manifest: the single source of truth for per-document metadata.

One CSV row per source file. Metadata that cannot be inferred reliably from the
files themselves (access groups, release applicability, revision lineage) lives
here, is validated on load, and is copied into every chunk at ingestion.

Columns:
    doc_id          unique slug, e.g. "install-guide"
    path            file path relative to the documents folder (KB_DOCS_DIR, subfolders included, e.g.
                    "Install/R2026x/guide.pdf"), or an absolute path for a document kept elsewhere
                    (e.g. "E:/Docs/New/guide.pdf", drafted by `kb manifest scan --folder`)
    title           human-readable title, used in citations
    family          logical document across revisions (defaults to doc_id)
    version         document revision, e.g. "3.2"; the highest per family is latest
    release_min     first release the document applies to, e.g. "R2015x" (blank = any)
    release_max     last release it applies to (blank = open-ended)
    allowed_groups  ";"-separated access groups; "all" = everyone
    external_ok     true/false: may chunks be sent to an external LLM (OpenAI)
    category        one of CATEGORIES
    added           date the row was added, YYYY-MM-DD (blank allowed)
    review          what still needs a human check, e.g. "category guessed from the name" (blank = reviewed)

`kb manifest scan` drafts rows for new files: doc id and family from the full file name (version suffixes
such as "_V2.0" become the version, so revisions share a family), the release from the name ("R2021x",
"21x") or, failing that, from the PDF's first pages, a category guessed from the name, and review notes
for every guess. A manifest without the last two columns (before 2026-10-10) is still read.
"""

import csv
import datetime
import hashlib
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from kb.core.config import get_settings

LEGACY_COLUMNS = ["doc_id", "path", "title", "family", "version", "release_min", "release_max",
                  "allowed_groups", "external_ok", "category"]
COLUMNS = [*LEGACY_COLUMNS, "added", "review"]
CATEGORIES = {"installation", "administration", "authentication", "infrastructure",
              "upgrade", "performance", "applications", "functional", "troubleshooting"}
# Extension → the format that is parsed: legacy .ppt / .doc are converted to .pptx / .docx first (kb.ingest.parse).
PARSED_FORMATS = {".pdf": "pdf", ".pptx": "pptx", ".ppt": "pptx", ".docx": "docx", ".doc": "docx"}
LEGACY_FORMATS = {".ppt": "pptx", ".doc": "docx"}


def doc_types() -> dict[str, str]:
    """The enabled extensions (KB_DOC_TYPES) and the format each is parsed as."""
    return {f".{t}": PARSED_FORMATS[f".{t}"] for t in get_settings().doc_types.split(",")}


def display_path(stored: str) -> str:
    """A source's file as answers show it (KB_SOURCE_PATH). stored = the manifest form (relative to
    KB_DOCS_DIR, or absolute). full: the absolute path in the system's own form; relative: the stored relative
    path, or only the file name for a document kept outside KB_DOCS_DIR (no disk layout revealed)."""
    if not stored:
        return ""
    settings = get_settings()
    path = Path(stored)
    if settings.source_path == "relative":
        return path.name if path.is_absolute() else stored
    return str(path if path.is_absolute() else (settings.docs_dir / path).resolve())


def manifest_path_for(path: Path, docs_dir: Path) -> str:
    """How a file is written in the manifest: relative to docs_dir when inside it, else absolute (posix)."""
    resolved, base = path.resolve(), docs_dir.resolve()
    return resolved.relative_to(base).as_posix() if resolved.is_relative_to(base) else resolved.as_posix()

# Releases are stored as their year (R2026x -> 2026) so ranges can be filtered numerically.
RELEASE_ANY_MIN = 0
RELEASE_ANY_MAX = 9999

_RELEASE = re.compile(r"^(?:V6)?R(\d{4})x$", re.IGNORECASE)
# A release in a file name or a cover page: R2021x, V6R2021x, 2021x, 21x, R21x, R2017xFP1705, R15xGA.
_RELEASE_TOKEN = re.compile(r"(?<![0-9])(?<![0-9]\.)(?:v6)?r?(?:20)?(1[3-9]|2\d|3[0-5])x(?=fp|ga|hf|[^a-z0-9]|$)", re.IGNORECASE)
# A version at the end of a file name: "_V2.0", "-v1", " V3.1 Internal", " Rev 3".
_VERSION_AT_END = re.compile(r"[\s_\-(.]+(?:v|rev\.?\s*)(\d+(?:[._]\d+)*)\)?(?:[\s_\-]+(?:internal|external|final|draft|en))*\s*$",
                             re.IGNORECASE)
# "ENOVIA V6", "CATIA V5": a product generation, not a document version.
_PRODUCT_GENERATION = re.compile(r"(?:enovia|catia|simulia|delmia|solidworks|3dexperience)[\s_\-]*$", re.IGNORECASE)
MAX_SLUG = 100          # longer names are shortened and get a short hash, so ids stay unique
# Category from words in the file name and folders; first match wins (measured on 221 labelled documents: 0.85).
CATEGORY_RULES = (
    ("troubleshooting", r"troubleshoot"),
    ("installation", r"install|deploy|fix ?pack|\bfp\d{4}|haproxy|load ?balancer|adfs|\bga\b|ssl between"),
    ("upgrade", r"upgrade|migrat"),
    ("performance", (r"performance (checklist|optimi|trace)|tuning|large assembl|crash|hang|memory dump|database locks?|"
                     r"timeout|security mask|r&d recommended|fiddler|columbo")),
    ("authentication", r"saml|\bsso\b|single sign|ldap|kerberos|authenticat|notification"),
    ("infrastructure", r"load balancing|security (checkpoint|privacy)|cloud security|\bf5\b"),
    ("administration", r"\bindex|catsettings|monitoring|maintenance|trace dictionary|admin|widget|backup|email|launcher"),
    ("functional", (r"\brole\b|\bfp\b|functional|presentation|deep dive|manager|costing|pricing|best ?practice|"
                    r"\b(ccm|cfg|chg|chp|csv|dey|dpm)\b|engineering|requirements|xpdm|design|catia|material")),
)
DEFAULT_CATEGORY = "administration"
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
    added: str = ""                     # YYYY-MM-DD the row was added ('' when unknown)
    review: str = ""                    # open review notes ('' = reviewed)

    @property
    def release_label(self) -> str:
        """This document's release range as text (see release_label())."""
        return release_label(self.release_min, self.release_max)

    def applies_to(self, release: int) -> bool:
        """Does the document apply to the given release year?"""
        return self.release_min <= release <= self.release_max


def releases_in(text: str) -> list[int]:
    """Release years named in a file name or text, in order of first mention: 'CCM 21x' -> [2021],
    'R2017xFP1705' -> [2017], 'from V6 to R15xGA' -> [2015]."""
    years: list[int] = []
    for two in _RELEASE_TOKEN.findall(text):
        year = 2000 + int(two)
        if year not in years:
            years.append(year)
    return years


def normalise_date(text: str) -> str | None:
    """A manifest date as YYYY-MM-DD: ISO as written by the tools, M/D/YYYY as Excel writes it with a US
    locale, D.M.YYYY as with a European one; '' stays ''; anything else → None (invalid)."""
    text = text.strip()
    if not text:
        return ""
    for pattern, order in ((r"(\d{4})-(\d{1,2})-(\d{1,2})", "ymd"), (r"(\d{1,2})/(\d{1,2})/(\d{4})", "mdy"),
                           (r"(\d{1,2})\.(\d{1,2})\.(\d{4})", "dmy")):
        match = re.fullmatch(pattern, text)
        if match:
            parts = dict(zip(order, map(int, match.groups()), strict=True))
            try:
                return datetime.date(parts["y"], parts["m"], parts["d"]).isoformat()
            except ValueError:
                return None
    return None


def version_in_name(stem: str) -> tuple[str, str | None]:
    """(name without the version, version) for a file name ending in a version: 'Guide_V2.0' -> ('Guide', '2.0');
    'Guide v1' -> ('Guide', '1.0'); no version at the end -> (stem, None)."""
    match = _VERSION_AT_END.search(stem)
    if not match or _PRODUCT_GENERATION.search(stem[:match.start() + 1]):
        return stem, None
    parts = match.group(1).replace("_", ".").split(".")
    version = ".".join(str(int(p)) for p in parts) + (".0" if len(parts) == 1 else "")
    return stem[:match.start()], version


def name_slug(text: str) -> str:
    """Lowercase slug of a whole name ('-' between words); longer than MAX_SLUG: shortened plus a short hash,
    so two long names that only differ at the end never share a slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) <= MAX_SLUG:
        return slug
    digest = hashlib.sha1(text.lower().encode("utf-8")).hexdigest()[:6]
    return f"{slug[:MAX_SLUG - 7].rstrip('-')}-{digest}"


def guess_category(text: str) -> str:
    """Category from words in a file name and its folders (CATEGORY_RULES, first match), else DEFAULT_CATEGORY."""
    words = re.sub(r"[_\-/\\]+", " ", text.lower())
    return next((category for category, pattern in CATEGORY_RULES if re.search(pattern, words)), DEFAULT_CATEGORY)


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
    path = docs_dir / rel_path                                 # an absolute path stays as it is
    types = doc_types()
    doc_type = types.get(path.suffix.lower())
    if not rel_path:
        err("path is empty")
    elif doc_type is None:
        err(f"unsupported file type {path.suffix!r} (KB_DOC_TYPES: {', '.join(sorted(types))})")
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

    added = normalise_date(row.get("added") or "")
    if added is None:
        err(f"added {row['added']!r} must be a date like 2026-10-10 (Excel's 10/10/2026 is accepted too)")
        added = ""

    if len(errors) > n_errors:
        return None
    return {
        "doc_id": doc_id, "path": path.resolve(), "title": title, "family": row["family"].strip() or doc_id,
        "version": version, "release_min": release_min, "release_max": release_max, "allowed_groups": groups,
        "external_ok": external_ok, "category": category, "doc_type": doc_type,
        "added": added, "review": (row.get("review") or "").strip(),
    }


def load_manifest(manifest_path: Path, docs_dir: Path) -> list[Document]:
    """Load and validate the manifest. Raises ManifestError listing every problem found."""
    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if header not in (COLUMNS, LEGACY_COLUMNS):
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
    first: dict[tuple[str, tuple[int, ...]], tuple[int, str]] = {}
    for line, item in parsed:
        key = _version_key(item["version"])
        clash = first.get((item["family"], key))
        if clash is not None:
            errors.append(
                f"line {line}: family {item['family']!r} already has version {item['version']} "
                f"(line {clash[0]}: {clash[1]}; this row: {item['path'].name}). If this file is a newer revision, "
                f"give it a higher version; if it is a different document, give it a family of its own "
                f"(e.g. its doc_id).")
        first.setdefault((item["family"], key), (line, item["path"].name))
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
    types = doc_types()
    return sorted(p for p in docs_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in types and p.resolve() not in listed)


def draft_row(path: Path, docs_dir: Path, *, today: str = "", cover_text: str = "",
              scan_root: Path | None = None) -> dict[str, str]:
    """Best-guess row for a new file, with every guess listed in `review`.

    family = slug of the whole name without its version; doc_id = family plus '-v<version>' when the name
    ends in a version. Release: from the name, otherwise from the cover text; always 'from that release on'
    (release_min only, release_max left open: a guide stays valid for later releases unless the owner sets an
    end); several releases on the cover are only noted. Category: guessed from the name and folders.
    Access: everyone, not external (to be reviewed). The path is relative to docs_dir when the file is inside
    it, else absolute; the category is guessed from the path below scan_root only (the folder scanned), so
    words in the folder's own location ("…/TROUBLESHOOTING/…") do not count."""
    rel = manifest_path_for(path, docs_dir)
    below_root = path.resolve().relative_to((scan_root or docs_dir).resolve()).as_posix()
    base, version = version_in_name(path.stem)
    family = name_slug(base) or "document"
    notes = []
    release_min = release_max = ""
    named = releases_in(base)
    if named:
        release_min = f"R{min(named)}x"
        notes.append(f"release from the name: R{min(named)}x onwards")
        if len(named) > 1:
            notes.append(f"several releases in the name ({', '.join(f'R{y}x' for y in named)}): check the range")
    else:
        cover = releases_in(cover_text)
        if len(cover) == 1:
            release_min = f"R{cover[0]}x"
            notes.append(f"release from the first pages: R{cover[0]}x onwards")
        elif cover:
            notes.append(f"first pages name {', '.join(f'R{y}x' for y in cover)}: set the release range")
    notes.append("category guessed from the name")
    return {
        "doc_id": f"{family}-v{version.replace('.', '-')}" if version else family, "path": rel,
        "title": re.sub(r"[_]+", " ", path.stem).strip(), "family": family, "version": version or "1.0",
        "release_min": release_min, "release_max": release_max, "allowed_groups": "all", "external_ok": "false",
        "category": guess_category(below_root), "added": today, "review": "; ".join(notes),
        "_named_version": "yes" if version else "",
    }


@dataclass
class ScanReport:
    """What `kb manifest scan` found in the folder it scanned (the documents folder or --folder)."""
    docs_dir: Path                                   # the folder scanned
    supported: int                                   # supported files under docs_dir
    listed: int                                      # of those, already in the manifest
    rows: list[dict[str, str]] = field(default_factory=list)          # draft rows for new files
    duplicates: list[tuple[str, str]] = field(default_factory=list)   # (new file, what it repeats), not drafted

    def by_folder(self) -> dict[str, int]:
        """Number of new files per subfolder of the scanned folder ('.' = the folder itself), sorted."""
        counts: dict[str, int] = {}
        root = self.docs_dir.resolve().as_posix().rstrip("/") + "/"
        for row in self.rows:
            path = row["path"].removeprefix(root)
            folder = path.rsplit("/", 1)[0] if "/" in path else "."
            counts[folder] = counts.get(folder, 0) + 1
        return dict(sorted(counts.items()))


def _unique_id(slug: str, rel_path: str, taken: set[str]) -> str:
    """slug, or (for a name already used, e.g. the same file name in two subfolders) slug prefixed with its
    folder's name, or numbered: doc ids must stay unique across the whole tree."""
    if slug not in taken:
        return slug
    parent = rel_path.rsplit("/", 2)[-2] if rel_path.count("/") >= 1 else ""
    folder = re.sub(r"[^a-z0-9]+", "-", parent.lower()).strip("-")
    candidate = f"{folder}-{slug}"[:60].rstrip("-") if folder else slug
    n = 2
    while candidate in taken:
        candidate = f"{slug[:56]}-{n}"
        n += 1
    return candidate


def _note(row: dict[str, str], text: str) -> None:
    """Add a review note to a drafted row."""
    row["review"] = "; ".join(filter(None, [row.get("review", ""), text]))


def _resolve_families(rows: list[dict[str, str]], listed_rows: list[dict[str, str]]) -> None:
    """Make drafted families valid without ever hiding a document already in the manifest.

    - A new file without a version in its name whose family already exists in the manifest gets a family of
      its own (with a note): joining would make the listed document "not latest", i.e. hide it from search.
    - Among new files of one family, a file without a version next to versioned ones is drafted as the
      newest (with a note); a second unversioned file gets a family of its own.
    - A version in the name that the family already has → family of its own (with a note); a new version
      of an existing family joins it (the version in the name makes that safe).
    - A family that looks like the longer form of an existing (shortened, pre-2026-10-10) family is noted.
    Rows to be split are marked `_split`; scan_folder() then gives them a unique doc id as their family."""
    existing: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for r in listed_rows:
        version = (r.get("version") or "").strip()
        if _VERSION.match(version):
            existing[(r.get("family") or "").strip() or r["doc_id"].strip()].append(_version_key(version))
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for family, group in by_family.items():
        named = [r for r in group if r["_named_version"]]
        sibling_folders = {r["path"].rpartition("/")[0] for r in named}
        # the unversioned file next to its versioned siblings is the likely newest; then path order
        plain = sorted((r for r in group if not r["_named_version"]),
                       key=lambda r: (r["path"].rpartition("/")[0] not in sibling_folders, r["path"]))
        for row in named:
            if _version_key(row["version"]) in existing.get(family, []):
                row["_split"] = "yes"
                _note(row, f"family {family!r} already has version {row['version']}: own family drafted; check")
        for i, row in enumerate(plain):
            if family in existing:
                row["_split"] = "yes"
                _note(row, f"same name as manifest family {family!r}: own family drafted; if it is a newer "
                           f"revision, set family {family!r} and a higher version")
            elif i == 0 and named:
                row["version"] = f"{max(_version_key(r['version']) for r in named)[0] + 1}.0"
                _note(row, "version guessed: newest (no version in the name)")
            elif i > 0:
                row["_split"] = "yes"
                _note(row, "same name as another new file: own family drafted; set family and version if it is a revision")
    for row in rows:
        for other in existing:
            if other != row["family"] and len(other) >= 50 and row["family"].startswith(other):
                _note(row, f"may be a revision of family {other!r}: check family and version")
                break
        row.pop("_named_version", None)


def scan_folder(manifest_path: Path, docs_dir: Path, file_hash: Callable[[Path], str], *, today: str = "",
                cover_text: Callable[[Path], str] = lambda path: "", folder: Path | None = None) -> ScanReport:
    """Draft rows for supported files under `folder` (default: docs_dir) and its subfolders that are not in the
    manifest. Files inside docs_dir get paths relative to it; files elsewhere get absolute paths.

    A new file whose content (file_hash) is already in the manifest, or that repeats another new file, is
    not drafted but reported as a duplicate, so the same document is not indexed twice under two paths.
    Doc ids stay unique across the tree; families are resolved (_resolve_families); cover_text(path) is only
    read for files without a release in their name. Nothing is written (see append_rows)."""
    listed_rows = []
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8-sig", newline="") as f:
            listed_rows = [r for r in csv.DictReader(f) if r.get("path", "").strip()]
    listed_paths = {(docs_dir / r["path"].strip()).resolve(): r["doc_id"].strip() for r in listed_rows}
    root = folder or docs_dir
    types = doc_types()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in types)
    report = ScanReport(docs_dir=root, supported=len(files),
                        listed=sum(1 for p in files if p.resolve() in listed_paths))
    known: dict[str, str] = {}                         # content hash -> what holds it
    for path, doc_id in listed_paths.items():
        if path.is_file():
            known.setdefault(file_hash(path), f"manifest doc_id '{doc_id}'")
    taken = {r["doc_id"].strip() for r in listed_rows}
    for path in files:
        if path.resolve() in listed_paths:
            continue
        rel = path.relative_to(root).as_posix()
        digest = file_hash(path)
        if digest in known:
            report.duplicates.append((rel, known[digest]))
            continue
        known[digest] = f"new file '{rel}'"
        needs_cover = not releases_in(version_in_name(path.stem)[0])
        report.rows.append(draft_row(path, docs_dir, today=today, cover_text=cover_text(path) if needs_cover else "",
                                     scan_root=root))
    _resolve_families(report.rows, listed_rows)
    # unique doc ids: rows that keep the shared family first, then the rows split into a family of their own
    for row in sorted(report.rows, key=lambda r: bool(r.get("_split"))):
        row["doc_id"] = _unique_id(row["doc_id"], row["path"], taken)
        taken.add(row["doc_id"])
        if row.pop("_split", ""):
            row["family"] = row["doc_id"]
    return report


def read_rows(manifest_path: Path) -> list[dict[str, str]]:
    """The manifest's rows as dicts with every COLUMNS key (missing ones empty), in file order; dates that
    Excel rewrote (10/8/2026) come back as YYYY-MM-DD, so a rewrite by the tools restores them."""
    with manifest_path.open(encoding="utf-8-sig", newline="") as f:
        rows = [{c: (r.get(c) or "") for c in COLUMNS} for r in csv.DictReader(f)]
    for row in rows:
        row["added"] = normalise_date(row["added"]) or row["added"]
    return rows


def write_rows(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    """Rewrite the manifest with the current header and these rows."""
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plan_backfill(rows: list[dict[str, str]], docs_dir: Path, added_for: Callable[[dict[str, str]], str],
                  cover_text: Callable[[Path], str]) -> list[str]:
    """Fill what older rows lack, in place, and describe each change: `added` (added_for(row)) and, for rows
    without any release, the release from the file name or the first pages, as 'from that release on'
    (release_max left open); several releases on the first pages are only noted. Category, access and
    versions are not touched."""
    changes = []
    for row in rows:
        doc = row["doc_id"]
        if not row["added"]:
            row["added"] = added_for(row)
            if row["added"]:
                changes.append(f"{doc}: added = {row['added']}")
        if row["release_min"].strip() or row["release_max"].strip():
            continue
        path = docs_dir / row["path"]
        named = releases_in(version_in_name(path.stem)[0])
        if named:
            row["release_min"] = f"R{min(named)}x"
            _note(row, f"release from the name: R{min(named)}x onwards")
            changes.append(f"{doc}: release {row['release_min']}+ (from the name)")
            continue
        cover = releases_in(cover_text(path))
        if len(cover) == 1:
            row["release_min"] = f"R{cover[0]}x"
            _note(row, f"release from the first pages: R{cover[0]}x onwards")
            changes.append(f"{doc}: release R{cover[0]}x+ (from the first pages)")
        elif cover:
            _note(row, f"first pages name {', '.join(f'R{y}x' for y in cover)}: set the release range")
            changes.append(f"{doc}: review note only (first pages name {', '.join(f'R{y}x' for y in cover)})")
    return changes


def append_rows(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    """Append rows to the manifest CSV, writing the header first if the file is new or empty; a manifest with
    the older header (no added / review columns) is rewritten with the current one first."""
    if manifest_path.exists() and manifest_path.stat().st_size:
        with manifest_path.open(encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            write_rows(manifest_path, read_rows(manifest_path))
    new_file = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
