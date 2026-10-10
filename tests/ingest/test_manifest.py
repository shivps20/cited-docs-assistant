import csv
import json

import pytest

from kb.core.config import get_settings
from kb.ingest.manifest import (
    COLUMNS,
    RELEASE_ANY_MAX,
    RELEASE_ANY_MIN,
    ManifestError,
    draft_row,
    find_unlisted,
    load_manifest,
    parse_release,
    release_label,
    scan_folder,
)


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def row(**overrides):
    base = {"doc_id": "doc-a", "path": "a.pdf", "title": "Doc A", "family": "", "version": "1.0",
            "release_min": "R2026x", "release_max": "R2026x", "allowed_groups": "all", "external_ok": "false",
            "category": "installation"}
    return base | overrides


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for name in ("a.pdf", "a_v2.pdf", "b.pptx", "notes.txt"):
        (d / name).write_bytes(b"x")
    return d


def test_parse_release():
    assert parse_release("R2026x") == 2026
    assert parse_release("V6R2013x") == 2013
    assert parse_release(" r2015X ") == 2015
    with pytest.raises(ValueError):
        parse_release("2026x")


def test_release_label():
    assert release_label(2026, 2026) == "R2026x"
    assert release_label(2015, RELEASE_ANY_MAX) == "R2015x+"
    assert release_label(2017, 2026) == "R2017x-R2026x"
    assert release_label(RELEASE_ANY_MIN, RELEASE_ANY_MAX) == "any"


def test_valid_manifest(tmp_path, docs_dir):
    m = tmp_path / "manifest.csv"
    write_manifest(m, [row(), row(doc_id="doc-b", path="b.pptx", release_min="R2015x", release_max="",
                                  allowed_groups="eng; internal", external_ok="yes")])
    a, b = load_manifest(m, docs_dir)
    assert (a.doc_type, a.family, a.is_latest) == ("pdf", "doc-a", True)
    assert b.doc_type == "pptx"
    assert b.allowed_groups == ("eng", "internal")
    assert b.external_ok is True
    assert b.applies_to(2025) and b.applies_to(2030) and not b.applies_to(2014)


def test_is_latest_per_family(tmp_path, docs_dir):
    m = tmp_path / "manifest.csv"
    write_manifest(m, [row(family="fam", version="1.10"), row(doc_id="doc-a2", path="a_v2.pdf", family="fam", version="1.9")])
    v110, v19 = load_manifest(m, docs_dir)
    assert v110.is_latest and not v19.is_latest  # numeric, not string, comparison


def test_collects_all_errors(tmp_path, docs_dir):
    m = tmp_path / "manifest.csv"
    write_manifest(m, [
        row(doc_id="Bad ID"),
        row(doc_id="doc-b", path="missing.pdf"),
        row(doc_id="doc-c", path="notes.txt"),
        row(doc_id="doc-d", path="b.pptx", release_min="R2026x", release_max="R2024x"),
        row(doc_id="doc-e", path="a_v2.pdf", category="misc", external_ok="maybe", allowed_groups=""),
    ])
    with pytest.raises(ManifestError) as e:
        load_manifest(m, docs_dir)
    text = "\n".join(e.value.errors)
    for expected in ("doc_id 'Bad ID'", "file not found", "unsupported file type '.txt'",
                     "is after release_max", "category 'misc'", "external_ok 'maybe'", "allowed_groups is empty"):
        assert expected in text


def test_duplicates_rejected(tmp_path, docs_dir):
    m = tmp_path / "manifest.csv"
    write_manifest(m, [row(), row(path="a_v2.pdf"), row(doc_id="doc-z")])
    with pytest.raises(ManifestError) as e:
        load_manifest(m, docs_dir)
    assert "duplicate doc_id 'doc-a'" in e.value.errors[0]
    assert "already listed" in e.value.errors[1]


def test_same_family_same_version_rejected(tmp_path, docs_dir):
    m = tmp_path / "manifest.csv"
    write_manifest(m, [row(family="fam"), row(doc_id="doc-a2", path="a_v2.pdf", family="fam")])
    with pytest.raises(ManifestError, match="already has version 1.0"):
        load_manifest(m, docs_dir)


def test_wrong_header(tmp_path, docs_dir):
    m = tmp_path / "manifest.csv"
    m.write_text("doc_id,path\nx,a.pdf\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="header must be exactly"):
        load_manifest(m, docs_dir)


def test_scan_drafts(tmp_path, docs_dir):
    m = tmp_path / "manifest.csv"
    write_manifest(m, [row()])
    unlisted = find_unlisted(m, docs_dir)
    assert [p.name for p in unlisted] == ["a_v2.pdf", "b.pptx"]
    (docs_dir / "Acme_Setup_R2025x.pdf").write_bytes(b"x")
    draft = draft_row(docs_dir / "Acme_Setup_R2025x.pdf", docs_dir)
    assert draft["doc_id"] == "acme-setup-r2025x"
    assert (draft["release_min"], draft["release_max"], draft["category"]) == ("R2025x", "R2025x", "")



def content_hash(path):
    """The file's bytes as its 'hash' (enough to tell test files apart)."""
    return path.read_bytes().hex()


def test_scan_walks_subfolders_skips_duplicates_and_keeps_ids_unique(tmp_path):
    root = tmp_path / "library"
    for rel, content in {"guide.pdf": b"listed", "Install/R2026x/guide.pdf": b"new guide",
                         "Install/R2026x/Setup_R2026x.docx": b"setup", "Old/copy of guide.pdf": b"listed",
                         "Old/twin.pptx": b"setup", "Old/notes.txt": b"ignored"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(content)
    m = tmp_path / "manifest.csv"
    write_manifest(m, [row(doc_id="guide", path="guide.pdf")])
    report = scan_folder(m, root, content_hash)
    assert (report.supported, report.listed) == (5, 1)
    assert [(r["doc_id"], r["path"]) for r in report.rows] == [
        ("r2026x-guide", "Install/R2026x/guide.pdf"),         # same name as a listed document: folder prefix
        ("setup-r2026x", "Install/R2026x/Setup_R2026x.docx")]
    assert report.rows[0]["family"] == "r2026x-guide" and report.rows[1]["release_min"] == "R2026x"
    assert report.duplicates == [("Old/copy of guide.pdf", "manifest doc_id 'guide'"),
                                 ("Old/twin.pptx", "new file 'Install/R2026x/Setup_R2026x.docx'")]
    assert report.by_folder() == {"Install/R2026x": 2}
    assert len(m.read_text(encoding="utf-8").splitlines()) == 2      # nothing written by the scan itself


def test_scan_of_an_absolute_folder_without_a_manifest(tmp_path):
    root = tmp_path / "elsewhere"
    root.mkdir()
    (root / "a.pdf").write_bytes(b"a")
    report = scan_folder(tmp_path / "missing.csv", root.resolve(), content_hash)
    assert (report.supported, report.listed, [r["path"] for r in report.rows]) == (1, 0, ["a.pdf"])
    assert report.by_folder() == {".": 1}

_LOCAL = get_settings()


@pytest.mark.skipif(not (_LOCAL.docs_dir.exists() and _LOCAL.manifest_path.exists() and _LOCAL.golden_path.exists()),
                    reason="local project data (documents, manifest.csv, golden set) not present")
def test_project_manifest_matches_golden_set():
    s = get_settings()
    docs = {d.doc_id: d for d in load_manifest(s.manifest_path, s.docs_dir)}
    golden = json.loads(s.golden_path.read_text(encoding="utf-8"))
    cited = {src["doc_id"] for q in golden for src in q["sources"]}
    assert cited <= docs.keys(), f"golden set cites unknown documents: {cited - docs.keys()}"
