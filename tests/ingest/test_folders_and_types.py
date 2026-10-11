import csv
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from kb.core.config import Settings, get_settings
from kb.ingest import parse
from kb.ingest.manifest import (
    COLUMNS,
    doc_types,
    load_manifest,
    manifest_path_for,
    scan_folder,
)


def write_csv(path, rows):
    """A manifest file with the current header."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row(**overrides):
    """A valid manifest row."""
    base = {"doc_id": "doc-a", "path": "a.pdf", "title": "Doc A", "family": "", "version": "1.0",
            "release_min": "", "release_max": "", "allowed_groups": "all", "external_ok": "false",
            "category": "installation", "added": "", "review": ""}
    return base | overrides


@pytest.fixture
def doc_types_env(monkeypatch):
    """Set KB_DOC_TYPES for one test (settings are cached per process)."""
    def set_types(value):
        monkeypatch.setenv("KB_DOC_TYPES", value)
        get_settings.cache_clear()
    yield set_types
    monkeypatch.delenv("KB_DOC_TYPES", raising=False)
    get_settings.cache_clear()


def test_documents_outside_the_folder_keep_absolute_paths(tmp_path):
    docs, elsewhere = tmp_path / "documents", tmp_path / "Shared" / "TROUBLESHOOTING" / "New"
    for d in (docs, elsewhere / "Install"):
        d.mkdir(parents=True)
    (docs / "a.pdf").write_bytes(b"a")
    (elsewhere / "Install" / "Setup_Guide.docx").write_bytes(b"s")
    (elsewhere / "Copy of a.pdf").write_bytes(b"a")                     # same content as the listed a.pdf
    m = tmp_path / "manifest.csv"
    write_csv(m, [row()])
    report = scan_folder(m, docs, lambda p: p.read_bytes().hex(), folder=elsewhere)
    (draft,) = report.rows
    assert draft["path"] == (elsewhere / "Install" / "Setup_Guide.docx").resolve().as_posix()
    assert draft["category"] == "installation"                          # not 'troubleshooting' from the location
    assert report.duplicates == [("Copy of a.pdf", "manifest doc_id 'doc-a'")]
    assert report.by_folder() == {"Install": 1}
    write_csv(m, [row(), row(doc_id="setup", path=draft["path"], category="installation")])
    assert [d.path.name for d in load_manifest(m, docs)] == ["a.pdf", "Setup_Guide.docx"]
    assert manifest_path_for(docs / "a.pdf", docs) == "a.pdf"
    assert manifest_path_for(elsewhere / "Copy of a.pdf", docs).startswith(tmp_path.resolve().as_posix())


def test_doc_types_setting_filters_and_validates(tmp_path, doc_types_env):
    assert doc_types() == {".pdf": "pdf", ".pptx": "pptx", ".ppt": "pptx", ".docx": "docx", ".doc": "docx"}
    for name in ("a.pdf", "b.ppt", "c.doc", "d.xlsx"):
        (tmp_path / name).write_bytes(name.encode())
    assert sorted(r["path"] for r in scan_folder(tmp_path / "m.csv", tmp_path, lambda p: p.name).rows) == ["a.pdf", "b.ppt", "c.doc"]
    doc_types_env(" PDF, .docx ")
    assert get_settings().doc_types == "pdf,docx"
    assert [r["path"] for r in scan_folder(tmp_path / "m.csv", tmp_path, lambda p: p.name).rows] == ["a.pdf"]
    with pytest.raises(ValidationError, match="unsupported xlsx"):
        Settings(_env_file=None, KB_DOC_TYPES="pdf,xlsx")


def test_legacy_formats_are_converted_once_with_libreoffice(tmp_path, monkeypatch):
    src = tmp_path / "Old Deck.ppt"
    src.write_bytes(b"legacy")
    doc = load_one(tmp_path, src)
    assert doc.doc_type == "pptx"                                        # parsed as a deck
    monkeypatch.setattr(parse, "get_settings", lambda: get_settings().model_copy(update={"parsed_dir": tmp_path / "parsed"}))
    monkeypatch.setattr(parse, "find_soffice", lambda: None)
    with pytest.raises(RuntimeError, match="needs LibreOffice"):
        parse.convert_legacy(doc, "abc123")
    calls = []

    def fake_run(args, **kwargs):
        """Pretend to be soffice: write the converted file into --outdir."""
        calls.append(args)
        Path(args[args.index("--outdir") + 1], "Old Deck.pptx").write_bytes(b"converted")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(parse, "find_soffice", lambda: "soffice")
    monkeypatch.setattr(parse.subprocess, "run", fake_run)
    target = parse.convert_legacy(doc, "abc123def456aa")
    assert target.name == f"{doc.doc_id}.abc123def456.pptx" and target.read_bytes() == b"converted"
    assert parse.convert_legacy(doc, "abc123def456aa") == target and len(calls) == 1      # cached


def load_one(folder, path):
    """The Document for one file, through a one-row manifest."""
    m = folder / "manifest.csv"
    write_csv(m, [row(doc_id="old-deck", path=path.name, category="functional")])
    (doc,) = load_manifest(m, folder)
    return doc


def test_scan_goes_into_every_subfolder_and_skips_office_lock_files(tmp_path):
    """Files at any depth are drafted; '~$' owner files of open documents are not."""
    deep = tmp_path / "new" / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (tmp_path / "new" / "top.pdf").write_bytes(b"1")
    (deep / "deep.pptx").write_bytes(b"2")
    (deep / "~$deep.pptx").write_bytes(b"3")
    rows = scan_folder(tmp_path / "m.csv", tmp_path, lambda p: p.name, folder=tmp_path / "new").rows
    assert sorted(Path(r["path"]).name for r in rows) == ["deep.pptx", "top.pdf"]
