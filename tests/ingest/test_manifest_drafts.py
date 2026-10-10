import csv

import pytest

from kb.ingest.manifest import (
    COLUMNS,
    LEGACY_COLUMNS,
    ManifestError,
    append_rows,
    draft_row,
    guess_category,
    load_manifest,
    name_slug,
    plan_backfill,
    read_rows,
    releases_in,
    scan_folder,
    version_in_name,
)


def write_csv(path, header, rows):
    """A manifest file with the given header and rows (dicts)."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row(**overrides):
    """A valid manifest row."""
    base = {"doc_id": "doc-a", "path": "a.pdf", "title": "Doc A", "family": "", "version": "1.0",
            "release_min": "", "release_max": "", "allowed_groups": "all", "external_ok": "false",
            "category": "installation", "added": "", "review": ""}
    return base | overrides


def make(root, files):
    """Create files (relative path → bytes) under root."""
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(content)


def content_hash(path):
    """The file's bytes as its 'hash'."""
    return path.read_bytes().hex()


def test_releases_from_names_and_text():
    assert releases_in("CCM 21x ENOVIA Role FP") == [2021]
    assert releases_in("3DEXPERIENCE R2017xFP1705 Linux Oracle") == [2017]
    assert releases_in("upgrade-overview-from-v6-to-r15xga-v3") == [2015]
    assert releases_in("fix-pack-installation-3dexperience24x") == [2024]
    assert releases_in("Connector.22x.FD02") == [2022]
    assert releases_in("V6R2013x and R2026x") == [2013, 2026]
    assert releases_in("10x faster, 1.15x more") == []


def test_version_at_the_end_of_a_name():
    assert version_in_name("Guide_V2.0") == ("Guide", "2.0")
    assert version_in_name("Guide v1") == ("Guide", "1.0")
    assert version_in_name("Timeout-Settings V1.0 Internal") == ("Timeout-Settings", "1.0")
    assert version_in_name("Document Manager - QUC-OC Rev 3") == ("Document Manager - QUC-OC", "3.0")
    assert version_in_name("Upgrade from ENOVIA V6") == ("Upgrade from ENOVIA V6", None)    # a product generation
    assert version_in_name("Overview from ENOVIA V6 to R2018x V1") == ("Overview from ENOVIA V6 to R2018x", "1.0")


def test_slugs_are_full_length_and_long_ones_stay_unique():
    a = "DS_WhitePapers_3DEXPERIENCE-R2015x-AdaptFullTextSearchwithyourFormerTypingconfiguration"
    b = "DS_WhitePapers_3DEXPERIENCE-R2015x-AdaptFullTextSearchwithyourUnifiedTypingconfiguration"
    assert name_slug(a) != name_slug(b) and name_slug(a).endswith("formertypingconfiguration")
    long_a, long_b = "x" * 120 + "a", "x" * 120 + "b"
    assert len(name_slug(long_a)) == 100 and name_slug(long_a) != name_slug(long_b)
    assert name_slug("Configuring-Secure_Socket Layer") == name_slug("Configuring Secure-Socket_Layer")


def test_category_guesses():
    assert guess_category("R2024x Windows Oracle DB On Premise Installation") == "installation"
    assert guess_category("Overview of Upgrade from ENOVIA V6") == "upgrade"
    assert guess_category("Troubleshooting-3D-Passport-in-Federated-SAML-Mode") == "troubleshooting"
    assert guess_category("CHG 21x ENOVIA Role FP") == "functional"
    assert guess_category("3DSpace Index Best Practices") == "administration"
    assert guess_category("Something unknown") == "administration"


def test_draft_row_uses_the_version_release_and_cover(tmp_path):
    make(tmp_path, {"Server/File_Collaboration_Server_V2.0.pdf": b"x", "Launcher.pdf": b"y", "Many.pdf": b"z"})
    d = draft_row(tmp_path / "Server/File_Collaboration_Server_V2.0.pdf", tmp_path, today="2026-10-10")
    assert (d["doc_id"], d["family"], d["version"]) == ("file-collaboration-server-v2-0", "file-collaboration-server", "2.0")
    assert (d["added"], d["category"], d["release_min"]) == ("2026-10-10", "administration", "")
    c = draft_row(tmp_path / "Launcher.pdf", tmp_path, cover_text="Version 2.0 - for R2015x and later")
    assert (c["release_min"], c["release_max"]) == ("R2015x", "")                  # from that release on
    assert "release from the first pages: R2015x onwards" in c["review"]
    m = draft_row(tmp_path / "Many.pdf", tmp_path, cover_text="R2019x ... R2023x")
    assert m["release_min"] == "" and "first pages name R2019x, R2023x" in m["review"]


def test_revisions_share_a_family_and_never_hide_listed_documents(tmp_path):
    root = tmp_path / "docs"
    make(root, {"Usage_V1.0.pdf": b"1", "Usage.pdf": b"2", "Old/Usage.pdf": b"3",
                "Listed.pdf": b"4", "New/Listed.pdf": b"5", "Report_V1.0.pdf": b"6", "Report_V3.0.pdf": b"7"})
    m = tmp_path / "manifest.csv"
    write_csv(m, COLUMNS, [row(doc_id="listed", path="Listed.pdf"),
                           row(doc_id="report-v1-0", path="Report_V1.0.pdf", family="report", version="1.0")])
    rows = {r["path"]: r for r in scan_folder(m, root, content_hash, today="2026-10-10").rows}
    assert (rows["Usage_V1.0.pdf"]["family"], rows["Usage_V1.0.pdf"]["version"]) == ("usage", "1.0")
    assert (rows["Usage.pdf"]["family"], rows["Usage.pdf"]["version"]) == ("usage", "2.0")     # newest of the new
    assert "version guessed: newest" in rows["Usage.pdf"]["review"]
    assert rows["Old/Usage.pdf"]["family"] == rows["Old/Usage.pdf"]["doc_id"] != "usage"       # second unversioned
    assert rows["New/Listed.pdf"]["family"] == rows["New/Listed.pdf"]["doc_id"] != "listed"    # listed one stays latest
    assert (rows["Report_V3.0.pdf"]["family"], rows["Report_V3.0.pdf"]["version"]) == ("report", "3.0")  # joins
    assert all("_named_version" not in r for r in rows.values())


def test_older_header_still_loads_and_append_upgrades_it(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"a")
    m = tmp_path / "manifest.csv"
    write_csv(m, LEGACY_COLUMNS, [row()])
    (doc,) = load_manifest(m, tmp_path)
    assert (doc.added, doc.review) == ("", "")
    (tmp_path / "b.pdf").write_bytes(b"b")
    append_rows(m, [row(doc_id="doc-b", path="b.pdf", added="2026-10-10", review="category guessed from the name")])
    assert m.read_text(encoding="utf-8").splitlines()[0] == ",".join(COLUMNS)
    docs = load_manifest(m, tmp_path)
    assert [(d.doc_id, d.added, d.review) for d in docs] == [("doc-a", "", ""), ("doc-b", "2026-10-10", "category guessed from the name")]


def test_validation_of_added_and_a_helpful_family_clash_message(tmp_path):
    for name in ("a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(name.encode())
    m = tmp_path / "manifest.csv"
    write_csv(m, COLUMNS, [row(added="last week"), row(doc_id="doc-b", path="b.pdf", family="doc-a")])
    with pytest.raises(ManifestError) as e:
        load_manifest(m, tmp_path)
    assert any("added 'last week' must be a date" in x for x in e.value.errors)
    write_csv(m, COLUMNS, [row(), row(doc_id="doc-b", path="b.pdf", family="doc-a")])
    with pytest.raises(ManifestError) as e:
        load_manifest(m, tmp_path)
    assert "line 2: a.pdf; this row: b.pdf" in e.value.errors[0] and "give it a higher version" in e.value.errors[0]


def test_backfill_fills_added_and_missing_releases_only(tmp_path):
    make(tmp_path, {"Guide_R2021x.pdf": b"1", "Cover.pdf": b"2", "Several.pdf": b"3", "Set.pdf": b"4", "None.pdf": b"5"})
    m = tmp_path / "manifest.csv"
    write_csv(m, LEGACY_COLUMNS, [row(doc_id="named", path="Guide_R2021x.pdf"), row(doc_id="cover", path="Cover.pdf"),
                                  row(doc_id="several", path="Several.pdf"),
                                  row(doc_id="set", path="Set.pdf", release_min="R2024x", release_max="R2024x"),
                                  row(doc_id="none", path="None.pdf")])
    rows = read_rows(m)
    covers = {"Cover.pdf": "written for R2018x", "Several.pdf": "R2019x and R2023x", "Set.pdf": "R2015x", "None.pdf": ""}
    changes = plan_backfill(rows, tmp_path, lambda r: "2026-10-04", lambda p: covers.get(p.name, ""))
    by_id = {r["doc_id"]: r for r in rows}
    assert all(r["added"] == "2026-10-04" for r in rows)
    assert (by_id["named"]["release_min"], by_id["named"]["release_max"]) == ("R2021x", "")   # max left open
    assert (by_id["cover"]["release_min"], by_id["cover"]["release_max"]) == ("R2018x", "")
    assert by_id["several"]["release_min"] == "" and "R2019x, R2023x" in by_id["several"]["review"]
    assert by_id["set"]["release_min"] == "R2024x" and by_id["set"]["review"] == ""               # untouched
    assert by_id["none"]["release_min"] == "" and by_id["none"]["review"] == ""
    assert sum(": release" in c for c in changes) == 2


def test_dates_rewritten_by_excel_are_accepted_and_restored(tmp_path):
    from kb.ingest.manifest import normalise_date

    assert [normalise_date(t) for t in ("2026-10-08", "10/8/2026", "8.10.2026", "")] == ["2026-10-08"] * 3 + [""]
    assert normalise_date("13/13/2026") is None and normalise_date("yesterday") is None
    (tmp_path / "a.pdf").write_bytes(b"a")
    m = tmp_path / "manifest.csv"
    write_csv(m, COLUMNS, [row(added="10/8/2026")])
    (doc,) = load_manifest(m, tmp_path)
    assert doc.added == "2026-10-08" and read_rows(m)[0]["added"] == "2026-10-08"
