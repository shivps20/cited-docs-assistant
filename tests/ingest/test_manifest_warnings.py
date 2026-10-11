from kb.ingest.manifest import COLUMNS, manifest_warnings, scan_folder


def row(**overrides):
    """A manifest row as read_rows returns it (every column, strings)."""
    base = dict.fromkeys(COLUMNS, "") | {"version": "1.0", "allowed_groups": "ds-internal",
                                         "external_ok": "false", "category": "installation"}
    return base | overrides


def kinds(rows):
    """(kind, lines) of every warning for these rows."""
    return [(w.kind, w.lines) for w in manifest_warnings(rows)]


def test_a_deck_and_its_pdf_export_keep_the_deck():
    """Same name in two formats: one warning, which keeps the .pptx and names the PDF line to delete."""
    rows = [row(doc_id="a", path="E:/Docs/Guide v1.pdf"), row(doc_id="b", path="E:/Docs/Sub/Guide v1.pptx")]
    [w] = manifest_warnings(rows)
    assert (w.kind, w.lines) == ("other-format", (2, 3))
    assert "keep the .pptx (line 3) and delete line(s) 2" in w.message


def test_a_word_file_and_its_pdf_keep_the_pdf_for_its_page_numbers():
    """Word files cite page 1 everywhere (TD-27), so the PDF is the copy to keep."""
    [w] = manifest_warnings([row(doc_id="a", path="Case.docx"), row(doc_id="b", path="Case.pdf")])
    assert "keep the .pdf (line 3)" in w.message


def test_the_same_file_name_in_two_folders():
    """Same name and format in two folders is a different warning (which copy, or a newer edition?)."""
    rows = [row(doc_id="a", path="A/Deck.pptx"), row(doc_id="b", path="B/Deck.PPTX")]
    assert kinds(rows) == [("same-name", (2, 3))]


def test_internal_documents_open_to_all_but_not_documents_about_restricted_access():
    """INTERNAL / CONFIDENTIAL / DS_RESTRICTED open to all are flagged; 'Restricted access' is a feature name."""
    rows = [row(doc_id="a", path="Guide_INTERNAL_v1.pdf", allowed_groups="all"),
            row(doc_id="b", path="Plan_DS_RESTRICTED.pdf", allowed_groups="all;ds-internal"),
            row(doc_id="c", path="Restricted_Access_Management.pdf", allowed_groups="all"),
            row(doc_id="d", path="Internals of the server.pdf", allowed_groups="all"),
            row(doc_id="e", path="Tools_Confidential.pdf", allowed_groups="ds-internal")]
    assert kinds(rows) == [("access", (2,)), ("access", (3,))]


def test_latest_version_with_an_older_release_is_flagged_but_normal_succession_is_not():
    """v2 without a release over v1 R2019x contradicts the order; v1 R2020x+ then v2 R2021x+ is normal."""
    wrong = [row(doc_id="a2", path="Tips.pdf", family="tips", version="2.0"),
             row(doc_id="a1", path="Tips_V1.0.pdf", family="tips", version="1.0", release_min="R2019x")]
    normal = [row(doc_id="b2", path="Mail.pdf", family="mail", version="2", release_min="R2021x"),
              row(doc_id="b1", path="Mail_V1.0.pdf", family="mail", version="1", release_min="R2020x")]
    assert kinds(wrong) == [("releases", (2, 3))]
    assert kinds(normal) == []


def test_editions_in_different_families():
    """Same name apart from the version, same release, two families: both are searched as latest."""
    rows = [row(doc_id="a", path="Search_R2016x_V1_3.pdf", family="a", release_min="R2016x"),
            row(doc_id="b", path="Search_R2016x_V4.pdf", family="b", release_min="R2016x")]
    assert kinds(rows) == [("editions", (2, 3))]


def test_scan_drafts_only_the_preferred_format(tmp_path):
    """A new PDF whose deck is listed or new is skipped and reported, not drafted."""
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    for rel in ("Deck v1.pptx", "sub/Deck v1.pdf", "Listed.pdf", "Alone.pdf"):
        (root / rel).write_bytes(rel.encode())
    m = tmp_path / "manifest.csv"
    m.write_text(",".join(COLUMNS) + "\nlisted,Listed.pdf,Listed,,1.0,,,all,false,installation,,\n", encoding="utf-8")
    (root / "Listed.pptx").write_bytes(b"listed deck")
    report = scan_folder(m, root, lambda p: p.read_bytes().hex())
    assert sorted(r["path"] for r in report.rows) == ["Alone.pdf", "Deck v1.pptx", "Listed.pptx"]
    assert report.other_formats == [("sub/Deck v1.pdf", "new file 'Deck v1.pptx'")]
