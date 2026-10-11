import pytest

from kb.ingest import structure
from kb.ingest.structure import (
    Element,
    Section,
    Structure,
    build_structure,
    normalize_line,
    normalize_title,
    parse_toc,
    plausible_next,
    split_number,
    structure_for,
    table_markdown,
)


def E(kind, text="", page=1, **kw):
    return Element(kind, text, page, **kw)


def test_split_number_and_normalize():
    assert split_number("3.1. The config.xml File") == ("3.1", "The config.xml File")
    assert split_number("10.10. Compiling all Modules") == ("10.10", "Compiling all Modules")
    assert split_number("Acme Launcher") == (None, "Acme Launcher")
    assert split_number("R2026x Release") == (None, "R2026x Release")
    assert normalize_title("1.1. Advanced Product Quality 6 Planning (APQP)") == "advanced product quality planning apqp"
    assert normalize_title("Search Index - On Premise") == "search index on premise"


def test_parse_toc_handles_leaders_merged_rows_and_stray_pages():
    entries = parse_toc([
        "1. Introduction ...................... 4",
        "2.1. What is Acme Launcher ..... 4",
        "On Cloud � 8",
        "Using Control Plan and PFMEA ...... 4. Comparison of PFMEA Management in Excel vs Acme Platform ...... 16",
        "1.1. Advanced Product Quality 6 Planning (APQP) ...... 6",
        "Table of contents ...... 3",
    ])
    assert [(e.number, e.title) for e in entries] == [
        ("1", "Introduction"),
        ("2.1", "What is Acme Launcher"),
        (None, "On Cloud"),
        (None, "Using Control Plan and PFMEA"),
        ("4", "Comparison of PFMEA Management in Excel vs Acme Platform"),
        ("1.1", "Advanced Product Quality 6 Planning (APQP)"),
    ]
    assert entries[-1].key == "advanced product quality planning apqp"


def test_plausible_next():
    assert plausible_next(None, (1,))
    assert not plausible_next(None, (3,))
    assert plausible_next((4, 4, 1), (4, 4, 2))
    assert plausible_next((4, 4, 1), (4, 5))
    assert plausible_next((4, 4, 1), (5,))
    assert plausible_next((3, 1), (3, 1, 1))
    assert plausible_next((3, 1), (3, 3))          # one missed heading
    assert not plausible_next((3, 1), (3, 4))
    assert not plausible_next((4, 4, 1), (1,))     # a numbered step, not a section


def sample_pdf_elements():
    toc_rows = [["1.", "Introduction ..........", "4"],
                ["3.", "Installing .........", "8"],
                ["3.1.", "Prerequisites .......", "8"],
                ["", "On Cloud .......", "8"],
                ["", "On Premise .......", "8"],
                ["4.", "Uninstalling ........", "10"],
                ["10.", "Document History ......", "22"]]
    return [
        E("heading", "Best Practices", 1),                       # leaked page header
        E("heading", "Acme Launcher Services", 1),
        E("heading", "Executive Summary", 2),
        E("text", "This document is applicable to all releases from R2015x and above.", 2),
        E("heading", "Table of contents", 3),
        E("toc", "", 3, rows=toc_rows),
        E("heading", "1. Introduction", 4),
        E("text", "The Launcher is a background service.", 4),
        E("text", "© Acme Corp | Confidential Information | ref.: ACME_Document_2020", 4),
        E("heading", "3. Installing", 8),
        E("heading", "3.1. Prerequisites", 8),
        E("heading", "On Cloud", 8),
        E("list", "Run the cloud eligibility checker", 8, depth=3),
        E("list", "Download the tool", 8, depth=4),
        E("heading", "For example:", 8),
        E("text", "On Premise", 9),                               # heading Docling labelled as text
        E("text", "Refer to the Program Directory.", 9),
        E("heading", "1. Unzip the media file", 9),                # numbered step, not a section
        E("heading", "4. Uninstalling", 10),
        E("code", 'wmic product where name="Acme Launcher" call uninstall', 10),
        E("table", "", 10, rows=[["Param", "o o", "Description"], ["indexdepth", "o", "Default: 6"]]),
        E("heading", "10. Document History", 22),
        E("text", "1.0 01/15/2021 Original Document", 22),
    ]


def test_build_structure_numbers_sections_from_toc():
    st = build_structure(sample_pdf_elements(), doc_type="pdf", n_pages=22,
                         furniture={normalize_line("Best Practices")})
    assert [s.number for s in st.sections] == ["0", "1", "3", "3.1", "3.1.1", "3.1.2", "4"]

    front = st.get("0")
    assert front.title == "Executive Summary"
    assert any("R2015x and above" in b.text for b in front.blocks)

    on_cloud = st.get("3.1.1")
    assert (on_cloud.title, on_cloud.parent, on_cloud.level) == ("On Cloud", "3.1", 3)
    assert on_cloud.blocks[0].kind == "list"
    assert on_cloud.blocks[0].text == "- Run the cloud eligibility checker\n  - Download the tool"
    assert on_cloud.blocks[1].text == "For example:"           # demoted heading kept as text
    assert [p.number for p in st.path(on_cloud)] == ["3", "3.1", "3.1.1"]

    on_premise = st.get("3.1.2")                              # recovered from a text item
    assert on_premise.title == "On Premise"
    assert [b.text for b in on_premise.blocks] == ["Refer to the Program Directory.", "1. Unzip the media file"]
    assert on_premise.page_start == 9

    uninstall = st.get("4")
    assert [b.kind for b in uninstall.blocks] == ["code", "table"]
    assert uninstall.blocks[1].text == "| Param | Description |\n| --- | --- |\n| indexdepth | Default: 6 |"

    assert st.get("10") is None                               # Document History dropped
    assert st.unmatched_toc == []
    assert "1. Unzip the media file" in st.demoted_headings
    assert "Best Practices" in st.removed_lines
    assert not any("Confidential" in b.text for s in st.sections for b in s.blocks)


def test_heading_without_number_takes_toc_number_and_repeated_titles_advance():
    toc = [["3.1.", "Prerequisites ......", "8"],
           ["4.1.2.", "Post Installation Task ......", "30"],
           ["4.2.4.", "Post Installation Task ......", "38"]]
    st = build_structure([
        E("toc", "", 2, rows=toc),
        E("heading", "Prerequisites", 8),
        E("text", "a", 8),
        E("heading", "Post Installation Task", 30),
        E("text", "b", 30),
        E("heading", "Post Installation Task", 38),
        E("text", "c", 38),
    ], doc_type="pdf", n_pages=40)
    assert [s.number for s in st.sections] == ["3.1", "4.1.2", "4.2.4"]


def test_repeated_lines_removed():
    elements = [E("text", f"Page {p} of 10", p) for p in range(1, 11)] + \
               [E("heading", "1. Introduction", 1), E("text", "body", 1)]
    st = build_structure(elements, doc_type="pdf", n_pages=10)
    assert [b.text for s in st.sections for b in s.blocks] == ["body"]


def test_documents_without_toc_use_all_headings():
    st = build_structure([E("heading", "Overview", 1), E("text", "a", 1),
                          E("heading", "Details", 2), E("text", "b", 2)], doc_type="docx", n_pages=2)
    assert [(s.number, s.title) for s in st.sections] == [("1", "Overview"), ("2", "Details")]


def test_pptx_one_section_per_slide():
    st = build_structure([
        E("heading", "Product Designer (PDS)", 1),
        E("text", "© Acme Corp | Confidential Information | 23/07/2021 | ref.: ACME_Document_2021", 1),
        E("list", "Functional Overview", 1, depth=2),
        E("heading", "Proprietary Disclosure Statement", 2), E("text", "The information ...", 2),
        E("heading", "agenda", 3), E("list", "Introduction", 3, depth=2),
        E("heading", "Key Functionality", 4), E("text", "Feature A", 4),
        E("heading", "Sub point", 4), E("text", "detail", 4),
        E("text", "Content on a slide without title", 5),
    ], doc_type="pptx", n_pages=5)
    assert [(s.number, s.title) for s in st.sections] == [
        ("1", "Product Designer (PDS)"), ("4", "Key Functionality"), ("5", "Content on a slide without title")]
    assert [b.text for b in st.get("5").blocks] == ["Content on a slide without title"]   # its only line stays
    assert [b.text for b in st.get("4").blocks] == ["Feature A", "Sub point", "detail"]
    assert st.get("1").blocks[0].text == "- Functional Overview"


def test_headings_merged_into_paragraphs_are_split():
    toc = [["2.", "Installation ......", "10"],
           ["3.", "Configuration Files Optimal Performance Recommendations for", "14 config.xml Configurations ......", "15"],
           ["3.1.", "The config.xml File ......", "14"],
           ["", "Index Computation Configuration ......", "20"],
           ["", "Best Practices for Team Assignment ......", "24"]]
    st = build_structure([
        E("toc", "", 3, rows=toc),
        E("heading", "2. Installation", 10), E("text", "Install it.", 10),
        E("heading", "3. Configuration Files", 14),
        E("heading", "3.1. The config.xml File", 14),
        E("text", "Optimal Performance Recommendations for config.xml Configurations When configuring the "
                  "config.xml file, follow the recommendations listed below.", 15),
        E("heading", "Index Computation Configuration", 20), E("text", "Index text.", 20),
        E("text", "Best Practices for Team Assignment Typically, a person is "
                  "assigned to several teams, most of which belong to a single department.", 24),
    ], doc_type="pdf", n_pages=30)
    assert [s.number for s in st.sections] == ["2", "3", "3.1", "3.1.1", "3.1.2", "3.1.3"]
    first = st.get("3.1.1")
    assert first.title == "Optimal Performance Recommendations for config.xml Configurations"
    assert first.blocks[0].text.startswith("When configuring the config.xml file")
    assert st.get("3.1.2").title == "Index Computation Configuration"
    assert st.get("3.1.3").blocks[0].text.startswith("Typically, a person")
    assert st.unmatched_toc == []


def test_cover_title_cannot_steal_a_toc_number():
    toc = [["1.", "Introduction ......", "4"], ["2.", "Acme Launcher Overview ......", "4"]]
    st = build_structure([
        E("heading", "Acme Launcher Services", 1),
        E("heading", "Executive Summary", 2), E("text", "Applies to R2015x and above.", 2),
        E("toc", "", 3, rows=toc),
        E("heading", "1. Introduction", 4), E("text", "Intro.", 4),
        E("heading", "2. Acme Launcher Overview", 4), E("text", "Overview.", 4),
    ], doc_type="pdf", n_pages=10)
    assert [(s.number, s.title) for s in st.sections] == [
        ("0", "Executive Summary"), ("1", "Introduction"), ("2", "Acme Launcher Overview")]


def test_pptx_consecutive_slides_with_same_title_merge():
    st = build_structure([
        E("heading", "Configured product development", 16), E("text", "step one", 16),
        E("heading", "Configured product development", 17), E("text", "step two", 17),
        E("heading", "CAD Visualization", 18), E("text", "viz", 18),
        E("heading", "Configured product development", 19), E("text", "not consecutive", 19),
    ], doc_type="pptx", n_pages=20)
    assert [(s.number, s.page_start, s.page_end) for s in st.sections] == [("16", 16, 17), ("18", 18, 18),
                                                                          ("19", 19, 19)]
    assert [b.text for b in st.get("16").blocks] == ["step one", "step two"]


def test_headings_labelled_as_list_code_or_noisy_text_are_recovered():
    toc = [["4.7.", "SocialServer ......", "59"],
           ["4.7.1.", "Installing Search Engine ......", "59"],
           ["4.7.2.", "Installing SocialServer Foundation, SocialServer Video Convertor, and SocialServer Index on", "", ""],
           ["", "a Single Machine ......", "62"],
           ["", "Upgrade Post-Processing ......", "42"],
           ["", "Supplier Management (SPM) ...... Quote Management (QTM) ......", "55"],
           ["", "Project Management (PRJ) ......", "54"],
           ["", "Blank Page When Setting Up Single Sign-On Delegation ......", "33"]]
    st = build_structure([
        E("toc", "", 3, rows=toc),
        E("list", "AuthServer", 5), E("list", "SocialServer", 5),          # service list: must not start sections
        E("heading", "4.7. SocialServer", 59), E("text", "Social text.", 59),
        E("heading", "4.7.1. Installing Search Engine", 59), E("text", "Search text.", 59),
        E("list", "Installing SocialServer Foundation, SocialServer Video Convertor, and SocialServer Index on a Single Machine", 62),
        E("text", "Social install text.", 62),
        E("code", "Upgrade Post-Processing", 63), E("text", "Post text.", 63),
        E("text", "Management Project Management (PRJ)", 64), E("text", "PRG text.", 64),
        E("text", "Quote Management (QTM)", 65), E("text", "SRC text.", 65),
        E("list", "Blank Page When Setting Up Single Sign-On Delegation Resolve this error as "
                  "described in the table below.", 66),
    ], doc_type="pdf", n_pages=70)
    titles = {s.number: s.title for s in st.sections}
    assert titles["4.7.2"] == "Installing SocialServer Foundation, SocialServer Video Convertor, and SocialServer Index on a Single Machine"
    # 4.7.2 took its number from the TOC, so the unnumbered headings that follow are its children.
    assert titles["4.7.2.1"] == "Upgrade Post-Processing"
    assert titles["4.7.2.2"] == "Project Management (PRJ)"
    assert titles["4.7.2.3"] == "Quote Management (QTM)"
    assert titles["4.7.2.4"] == "Blank Page When Setting Up Single Sign-On Delegation"
    assert st.get("4.7.2.4").blocks[0].text.startswith("Resolve this error")
    assert "SocialServer" not in [s.title for s in st.sections if s.number not in ("4.7",)]
    assert any(b.text == "- AuthServer\n- SocialServer" for b in st.get("0").blocks)


def test_heading_cut_short_in_body_takes_full_toc_title():
    toc = [["10.2.", "CoreServer And Other Apps Installation ......", "30"],
           ["", "Upgrading from Legacy Level before Version 13 ......", "33"],
           ["", "Upgrading from Version 13 or Later Legacy Levels ......", "33"]]
    st = build_structure([
        E("toc", "", 3, rows=toc),
        E("heading", "10.2. CoreServer And Other Apps Installation", 30), E("text", "x", 30),
        E("heading", "Upgrading from Legacy Level before Version 13", 33), E("text", "y", 33),
        E("heading", "Upgrading from Version 13 or Later", 33), E("text", "Legacy Levels", 33),
        E("text", "update to the latest fix pack first.", 33),
    ], doc_type="pdf", n_pages=40)
    assert st.get("10.2.2").title == "Upgrading from Version 13 or Later Legacy Levels"


def test_numbered_heading_merged_into_paragraph_has_clean_title():
    toc = [["7.3.", "Creating Causality Relationships ......", "42"],
           ["7.3.5.", "Set the Risk, Impact and Priority Values ......", "47"]]
    st = build_structure([
        E("toc", "", 3, rows=toc),
        E("heading", "7.3. Creating Causality Relationships", 42), E("text", "x", 42),
        E("text", "7.3.5. Set the Risk, Impact and Priority Values Select a risk from the Analysis section "
                  "and input the impact and priority values for the risk.", 47),
    ], doc_type="pdf", n_pages=50)
    section = st.get("7.3.5")
    assert section.title == "Set the Risk, Impact and Priority Values"
    assert section.blocks[0].text.startswith("Select a risk")


def test_list_items_repeating_later_toc_titles_do_not_take_their_numbers():
    toc = [["2.1.", "Workflow ......", "6"],
           ["2.2.", "Setup and Export the Metadata ......", "7"],
           ["2.2.3.", "Configure the AuthServer SAML Metadata ......", "9"],
           ["4.10.1.", "Installing CoreServer Derived Data Server ......", "88"]]
    st = build_structure([
        E("toc", "", 3, rows=toc),
        E("heading", "2.1. Workflow", 6),
        E("list", "Configure the AuthServer SAML metadata to define the service provider parameters.", 6),
        E("list", "CoreServer Derived Data Server", 6),              # tail of the 4.10.1 TOC title
        E("heading", "2.2. Setup and Export the Metadata", 7), E("text", "x", 7),
        E("heading", "2.2.3. Configure the AuthServer SAML Metadata", 9), E("text", "y", 9),
    ], doc_type="pdf", n_pages=90)
    assert [s.number for s in st.sections] == ["2.1", "2.2", "2.2.3"]
    assert st.get("2.2.3").page_start == 9
    assert len(st.get("2.1").blocks[0].text.splitlines()) == 2       # both steps stay in the workflow


def test_numbered_faq_headings_do_not_become_chapters():
    toc = [["5.", "Support ......", "29"], ["5.1.", "Frequently Asked Questions (FAQ) ......", "30"],
           ["5.2.", "Common Issues ......", "33"], ["6.", "References ......", "40"]]
    st = build_structure([
        E("toc", "", 3, rows=toc),
        E("heading", "5. Support", 29), E("heading", "5.1. Frequently Asked Questions (FAQ)", 30),
        E("heading", "5. Can I assign projects to pending users?", 31), E("text", "No.", 31),
        E("heading", "6. Can I use self-signed certificates?", 31), E("text", "Yes.", 31),
        E("heading", "7. Which Identity Providers are supported?", 31), E("text", "Any SAML 2.0 IdP.", 31),
        E("heading", "5.2. Common Issues", 33), E("text", "z", 33),
        E("heading", "6. References", 40), E("text", "links", 40),
    ], doc_type="pdf", n_pages=40)
    assert [s.number for s in st.sections] == ["5", "5.1", "5.2", "6"]
    assert st.get("6").title == "References"
    assert "self-signed" in " ".join(b.text for b in st.get("5.1").blocks)


def test_numbered_heading_with_garbled_or_missing_toc_entry_is_kept():
    toc = [["3.", "Configuration Files ......", "14"],
           ["3.2.", "3.3. Server Roles Best Practices to Implement ......", "23"],
           ["4.", "Next Chapter ......", "30"]]
    st = build_structure([
        E("toc", "", 3, rows=toc),
        E("heading", "3. Configuration Files", 14), E("text", "a", 14),
        E("heading", "3.1. Not In The TOC", 20), E("text", "b", 20),       # TOC lists no 3.1
        E("heading", "3.2. Server Roles", 23), E("text", "c", 23),        # TOC title garbled
        E("heading", "4. Next Chapter", 30), E("text", "d", 30),
    ], doc_type="pdf", n_pages=40)
    assert [s.number for s in st.sections] == ["3", "3.1", "3.2", "4"]


def test_table_markdown_escapes_pipes():
    assert table_markdown([["a|b", "c"], ["1", "2"]]) == "| a\\|b | c |\n| --- | --- |\n| 1 | 2 |"
    assert table_markdown([]) == ""


def toc(page, *rows):
    return E("toc", page=page, rows=[[r] for r in rows])


def test_table_mislabelled_as_toc_late_in_the_document_is_a_table():
    elements = [E("heading", "Executive Summary", 1), E("text", "Summary text.", 1),
                E("heading", "Contents", 2), toc(2, "1. Introduction ..... 3", "2. Installation ..... 5"),
                E("heading", "1. Introduction", 3), E("text", "Intro text.", 3),
                E("heading", "2. Installation", 5), E("text", "Install text.", 5),
                toc(40, "Port | 443", "Host | acme")]          # an ordinary table Docling called a TOC
    st = build_structure(elements, doc_type="pdf", n_pages=60)
    assert [s.number for s in st.sections] == ["0", "1", "2"]
    assert st.get("2").blocks[-1].kind == "table"             # kept, as a table of section 2
    assert st.unmatched_toc == []


def test_contents_heading_in_front_matter_does_not_drop_the_body():
    elements = [E("heading", "Contents", 2), toc(2, "Overview ..... 3"),
                E("text", "Body text on page 3.", 3), E("text", "More body text.", 4)]
    st = build_structure(elements, doc_type="pdf", n_pages=4)
    assert [b.text for s in st.sections for b in s.blocks] == ["Body text on page 3.", "More body text."]


def test_unnumbered_toc_headings_become_top_level_sections():
    elements = [E("heading", "Contents", 2), toc(2, "Introduction ..... 3", "Pre-requisites ..... 4",
                                                 "Installation ..... 5"),
                E("heading", "Introduction", 3), E("text", "Intro.", 3),
                E("heading", "Pre-requisites", 4), E("text", "Java 17.", 4),
                E("heading", "Note:", 4), E("text", "A note.", 4),     # not in the TOC: stays text
                E("heading", "Installation", 5), E("text", "Run setup.", 5)]
    st = build_structure(elements, doc_type="pdf", n_pages=5)
    assert [(s.number, s.title, s.level) for s in st.sections] == [
        ("1", "Introduction", 1), ("2", "Pre-requisites", 1), ("3", "Installation", 1)]
    assert [b.text for b in st.get("2").blocks] == ["Java 17.", "Note:", "A note."]


def test_pptx_chapter_title_repeated_over_slides_is_kept():
    # "Storage Systems" titles slides 3-6 (and is listed on the agenda): a chapter, not a running header;
    # the footer repeated on every slide is plain text and still goes.
    footer = "Acme Corp | Internal use"
    elements = [E("heading", "Agenda", 1), E("text", "Storage Systems", 1), E("text", footer, 1),
                E("heading", "Guidelines", 2), E("text", "Keep connections low.", 2), E("text", footer, 2)]
    for slide, body in zip(range(3, 7), ("RAID layout", "JBOD placement", "SAN striping", "Veritas option"), strict=True):
        elements += [E("heading", "Storage Systems", slide), E("text", body, slide), E("text", footer, slide)]
    s = build_structure(elements, doc_type="pptx", n_pages=6)
    assert [(sec.number, sec.title) for sec in s.sections] == [("2", "Guidelines"), ("3", "Storage Systems")]
    assert sum(1 for sec in s.sections[1:] for b in sec.blocks) == 4 and s.removed_lines == {footer: 6}


def test_pptx_slides_without_title_placeholder_take_their_first_line():
    """Titles typed in a text box: the first line titles the slide, consecutive slides of one chapter merge,
    and lines that are commands, paths, URLs, code, sentences or a lowercase label keep "Slide N"."""
    st = build_structure([
        E("text", "13. Installing & Configuring 3DSpace", 3), E("text", "Preparation:", 3), E("text", "step a", 3),
        E("text", "13. Installing & Configuring 3DSpace", 4), E("text", "step b", 4),
        E("list", "Configuring the database:", 5), E("text", "create users", 5),
        E("text", "[x3ds@host ~]$ ./StartInstall.sh", 6), E("text", "output", 6),
        E("text", "https://host:443/3dswym/#home", 7), E("text", "page", 7),
        E("text", "Click on Files for Microsoft Windows.", 8), E("text", "picture", 8),
        E("text", "root", 9), E("text", "# useradd x3ds", 9),
        E("text", "USE [master];", 10), E("text", "GO", 10),
        E("text", "/app/DassaultSystemes/R2022x/3DSpace/logs", 11), E("text", "log", 11),
    ], doc_type="pptx", n_pages=11)
    assert [(s.number, s.page_start, s.page_end, s.title) for s in st.sections] == [
        ("3", 3, 4, "13. Installing & Configuring 3DSpace"), ("5", 5, 5, "Configuring the database"),
        ("6", 6, 6, "Slide 6"), ("7", 7, 7, "Slide 7"), ("8", 8, 8, "Slide 8"), ("9", 9, 9, "Slide 9"),
        ("10", 10, 10, "Slide 10"), ("11", 11, 11, "Slide 11")]
    assert [b.text for b in st.get("3").blocks] == ["Preparation:", "step a", "step b"]



def course_pdf_elements():
    """A slide deck exported to PDF: one heading and one paragraph per slide, 46 slides."""
    elements = []
    for page in range(1, 47):
        # Distinct words per slide: titles or lines that differ only by a number count as one chapter (titles)
        # or as a running footer (text), so 10 -> "ba".
        words = "".join(chr(97 + int(d)) for d in str(page))
        elements += [E("heading", f"Topic {words} overview", page),
                     E("text", f"Details about {words} configuration.", page)]
    return elements


@pytest.fixture
def misread_pdf_rules(monkeypatch):
    """The PDF rules as they misread the real course PDFs: every topic numbered under one chapter (TD-32)."""
    real = structure.build_structure

    def fake(elements, doc_type, n_pages, furniture=None):
        """A misread tree for PDFs; the real rules otherwise."""
        if doc_type != "pdf":
            return real(elements, doc_type=doc_type, n_pages=n_pages, furniture=furniture)
        sections = [Section("3", "Caution / Warning", 1, None, 3)]
        sections += [Section(f"3.{i}", f"Topic {i}", 2, "3", i) for i in range(1, n_pages)]
        return Structure(sections, [], [], [], {})
    monkeypatch.setattr(structure, "build_structure", fake)


def test_a_slide_deck_exported_to_pdf_is_built_like_a_deck(misread_pdf_rules):
    """Landscape pages + 45 topics under one chapter: rebuilt as one section per slide (TD-32)."""
    st = structure_for(course_pdf_elements(), set(), doc_type="pdf", n_pages=46, landscape=1.0)
    assert st.layout == "slides (landscape PDF)"
    assert all(x.parent is None for x in st.sections)
    assert st.get("10").title == "Topic ba overview" and st.get("10").page_start == 10


def test_portrait_or_well_built_pdfs_keep_the_pdf_rules(misread_pdf_rules):
    """Portrait pages keep the PDF rules even when misread; a landscape PDF with a sound tree too; decks are 'slides'."""
    elements = course_pdf_elements()
    assert structure_for(elements, set(), doc_type="pdf", n_pages=46, landscape=0.0).layout == "document"
    assert structure_for(elements[:12], set(), doc_type="pdf", n_pages=6, landscape=1.0).layout == "document"
    assert structure_for(elements[:12], set(), doc_type="pptx", n_pages=6, landscape=1.0).layout == "slides"
