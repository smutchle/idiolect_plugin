from collections import Counter

import extract as ex

PAPER_MD = """---
title: x
---
# A Study of Things

Jane Doe, Department of Statistics, University of Somewhere, jane@uni.edu

## Abstract

We study things. Things are studied here in some detail.

## 1. Introduction

Prior work has looked at this problem [3]. Others disagree (Smith et al., 2020). The
effect was large.¹ Our goal is to under-
stand why.

Figure 1: A plot of the things.

$$ y = \\alpha + \\beta x $$

## 2. Methods

We collected 120 samples and measured each one twice.

- first item in a list
- second item in a list

| a | b |
|---|---|
| 1 | 2 |

## Acknowledgments

We thank our funders.

## References

[1] A. Author. Some paper. 2019.
[2] B. Author. Another paper. 2020.
"""


def test_markdown_academic_stripping(tmp_path):
    p = tmp_path / "paper.md"
    p.write_text(PAPER_MD)
    doc = ex.extract(p)
    text = doc.text
    assert "Some paper" not in text and doc.stripped["references"] >= 2
    assert "funders" not in text and doc.stripped["acknowledgments"] >= 1
    assert "jane@uni.edu" not in text and doc.stripped["author_affiliation"] >= 1
    assert "A plot of the things" not in text and doc.stripped["caption"] == 1
    assert "alpha" not in text and doc.stripped["equation"] == 1
    assert doc.stripped["table"] == 1
    assert "understand why" in text                 # dehyphenated across the line break
    assert "¹" not in text and doc.stripped["footnote_marker"] == 1
    assert doc.citations["numeric"] == 1 and doc.citations["author_year"] == 1
    assert "[3]" not in text and "Smith" not in text
    roles = {b.text[:12]: b.role for b in doc.paragraphs}
    assert roles["Prior work h"] == "intro"
    assert roles["We collected"] == "methods"
    lists = [b for b in doc.blocks if b.kind == "list"]
    assert len(lists) == 2


def test_txt_wrapped_lines_and_headings(tmp_path):
    p = tmp_path / "essay.txt"
    p.write_text("THE TITLE\n\nThis is a wrapped\nparagraph of text that goes\non for a while.\n\n"
                 "Another para-\ngraph here.\n")
    doc = ex.extract(p)
    assert [b.kind for b in doc.blocks] == ["heading", "para", "para"]
    assert doc.paragraphs[0].text == "This is a wrapped paragraph of text that goes on for a while."
    assert doc.paragraphs[1].text == "Another paragraph here."


def test_docx_headings_captions_lists(tmp_path):
    import docx
    d = docx.Document()
    d.add_heading("Results", level=1)
    d.add_paragraph("The model improved accuracy by 4.2% on the held-out set.")
    d.add_paragraph("Figure 2. Accuracy by epoch.", style="Caption")
    d.add_paragraph("A bullet point", style="List Bullet")
    d.add_heading("References", level=1)
    d.add_paragraph("[1] Someone. A paper. 2001.")
    p = tmp_path / "x.docx"
    d.save(p)
    doc = ex.extract(p)
    assert doc.paragraphs[0].role == "results"
    assert "Accuracy by epoch" not in doc.text and doc.stripped["caption"] == 1
    assert any(b.kind == "list" for b in doc.blocks)
    assert "Someone" not in doc.text


def test_pdf_headers_footers_and_hyphenation(tmp_path):
    import pymupdf as fitz
    pdf = fitz.open()
    body = ("This is body text that continues across lines and includes a hyphen-\n"
            "ated word in the middle of the paragraph so that we can test joining.")
    for i in range(3):
        page = pdf.new_page()
        page.insert_text((72, 30), "Journal of Testing  Vol. 3", fontsize=9)
        page.insert_textbox(fitz.Rect(72, 100, 520, 300), body, fontsize=11)
        page.insert_text((300, 820), str(i + 1), fontsize=9)
    p = tmp_path / "x.pdf"
    pdf.save(p)
    doc = ex.extract(p)
    assert "Journal of Testing" not in doc.text
    assert doc.stripped["header_footer"] >= 5
    assert "hyphenated word" in doc.text
    assert len(doc.paragraphs) == 3


def test_collect_files_recurses_and_filters(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.md").write_text("x")
    (tmp_path / "two.txt").write_text("y")
    (tmp_path / "skip.csv").write_text("z")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "three.md").write_text("w")
    files = ex.collect_files([tmp_path])
    assert sorted(f.name for f in files) == ["one.md", "two.txt"]


def test_equation_and_table_detectors():
    assert ex.is_equation("E = mc^2 + \\frac{1}{2} m v^2")
    assert not ex.is_equation("The energy of the system was conserved in every trial we ran.")
    assert ex.is_table_row("Model A 0.81 0.77 0.79 12.3")
    assert not ex.is_table_row("We ran the model three times and averaged results.")


def test_role_heuristics_without_headings():
    blocks = [ex.Block(f"Paragraph {i} text.") for i in range(10)]
    blocks[4].text = "We found that results were significantly better (p < 0.05)."
    ex.tag_roles(blocks)
    roles = Counter(b.role for b in blocks)
    assert blocks[0].role == "intro"
    assert blocks[-1].role == "discussion"
    assert blocks[4].role == "results"
    assert roles["other"] >= 5
