import pytest

from rag_ingest.chunk import chunk_document, window_words
from rag_ingest.parse import Document, Section, parse_html, parse_markdown

MD = """# Plan Guide

Intro text before any section.

## Claims

Submit within 180 days.

### Claim Forms

Use form C-1.

## Claims

Second section with the same heading.
"""


def test_markdown_sections_follow_headings() -> None:
    title, sections = parse_markdown(MD)
    assert title == "Plan Guide"
    assert [(s.heading, s.level) for s in sections] == [
        ("Plan Guide", 1),
        ("Claims", 2),
        ("Claim Forms", 3),
        ("Claims", 2),
    ]
    assert sections[2].text == "Use form C-1."


def test_html_parsing_skips_title_and_script() -> None:
    html = (
        "<html><head><title>T</title><script>var x=1;</script></head><body>"
        "<h1>Policy</h1><h2>Scope</h2><p>Applies to <b>all</b> members.</p></body></html>"
    )
    title, sections = parse_html(html)
    assert title == "Policy"
    assert sections[0].heading == "Scope" and sections[0].text == "Applies to all members."


def test_window_overlap() -> None:
    words = [str(i) for i in range(25)]
    windows = window_words(words, max_words=10, overlap=3)
    assert windows[0] == words[:10] and windows[1][:3] == words[7:10]
    assert windows[-1][-1] == "24"
    assert all(len(w) <= 10 for w in windows)
    with pytest.raises(ValueError):
        window_words(words, 5, 5)


def test_chunks_never_cross_sections_and_ids_are_stable() -> None:
    long_text = " ".join(f"w{i}" for i in range(50))
    doc = Document(
        "guide",
        "guide.md",
        "md",
        "Guide",
        [
            Section("Claims", 2, long_text),
            Section("Billing", 2, "Pay monthly."),
            Section("Claims", 2, "Duplicate heading."),
        ],
    )
    chunks = chunk_document(doc, max_words=20, overlap=5)
    ids = [c.chunk_id for c in chunks]
    assert ids[:3] == ["guide::claims::0", "guide::claims::1", "guide::claims::2"]
    assert "guide::billing::0" in ids and "guide::claims::0-2" in ids
    assert len(set(ids)) == len(ids)
    assert all(c.text.startswith(c.heading + "\n") for c in chunks)
    billing = next(c for c in chunks if c.heading == "Billing")
    assert "w49" not in billing.text
    # editing another section does not change these ids
    doc.sections[1] = Section("Billing", 2, "Pay quarterly now.")
    assert [c.chunk_id for c in chunk_document(doc, 20, 5)] == ids


def test_pdf_headings_are_recovered() -> None:
    from rag_ingest.parse import parse_file

    from .conftest import DOCS

    pdf = DOCS / "dental-rider.pdf"
    if not pdf.exists():
        pytest.skip("PDF sample not generated (reportlab missing)")
    doc = parse_file(pdf, DOCS)
    assert doc.doc_type == "pdf" and doc.title == "Dental Rider"
    assert [s.heading for s in doc.sections] == ["Preventive Care", "Major Services"]
    assert "12-month waiting period" in doc.sections[1].text
