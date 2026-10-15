"""Parse Markdown, HTML and PDF files into a common heading-structured document."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

SUPPORTED = {".md": "md", ".markdown": "md", ".html": "html", ".htm": "html", ".pdf": "pdf"}
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Section:
    heading: str
    level: int
    text: str


@dataclass
class Document:
    doc_id: str
    path: str
    doc_type: str
    title: str
    sections: list[Section] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(f"{s.heading}\n{s.text}" for s in self.sections)

    @property
    def content_hash(self) -> str:
        """Hash of the extracted text, so byte-level noise (e.g. PDF metadata) is ignored."""
        return hashlib.sha256(f"{self.title}\n{self.full_text}".encode()).hexdigest()


def _sections_from_lines(lines: list[tuple[int, str]]) -> tuple[str, list[Section]]:
    """lines: (heading level or 0 for body text, text). Level-1 heading becomes the title."""
    title, sections = "", []
    current: Section | None = None
    for level, text in lines:
        if level == 1 and not title:
            title = text
        elif level:
            current = Section(text, level, "")
            sections.append(current)
        elif text.strip():
            if current is None:
                current = Section(title or "Introduction", 1, "")
                sections.append(current)
            current.text = f"{current.text} {text.strip()}".strip()
    return title, [s for s in sections if s.text]


def parse_markdown(text: str) -> tuple[str, list[Section]]:
    lines = []
    for raw in text.splitlines():
        match = _MD_HEADING.match(raw)
        lines.append((len(match.group(1)), match.group(2)) if match else (0, raw))
    return _sections_from_lines(lines)


class _HeadingHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[tuple[int, str]] = []
        self._level = 0
        self._skip = 0
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "title"}:
            self._skip += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._level = int(tag[1])
        elif tag in {"p", "li", "br", "div"}:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "title"}:
            self._skip = max(0, self._skip - 1)
        elif re.fullmatch(r"h[1-6]|p|li|div", tag):
            self._flush()

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._buffer.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._buffer).split())
        if text:
            self.lines.append((self._level, text))
        self._buffer, self._level = [], 0


def parse_html(text: str) -> tuple[str, list[Section]]:
    parser = _HeadingHTMLParser()
    parser.feed(text)
    parser.close()
    return _sections_from_lines(parser.lines)


def _looks_like_heading(line: str) -> bool:
    words = line.split()
    return 0 < len(words) <= 8 and line[0].isupper() and not line.rstrip().endswith((".", ",", ";"))


def parse_pdf(path: Path) -> tuple[str, list[Section]]:
    """Text-layer PDFs only. Short, title-like lines without end punctuation are headings."""
    from pypdf import PdfReader

    raw_lines = []
    for page in PdfReader(str(path)).pages:
        raw_lines += [ln.strip() for ln in (page.extract_text() or "").splitlines() if ln.strip()]
    lines: list[tuple[int, str]] = []
    for i, line in enumerate(raw_lines):
        lines.append((1 if i == 0 else 2 if _looks_like_heading(line) else 0, line))
    return _sections_from_lines(lines)


def parse_file(path: Path, root: Path) -> Document:
    doc_type = SUPPORTED[path.suffix.lower()]
    if doc_type == "pdf":
        title, sections = parse_pdf(path)
    else:
        text = path.read_text(encoding="utf-8")
        title, sections = (parse_markdown if doc_type == "md" else parse_html)(text)
    doc_id = path.relative_to(root).with_suffix("").as_posix()
    return Document(doc_id, path.relative_to(root).as_posix(), doc_type, title or doc_id, sections)


def discover(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED and p.is_file())
